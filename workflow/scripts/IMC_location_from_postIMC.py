import copy
import cv2
from image_utils import get_image_shape, readimage_crop, saveimage_tile
import json
import logging, traceback
import logging_utils
import numpy as np
import pycpd
from scipy.stats import gaussian_kde
import shapely.affinity
import shapely.geometry
import sklearn
from sklearn.neighbors import KDTree
import SimpleITK as sitk
import sys,os
from utils import setNThreads, snakeMakeMock


if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["IMC_pixelsize"] = 1
    snakemake.params["microscopy_pixelsize"] = 0.22537
    snakemake.params["polygon_expand"] = 10
    snakemake.params["preprocessing_type"] = "saturation" 
    snakemake.params["use_pycpd"] = True
    snakemake.params["do_registration"] = True 
    snakemake.input["postIMC"] =  "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/postIMC/test_combined_postIMC.ome.tiff"
    snakemake.input["postIMC_geojson_file"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/IMC_location/test_combined_IMC_mask_on_postIMC_B1.geojson"
    snakemake.input["IMC"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/IMC/Cirrhosis-TMA-5_New_Detector_002.tiff"
    snakemake.output["postIMC_geojson_file"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/IMC_location/testout.geojson"
    snakemake.output["IMC_location_extraction_image"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/IMC_location/testout.geojson"

    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")

# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
# print all snakemake defined threads, params, input and output
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# params
microscopy_pixelsize = snakemake.params["microscopy_pixelsize"]
IMC_pixelsize = snakemake.params["IMC_pixelsize"]
polygon_expand = snakemake.params["polygon_expand"]
preprocessing_type = snakemake.params["preprocessing_type"]
use_pycpd = snakemake.params["use_pycpd"]
assert preprocessing_type in ["saturation", "grayscale"]
# input
postIMC_file = snakemake.input["postIMC"] 
postIMC_geojson_file = snakemake.input["postIMC_geojson_file"]
IMC_file = snakemake.input["IMC"]
# output
output_file = snakemake.output["postIMC_geojson_file"]
logimagefile = snakemake.output["IMC_location_extraction_image"]

if not snakemake.params["do_registration"]:
    logging.info("Registration is turned off. Copying input to output")
    os.system(f"cp {postIMC_geojson_file} {output_file}")
    os.system(f"touch {logimagefile}")
    sys.exit(0)


# get dimension of IMC image
imcsize = get_image_shape(IMC_file)[1:]

logging.info("Open IMC location and extract bounding box")
postIMC_geojson = json.load(open(postIMC_geojson_file, "r"))
postIMC_geojson_polygon = shapely.geometry.shape(postIMC_geojson['geometry'])
postIMC_geojson_polygon = postIMC_geojson_polygon.buffer(polygon_expand/microscopy_pixelsize)
bbox_raw = postIMC_geojson_polygon.bounds
bbox_raw = (int(bbox_raw[1]/IMC_pixelsize),int(bbox_raw[0]/IMC_pixelsize),int(bbox_raw[3]/IMC_pixelsize),int(bbox_raw[2]/IMC_pixelsize))

logging.info(f"unclipped bbox: {bbox_raw}")
imsh = get_image_shape(postIMC_file)
logging.info(f"clip bounding box to: {imsh}")
bbox = [
    np.clip(bbox_raw[0],0,imsh[0]),
    np.clip(bbox_raw[1],0,imsh[1]),
    np.clip(bbox_raw[2],0,imsh[0]),
    np.clip(bbox_raw[3],0,imsh[1])
]
logging.info(f"bbox final: {bbox}")

logging.info(f"Read image")
postIMCcut = readimage_crop(postIMC_file, bbox)


def command_iteration_single(method):
    if method.GetOptimizerIteration()%10==0:
        print(
            f"{method.GetOptimizerIteration():3} "
            + f"= {method.GetMetricValue():10.8f} "
            + f": {method.GetOptimizerPosition()}"
        )

def command_iteration(method) :
        print(
            f"{method.GetOptimizerIteration():3} "
            + f"= {method.GetMetricValue():10.8f} "
            + f": {method.GetOptimizerPosition()}"
        )

def command_multi_iteration(method) :
    if method.GetCurrentLevel() > 0:
        print("Optimizer stop condition: {0}".format(method.GetOptimizerStopConditionDescription()))
        print(" Iteration: {0}".format(method.GetOptimizerIteration()))
        print(" Metric value: {0}".format(method.GetMetricValue()))

    print("--------- Resolution Changing ---------")


def resample_image(moving, fixed, transform):
    resampler = sitk.ResampleImageFilter()
    resampler.SetDefaultPixelValue(255)
    resampler.SetTransform(transform)
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    tmp1 = resampler.Execute(moving)
    tmp1 = sitk.GetArrayFromImage(tmp1)
    return tmp1

def gamma_correction(img,gamma):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    return cv2.LUT(img, lookUpTable)

# from: https://stackoverflow.com/questions/65791233/auto-selection-of-gamma-value-for-image-brightness
logging.info(f"Adjust Gamma")
val = cv2.cvtColor(postIMCcut, cv2.COLOR_BGR2HSV)[:,:,2]
gamma = 1/ (np.log(0.5*255)/np.log(np.mean(val)))
logging.info(f"Gamma: {gamma}")
postIMCcutg = gamma_correction(postIMCcut, gamma)

logging.info(f"To HSV, extract saturation")
blur = cv2.cvtColor(postIMCcutg, cv2.COLOR_BGR2HSV)[:,:,1]

logging.info(f"Bilateral filter")
kernel = int(2/microscopy_pixelsize)
blur = cv2.bilateralFilter(blur,kernel,10,10)

logging.info(f"Threshold with Otsu")
th,blur = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

imcsize_scaled = (int(imcsize[0]/microscopy_pixelsize),int(imcsize[1]/microscopy_pixelsize))
logging.info(f"IMC dimensions to microscopy scale: {imcsize_scaled}")

yx_translation = (int((blur.shape[1]-imcsize_scaled[1])/2),int((blur.shape[0]-imcsize_scaled[0])/2))
logging.info(f"Initial translation: {yx_translation}")
imcboximg = np.zeros(blur.shape, dtype=float)
imcboximg[int(yx_translation[1]):int(yx_translation[1]+imcsize_scaled[0]),int(yx_translation[0]):int(yx_translation[0]+imcsize_scaled[1])] = 255
obs_area = np.sum(imcboximg==255)*(microscopy_pixelsize**2)

logging.info(f"IMC area on image: {obs_area}")
logging.info(f"IMC area expected: {np.prod(imcsize_scaled)*(microscopy_pixelsize**2)}")
assert obs_area == np.prod(imcsize_scaled)*(microscopy_pixelsize**2)

logging.info("Images to sitk format")
fixed = sitk.GetImageFromArray(blur.astype(float))
moving = sitk.GetImageFromArray(imcboximg.astype(float))

logging.info(f"Setup initial registration")
R = sitk.ImageRegistrationMethod()
R.SetMetricAsMeanSquares()
R.SetMetricSamplingStrategy(R.RANDOM)
R.SetMetricSamplingPercentage(0.005, seed=1234)
R.SetInterpolator(sitk.sitkNearestNeighbor)
R.SetOptimizerAsGradientDescent(
    learningRate=100, numberOfIterations=500, 
    convergenceMinimumValue=1e-7, convergenceWindowSize=5,
    estimateLearningRate=R.EachIteration
)
R.SetOptimizerScalesFromIndexShift()
R.SetInitialTransform(sitk.Euler2DTransform())
R.SetShrinkFactorsPerLevel(shrinkFactors = [8,4,2,1])
R.SetSmoothingSigmasPerLevel(smoothingSigmas = [4,2,1,0])
R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )
R.AddCommand( sitk.sitkMultiResolutionIterationEvent, lambda: command_multi_iteration(R) )

logging.info("Run registration")
transform = R.Execute(fixed, moving)


logging.info("Create mask for second registration")
halfbordersize = 5
mask = (cv2.dilate(imcboximg, np.ones((int(halfbordersize/microscopy_pixelsize),int(halfbordersize/microscopy_pixelsize)))) - cv2.erode(imcboximg, np.ones((int(halfbordersize/microscopy_pixelsize),int(halfbordersize/microscopy_pixelsize)))))>0
masksitk = sitk.GetImageFromArray(mask.astype(np.uint8))

logging.info("Setup second registration")
R = sitk.ImageRegistrationMethod()
R.SetMetricAsMeanSquares()
R.SetMetricSamplingStrategy(R.RANDOM)
R.SetMetricSamplingPercentage(0.05, seed=1234)
R.SetMetricMovingMask(masksitk)
R.SetInterpolator(sitk.sitkNearestNeighbor)
R.SetOptimizerAsRegularStepGradientDescent(learningRate=0.2, minStep=0.002, numberOfIterations=1000, gradientMagnitudeTolerance=1e-9)
init_transform_2 = sitk.Euler2DTransform()
init_transform_2.SetTranslation(np.array(transform.GetTranslation()))
init_transform_2.SetCenter(np.array(transform.GetCenter()))
init_transform_2.SetMatrix(np.array(transform.GetMatrix()))

R.SetOptimizerScalesFromIndexShift()
R.SetInitialTransform(init_transform_2)
R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration_single(R))

logging.info("Run second registration") 
transform_masked = R.Execute(fixed, moving)

logging.info("Create IMC corner points")
x1 = yx_translation[1]
x2 = yx_translation[1]+imcsize_scaled[1]
y1 = yx_translation[0]
y2 = yx_translation[0]+imcsize_scaled[0]

xytr = np.array([
    [y1,y1,y2,y2,y1],
    [x1,x2,x2,x1,x1]
    ]).T

logging.info("Transform IMC corner points")
xytrans = np.array([transform_masked.GetInverse().TransformPoint(xytr[i,:].astype(float)) for i in range(xytr.shape[0])])


if use_pycpd:
    logging.info("Prepare for pycpd registration")
    logging.info("Extract border points")
    borderpointsls = list()
    #imgls = list()
    #maskls = list()
    for side in range(4):
        # get single side of rectangle
        line = shapely.geometry.LineString(xytrans[side:(side+2),:])
        diffs=np.array([line.bounds[2]-line.bounds[0], line.bounds[3]-line.bounds[1]])
        wm = np.argmax([np.abs(diffs)])
        wi = np.argmin([np.abs(diffs)])
        # calculate angle relative to axis
        angle = np.arctan(diffs[wi]/diffs[wm])/np.pi*180
        # rotate line to have 0 angle to axis
        linerot = shapely.affinity.rotate(line, angle, origin=line.centroid)

        # create transformation
        linetr = sitk.Euler2DTransform()
        linetr.SetCenter(np.array(line.centroid.xy).flatten()[::-1])
        linetr.SetAngle(-angle/180*np.pi)

        # extract saturation
        sat = cv2.cvtColor(postIMCcutg, cv2.COLOR_BGR2HSV)[:,:,1]
        # apply transformation, i.e. aligned to axis
        blurtr = resample_image(sitk.GetImageFromArray(sat), fixed, linetr)

        # create mask for line
        linepoly = linerot.buffer(5/microscopy_pixelsize)
        points = [[y, x] for x, y in zip(*linepoly.boundary.coords.xy)]
        # create mask image
        linemask = np.ones(blur.shape, dtype=np.uint8)*255
        linemasktr = cv2.fillPoly(linemask, np.array([points]).astype(np.int32), color=0)

        # get bounding box
        contours, hierarchy = cv2.findContours((linemasktr==0).astype(np.uint8)*255,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        x,y,w,h = cv2.boundingRect(contours[0])
        # subset image
        linemasktrc = linemasktr[y:y+h,x:x+w]
        blurtrc = blurtr[y:y+h,x:x+w]

        # filter image
        kernel = int(3/microscopy_pixelsize)
        blurtrc = cv2.bilateralFilter(blurtrc,kernel,100,10)

        # apply derivative like filter perpendicular to the aligned axis
        halfkernel = np.array([1,2,3,4],dtype=float)
        if wm == 1:
            latkernel = np.zeros((9,1))
            if side == 2:
                latkernel[:4,0]=halfkernel[::-1]
                latkernel[5:,0]=-halfkernel
            else:
                latkernel[:4,0]=-halfkernel
                latkernel[5:,0]=halfkernel[::-1]
            sx = cv2.filter2D(blurtrc, cv2.CV_32F, np.ones((1,9))/9)
            sx = cv2.filter2D(sx, cv2.CV_32F, latkernel)
            sx = cv2.filter2D(sx, cv2.CV_32F, np.ones((1,3))/3)
        else:
            latkernel = np.zeros((1,9))
            if side == 3:
                latkernel[0,:4]=-halfkernel
                latkernel[0,5:]=halfkernel[::-1]
            else:
                latkernel[0,:4]=halfkernel[::-1]
                latkernel[0,5:]=-halfkernel
            sx = cv2.filter2D(blurtrc, cv2.CV_32F, np.ones((9,1))/9)
            sx = cv2.filter2D(sx, cv2.CV_32F, latkernel)
            sx = cv2.filter2D(sx, cv2.CV_32F, np.ones((3,1))/3)

        # find maximum perpendicularly to the aligned axis
        sxmax = np.max(sx, axis=wi)
        inds = np.arange(sx.shape[wi])
        indls = list()
        posls = list()
        for i in range(1,sx.shape[wm]):
            if wi==1:
                tmpinds = inds[sx[i,:]==sxmax[i]]
            else:
                tmpinds = inds[sx[:,i]==sxmax[i]]
            posls.append(tmpinds)
            indls.append(np.repeat(i, len(tmpinds)))
        posar = np.array([np.concatenate(indls), np.concatenate(posls)]).T

        # get number of neighbors
        width=int(20/microscopy_pixelsize)
        posarbuf = np.zeros((posar.shape[0]+width*2,3))
        posarbuf[width:-width,:2]=posar
        for i in range(width, posarbuf.shape[0]-width):
            tmpbo = np.logical_and(posarbuf[(i-width):(i+width),0]>posarbuf[i,0]-width/2,posarbuf[(i-width):(i+width),0]<posarbuf[i,0]+width/2)
            nsim = sum([k in np.arange(posarbuf[i,1]-2,posarbuf[i,1]+3) for k in posarbuf[(i-width):(i+width),1][tmpbo]])
            posarbuf[i,2] = nsim
        posarfilt = posarbuf[width:-width,:]

        # filter points based on maximum density
        density = gaussian_kde(posarfilt[:,2])
        xs = np.linspace(0,np.max(posarfilt[:,2]),1000)
        density.covariance_factor = lambda : .2
        density._compute_covariance()
        thr = xs[np.argmax(density(xs))]
        posarfilt = posarfilt[posarfilt[:,2]>thr,:]

        # fit line to points
        rsfit = sklearn.linear_model.RANSACRegressor(random_state=123, max_trials=10000, stop_probability=0.999, residual_threshold=1/microscopy_pixelsize, min_samples=0.75)
        rsfit.fit(posarfilt[:,0].reshape(-1,1), posarfilt[:,1].reshape(-1,1))
        # only keep inliers
        posarfilt = posarfilt[rsfit.inlier_mask_,:]

        # transform back to original image
        if wm == 0:
            posarfilt = posarfilt[:,[1,0]]
        else:
            posarfilt = posarfilt[:,:2]
        posarfilt[:,0]+=x
        posarfilt[:,1]+=y
        posarfilt = np.array([linetr.TransformPoint(posarfilt[i,:].astype(float)) for i in range(posarfilt.shape[0])])

        borderpointsls.append(posarfilt)

        # for debugging
        #blurtrcfulltr = resample_image(sitk.GetImageFromArray(sx), fixed, linetr.GetInverse())
        #linemasktrtr = resample_image(sitk.GetImageFromArray(linemasktr), fixed, linetr.GetInverse())
        #imgls.append(blurtrcfulltr)
        #maskls.append(linemasktrtr)

    # number of points per side
    npoints = np.array([len(p) for p in borderpointsls])

    logging.info("Filter points, based on corners")
    # remove points that are too far of the corners
    borderpointslsfilt = list()
    for side in range(len(borderpointsls)):
        diffs = np.abs(borderpointsls[side][0,:]-borderpointsls[side][-1,:])
        wm = np.argmax([np.abs(diffs)])
        wi = np.argmin([np.abs(diffs)])
        if side == 0 or side == 1:
            tb1 = borderpointsls[side][:,wm] > np.max(borderpointsls[side-1][:,wm])
            tb2 = borderpointsls[side][:,wm] < np.max(borderpointsls[(side+1)%4][:,wm])
        elif side == 2 or side == 3:
            tb1 = borderpointsls[side][:,wm] < np.max(borderpointsls[side-1][:,wm])
            tb2 = borderpointsls[side][:,wm] > np.max(borderpointsls[(side+1)%4][:,wm])
        tb = np.logical_and(tb1, tb2)
        borderpointslsfilt.append(borderpointsls[side][tb,:])

    # number of points per side
    npoints = np.array([len(p) for p in borderpointslsfilt])

    logging.info("Resample points")
    # resample points to have the same number of points per side
    borderpointslsfilt2 = list()
    for i in range(len(borderpointslsfilt)):
        kdt = KDTree(np.arange(npoints[i]).reshape(-1,1), leaf_size=30, metric='euclidean')
        distances, indices = kdt.query(np.linspace(0,npoints[i]-1, np.min(npoints)).reshape(-1,1), k=1, return_distance=True)
        borderpointslsfilt2.append(borderpointslsfilt[i][indices.flatten()])
    npoints = np.array([len(p) for p in borderpointslsfilt2])
    assert np.all(npoints==np.min(npoints))

    # concatenate points
    borderpoints = np.concatenate(borderpointslsfilt2, axis=0)

    logging.info("Create square points")
    squarepoints = list()
    for i in range(4):
        squarepoints.append(
        np.array([
            np.linspace(xytrans[i,0],xytrans[i+1,0],100),
            np.linspace(xytrans[i,1],xytrans[i+1,1],100)
        ]).T
        )
    squarepoints = np.concatenate(squarepoints, axis=0)


    logging.info("Run pycpd registration")

    # reg = pycpd.RigidRegistration(X=borderpoints, Y=squarepoints, w=0, s=1, max_iterations=1000, tolerance = 0.0001)
    # TY, (s_reg, R_reg, t_reg) = reg.register()
    reg = pycpd.AffineRegistration(X=borderpoints, Y=squarepoints, w=0, s=1, max_iterations=1000, tolerance = 0.0001)
    TY, (R_reg, t_reg) = reg.register()

    logging.info(f"Registered transformation: \n{R_reg}\n{t_reg}")
    logging.info(f"N iterations: {reg.iteration}, tolerance: {reg.tolerance}, max iterations: {reg.max_iterations}")


    logging.info("Transform IMC corner points")
    pycpd_transform_inverse = sitk.AffineTransform(2)
    pycpd_transform_inverse.SetTranslation(-t_reg)
    R_reg_inv = np.array([[1-(R_reg[0,0]-1),-R_reg[1,0]],[-R_reg[0,1],1-(R_reg[1,1]-1)]])
    pycpd_transform_inverse.SetMatrix(R_reg_inv.flatten())

    xytrans2 = np.array([pycpd_transform_inverse.GetInverse().TransformPoint(xytrans[i,:].astype(float)) for i in range(xytrans.shape[0])])


# to image coordinates
if use_pycpd:
    logging.info(f"Previous coordinates: \n{xytrans}")
    logging.info(f"Registered coordinates: \n{xytrans2}")
    xytransscaled = xytrans2.copy()
    xytransscaled[:,0]+=bbox[1]
    xytransscaled[:,1]+=bbox[0]
else:
    logging.info(f"Previous coordinates: \n{xytrans}")
    xytransscaled = xytrans.copy()
    xytransscaled[:,0]+=bbox[1]
    xytransscaled[:,1]+=bbox[0]

imcshape = shapely.geometry.shape({'type': 'Polygon', 'coordinates': [xytransscaled.tolist()]})
logging.info(f"IMC area after transformation: {imcshape.area*(microscopy_pixelsize**2)}")
logging.info(f"IMC area before transformation: {np.prod(imcsize_scaled)*(microscopy_pixelsize**2)}")

postIMC_geojson_out = copy.deepcopy(postIMC_geojson)
postIMC_geojson_out['geometry']['coordinates'] = [xytransscaled.tolist()]

logging.info(f"Previous coordinates: \n{np.array(postIMC_geojson['geometry']['coordinates'])}")
logging.info(f"Registered coordinates: \n{np.array(postIMC_geojson_out['geometry']['coordinates'])}")



logging.info("Create log image")
blurdraw = postIMCcutg.copy()

# create mask to reduce image size
halfbordersize = 50
mask = (cv2.dilate(imcboximg, np.ones((int(halfbordersize/microscopy_pixelsize),int(halfbordersize/microscopy_pixelsize)))) - cv2.erode(imcboximg, np.ones((int(halfbordersize/microscopy_pixelsize),int(halfbordersize/microscopy_pixelsize)))))>0
masksitk = sitk.GetImageFromArray(mask.astype(np.uint8))
maskre = resample_image(masksitk, fixed, transform_masked)
maskre = maskre>0
# non interesting regions to black
blurdraw[~np.stack([maskre,maskre,maskre],axis=2)] = 0

# draw the original and the transformed polygon
tmpxy = np.array(postIMC_geojson['geometry']['coordinates']).astype(int)
tmpxy[:,:,0]-=bbox[1]
tmpxy[:,:,1]-=bbox[0]
cv2.polylines(blurdraw, [tmpxy], True, (0,0,255), 1)
cv2.polylines(blurdraw, [xytrans.astype(int)], True, (255,0,0), 1)
if use_pycpd:
    cv2.polylines(blurdraw, [xytrans2.astype(int)], True, (0,255,0), 1)


text = "Blue: Manually drawn"
position = (int(blurdraw.shape[0]/2-100/microscopy_pixelsize), int(blurdraw.shape[0]/2)) 
color = (0, 0, 255)
cv2.putText(blurdraw, text, position, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = color, thickness=2, lineType = cv2.LINE_AA)

text = "Red: sitk registered"
position = (int(blurdraw.shape[0]/2-100/microscopy_pixelsize), int(blurdraw.shape[0]/2)+100) 
color = (255, 0, 0)
cv2.putText(blurdraw, text, position, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = color, thickness=2, lineType = cv2.LINE_AA)

if use_pycpd:
    text = "Green: sitk + pycpd registered"
    position = (int(blurdraw.shape[0]/2-100/microscopy_pixelsize), int(blurdraw.shape[0]/2)+200) 
    color = (0, 255, 0)
    cv2.putText(blurdraw, text, position, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = color, thickness=2, lineType = cv2.LINE_AA)

# reduce size (to 4-bit)
blurdraw = cv2.convertScaleAbs(cv2.convertScaleAbs(blurdraw, alpha=(15/255)),alpha=(255/15)).astype(np.uint8)

# import matplotlib.pyplot as plt
# plt.imshow(blurdraw)
# plt.scatter(xytrans[:,0], xytrans[:,1])
# plt.scatter(xytrans2[:,0], xytrans2[:,1],c="blue")
# plt.scatter(TY[:,0], TY[:,1])
# plt.scatter(borderpoints[:,0], borderpoints[:,1])
# plt.show()


logging.info(f"Save image to {logimagefile}")
saveimage_tile(blurdraw, logimagefile, 1)

logging.info(f"Save output to {output_file}")
json.dump([postIMC_geojson_out], open(output_file, "w"), indent=1)

logging.info("Finished")
