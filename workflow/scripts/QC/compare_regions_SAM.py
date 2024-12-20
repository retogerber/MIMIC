import sys,os
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "..","..","workflow","scripts","utils")))
import shapely
from sklearn.neighbors import KDTree
import numpy as np
import json
import cv2
import SimpleITK as sitk
from image_utils import readimage_crop, convert_and_scale_image, get_image_shape
from utils import setNThreads, snakeMakeMock
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["input_spacing_1"] = 1
    snakemake.params['input_spacing_2'] = 0.22537
    snakemake.params["input_spacing_IMC_location"] = 0.22537
    snakemake.params["output_spacing"] = 1
    snakemake.params['max_distance'] = 50
    snakemake.params['min_distance'] = 10
    snakemake.params["pixel_expansion"] = 501
    snakemake.input['microscopy_image_1'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/preIMC/Cirrhosis-TMA-5_New_Detector_001_transformed_on_preIMS.ome.tiff"
    snakemake.input['contours_in_1'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/preIMC/Cirrhosis-TMA-5_New_Detector_001_preIMC_on_preIMS_landmark_regions.json"
    snakemake.input['microscopy_image_2'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/preIMS/test_split_ims_preIMS.ome.tiff"
    snakemake.input['contours_in_2'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/preIMS/Cirrhosis-TMA-5_New_Detector_001_preIMS_landmark_regions.json"
    snakemake.input['IMC_location'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/IMC_location/test_split_ims_IMC_mask_on_preIMS_A1.geojson"
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

microscopy_file_1 = snakemake.input['microscopy_image_1']
contours_file_in_1 = snakemake.input['contours_in_1']

microscopy_file_2 = snakemake.input['microscopy_image_2']
contours_file_in_2 = snakemake.input['contours_in_2']

IMC_location = snakemake.input['IMC_location']
if isinstance(IMC_location, list):
    IMC_location = IMC_location[0]

input_spacing_1 = snakemake.params["input_spacing_1"]
input_spacing_2 = snakemake.params["input_spacing_2"]
input_spacing_IMC_location = snakemake.params["input_spacing_IMC_location"]
output_spacing = snakemake.params["output_spacing"]

dmax = snakemake.params["max_distance"]/input_spacing_1
dmin = snakemake.params["min_distance"]/input_spacing_1

pixel_expansion = snakemake.params["pixel_expansion"]


logging.info("core bounding box extraction")
IMC_geojson = json.load(open(IMC_location, "r"))
if isinstance(IMC_geojson,list):
    IMC_geojson=IMC_geojson[0]
boundary_points = np.array(IMC_geojson['geometry']['coordinates'])[0,:,:]
xmin=np.min(boundary_points[:,1])
xmax=np.max(boundary_points[:,1])
ymin=np.min(boundary_points[:,0])
ymax=np.max(boundary_points[:,0])


s1f = input_spacing_1/input_spacing_IMC_location
bb1 = [int(xmin/s1f-pixel_expansion/input_spacing_1),int(ymin/s1f-pixel_expansion/input_spacing_1),int(xmax/s1f+pixel_expansion/input_spacing_1),int(ymax/s1f+pixel_expansion/input_spacing_1)]
imxmax, imymax, _ = get_image_shape(microscopy_file_1)
imxmax=int(imxmax/input_spacing_1)
imymax=int(imymax/input_spacing_1)
bb1[0] = bb1[0] if bb1[0]>=0 else 0
bb1[1] = bb1[1] if bb1[1]>=0 else 0
bb1[2] = bb1[2] if bb1[2]<=imxmax else imxmax
bb1[3] = bb1[3] if bb1[3]<=imymax else imymax
logging.info(f"bounding box whole image 1: {bb1}")

s2f = input_spacing_2/input_spacing_IMC_location
bb2 = [int(xmin/s2f-pixel_expansion/input_spacing_2),int(ymin/s2f-pixel_expansion/input_spacing_2),int(xmax/s2f+pixel_expansion/input_spacing_2),int(ymax/s2f+pixel_expansion/input_spacing_2)]
imxmax, imymax, _ = get_image_shape(microscopy_file_2)
imxmax=int(imxmax/input_spacing_1)
imymax=int(imymax/input_spacing_1)
bb2[0] = bb2[0] if bb2[0]>=0 else 0
bb2[1] = bb2[1] if bb2[1]>=0 else 0
bb2[2] = bb2[2] if bb2[2]<=imxmax else imxmax
bb2[3] = bb2[3] if bb2[3]<=imymax else imymax
logging.info(f"bounding box whole image 2: {bb2}")

m2full_shape = get_image_shape(microscopy_file_1)
bb3 = [int(xmin/s1f-1251/input_spacing_1),int(ymin/s1f-1251/input_spacing_1),int(xmax/s1f+1251/input_spacing_1),int(ymax/s1f+1251/input_spacing_1)]
bb3[0] = bb3[0] if bb3[0]>=0 else 0
bb3[1] = bb3[1] if bb3[1]>=0 else 0
bb3[2] = bb3[2] if bb3[2]<=m2full_shape[0] else m2full_shape[0]
bb3[3] = bb3[3] if bb3[3]<=m2full_shape[1] else m2full_shape[1]
logging.info(f"bounding box mask whole image 1: {bb3}")

logging.info("load microscopy image 1")
img1 = readimage_crop(microscopy_file_1, bb1)
img1 = convert_and_scale_image(img1, input_spacing_1/output_spacing)

logging.info("load microscopy image 2")
img2 = readimage_crop(microscopy_file_2, bb2)
img2 = convert_and_scale_image(img2, input_spacing_2/output_spacing)

# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0].imshow(img1, cmap='gray')
# ax[1].imshow(img2, cmap='gray')
# plt.show()

logging.info("Read regions 1")
with open(contours_file_in_1, 'r') as f:
    data = json.load(f)
regionsls = data['regions']
regions1 = [np.array(reg,dtype=int) for reg in regionsls]
# remove regions at the border
regions1 = [ regions1[k] for k in range(len(regions1)) if not np.any(np.logical_or(
    np.min(regions1[k],axis=0)<=np.array([0,0]),
    np.max(regions1[k],axis=0)>=np.array([img1.shape[1]-2,img1.shape[0]-2])
))]
bboxes1=[cv2.boundingRect(reg) for reg in regions1]

logging.info("Read regions 2")
with open(contours_file_in_2, 'r') as f:
    data = json.load(f)
regionsls = data['regions']
regions2 = [np.array(reg,dtype=int) for reg in regionsls]
# remove regions at the border
regions2 = [ regions2[k] for k in range(len(regions2)) if not np.any(np.logical_or(
    np.min(regions2[k],axis=0)<=np.array([0,0]),
    np.max(regions2[k],axis=0)>=np.array([img2.shape[1]-2,img2.shape[0]-2])
))]
bboxes2=[cv2.boundingRect(reg) for reg in regions2]

# img1_stacked = np.zeros(img1.shape,dtype=np.uint8)
# for k in range(len(regions1)):
#     img1_stacked = cv2.drawContours(
#         img1_stacked, 
#         [regions1[k]], 
#         -1, 
#         k+1,
#         -1)

# img2_stacked = np.zeros(img2.shape,dtype=np.uint8)
# for k in range(len(regions2)):
#     img2_stacked = cv2.drawContours(
#         img2_stacked, 
#         [regions2[k]], 
#         -1, 
#         k+1,
#         -1)
# fig, ax = plt.subplots(nrows=2, ncols=2)
# ax[0,0].imshow(img1_stacked)
# ax[0,1].imshow(img1, cmap='gray')
# ax[1,0].imshow(img2_stacked)
# ax[1,1].imshow(img2, cmap='gray')
# plt.show()

def get_moments(regions):
    moments = np.zeros((len(regions),10))
    for k in range(len(regions)):
        ms = cv2.moments(regions[k])
        # area
        # ms['m00']
        # 'mu..' are the central moments
        if ms['m00']>0:
            # moments1[k,:]=np.array([ms['m00'], ms['m01']/ms['m00'], ms['m10']/ms['m00'], ms['mu20'], ms['mu11'], ms['mu02'], ms['mu30'], ms['mu21'], ms['mu12'], ms['mu03']])
            moments[k,:]=np.array([ms['m00'], ms['m10']/ms['m00'],ms['m01']/ms['m00'],  ms['nu20'], ms['nu11'], ms['nu02'], ms['nu30'], ms['nu21'], ms['nu12'], ms['nu03']])      
    return moments

def get_descriptors(regions, bboxes, maxdim, descriptor, px_exand=3):
    moments = np.zeros((len(regions),10))
    kp = list()
    for k in range(len(regions)):
        timg = cv2.drawContours(
            np.zeros((bboxes[k][3]+px_exand*2,bboxes[k][2]+px_exand*2),dtype=np.uint8), 
            [regions[k] - regions[k].min(axis=0)+px_exand], 
            -1, 
            1,
            thickness=-1)
        M=get_moments([timg])
        # coords[k,1] = bboxes[k][0]+M[0,2]-px_exand
        M[0,2] = bboxes[k][1]+M[0,2]-px_exand
        # coords[k,0] = bboxes[k][1]+M[0,1]-px_exand
        M[0,1] = bboxes[k][0]+M[0,1]-px_exand
        moments[k,:] = M.flatten()
        sc = maxdim/(max([bboxes[k][2],bboxes[k][3]])+px_exand*2)
        timg = cv2.resize(timg, (int(timg.shape[1]*sc),int(timg.shape[0]*sc)), interpolation=cv2.INTER_NEAREST)
        ttimg=np.zeros((maxdim,maxdim),dtype=np.uint8)
        offset = (np.array(ttimg.shape)-np.array(timg.shape))//2
        ttimg[offset[0]:offset[0]+timg.shape[0],offset[1]:offset[1]+timg.shape[1]] = timg
        kp.append(descriptor.compute(timg, [cv2.KeyPoint(x=px_exand+offset[0]+bboxes[k][2]/2,y=px_exand+offset[1]+bboxes[k][3]/2,size=1)]))

    return kp, moments

logging.info(f"Number of regions 1: {len(regions1)}")
logging.info(f"Number of regions 2: {len(regions2)}")

if len(regions1)>=1 and len(regions2)>=1:
    logging.info("get moments and filter")
    descriptor = cv2.BRISK_create(octaves=1)
    maxdim=128
    kp1,moments1 = get_descriptors(regions1, bboxes1, maxdim, descriptor, px_exand=3)
    areas1 = moments1[:,0]
    # notempty1 = np.array([len(m[0])>0 for m in kp1])
    notempty1 = np.ones(moments1.shape[0],dtype=bool)
    # des1 = np.float32([ m[1][0] for l,m in enumerate(kp1) if notempty1[l]])
    moments1 = moments1[notempty1,:]
    areas1=np.array(areas1)[notempty1]
    regions1 = [ regions1[l] for l in range(len(regions1)) if notempty1[l]]

    kp2, moments2 = get_descriptors(regions2, bboxes2, maxdim, descriptor, px_exand=3)
    areas2 = moments2[:,0]
    # notempty2 = np.array([len(m[0])>0 for m in kp2])
    notempty2 = np.ones(moments2.shape[0],dtype=bool)
    # des2 = np.float32([ m[1][0] for l,m in enumerate(kp2) if notempty2[l]])
    moments2 = moments2[notempty2,:]
    areas2=np.array(areas2)[notempty2]
    regions2 = [ regions2[l] for l in range(len(regions2)) if notempty2[l]]

    logging.info("match regions")
    # Matching
    # first step: based on physical distance (i.e. smaller than dmax)
    kdt = KDTree(moments1[:,1:3], leaf_size=30, metric='euclidean')
    physical_distances, indices = kdt.query(moments2[:,1:3], k=np.min([50,len(moments1[:,1:3])]), return_distance=True)

    kdt2 = KDTree(moments2[:,1:3], leaf_size=30, metric='euclidean')
    physical_distances2, indices2 = kdt2.query(moments1[:,1:3], k=np.min([50,len(moments2[:,1:3])]), return_distance=True)



    def single_match(k):  
        b1 = physical_distances[k,:]<dmax
        area_log = np.abs(np.log10(areas1[indices[k,:]]/areas2[k]))
        b2 = area_log<np.log10(1.5/1)
        bb = np.logical_and(b1,b2)
        filt_indices = indices[k,bb].copy()

        b3 = physical_distances2[filt_indices,:]<dmax
        indices2_filt = [indices2[fi,b3[p]] for p,fi in enumerate(filt_indices)]
        b4=[k in id2 for id2 in indices2_filt]
        filt_indices = filt_indices[b4].copy()

        distances= np.array([cv2.matchShapes(regions2[k],regions1[fi],cv2.CONTOURS_MATCH_I2,0) for fi in filt_indices])
        # distances = np.linalg.norm(des1[filt_indices,:] - des2[k,:], axis=1)
        # distances = np.linalg.norm(moments1[filt_indices,3:] - moments2[k,3:], axis=1)
        if len(filt_indices)<1:
            return cv2.DMatch(_distance=np.inf, _queryIdx=-1, _trainIdx=-1)
        else:
            ind = np.argmin(distances)
            # np.linalg.norm(moments1[filt_indices[ind[1]],3:] - moments2[k,3:])
            return cv2.DMatch(_distance=distances[ind], _queryIdx=filt_indices[ind], _trainIdx=k)

    matches = list(map(single_match, range(len(kp2))))

    matches_filt=list()
    for m in matches:
        if m.distance!=np.inf:
            matches_filt.append(m)

    regions1_filt = [ regions1[m.queryIdx] for m in matches_filt ]
    bboxes1_filt = [ bboxes1[m.queryIdx] for m in matches_filt ]
    areas1_filt = np.array([ areas1[m.queryIdx] for m in matches_filt ])
    regions2_filt = [ regions2[m.trainIdx] for m in matches_filt ]
    bboxes2_filt = [ bboxes2[m.trainIdx] for m in matches_filt ]
    areas2_filt = np.array([ areas2[m.trainIdx] for m in matches_filt ])

    # img1_stacked = np.zeros(img1.shape,dtype=np.uint8)
    # for k in range(len(regions1_filt)):
    #     img1_stacked = cv2.drawContours(
    #         img1_stacked, 
    #         [regions1_filt[k]], 
    #         -1, 
    #         k+1,
    #         -1)

    # img2_stacked = np.zeros(img2.shape,dtype=np.uint8)
    # for k in range(len(regions2_filt)):
    #     img2_stacked = cv2.drawContours(
    #         img2_stacked, 
    #         [regions2_filt[k]], 
    #         -1, 
    #         k+1,
    #         -1)

    # fig, ax = plt.subplots(nrows=2, ncols=2)
    # ax[0,0].imshow(img1_stacked)
    # ax[0,1].imshow(img1, cmap='gray')
    # ax[1,0].imshow(img2_stacked)
    # ax[1,1].imshow(img2, cmap='gray')
    # plt.show()


    src_pts = np.float32([ moments1[m.queryIdx,1:3] for m in matches_filt ]).reshape(-1,1,2)
    dst_pts = np.float32([ moments2[m.trainIdx,1:3] for m in matches_filt ]).reshape(-1,1,2)

    logging.info(f"Number of matches: {len(matches_filt)}")
    if len(matches_filt)>1:
        # second step: remove one to many matches
        kdt = KDTree(src_pts.reshape(-1,2), leaf_size=30, metric='euclidean')
        tmp_dist, tmp_indices = kdt.query(src_pts.reshape(-1,2), k=2, return_distance=True)
        tmparr = np.array(list(map(lambda x: np.sort(x), tmp_indices[tmp_dist[:,1]==0,:])))
        tmparr = np.unique(tmparr,axis=0)
        to_remove_1 = np.array(list(map(lambda k: tmparr[k,np.argmax([matches_filt[tmparr[k,0]].distance, matches_filt[tmparr[k,1]].distance])], range(tmparr.shape[0]))))

        kdt = KDTree(dst_pts.reshape(-1,2), leaf_size=30, metric='euclidean')
        tmp_dist, tmp_indices = kdt.query(dst_pts.reshape(-1,2), k=2, return_distance=True)
        tmparr = np.array(list(map(lambda x: np.sort(x), tmp_indices[tmp_dist[:,1]==0,:])))
        tmparr = np.unique(tmparr,axis=0)
        to_remove_2 = np.array(list(map(lambda k: tmparr[k,np.argmax([matches_filt[tmparr[k,0]].distance, matches_filt[tmparr[k,1]].distance])], range(tmparr.shape[0]))))

        to_remove = np.unique(np.append(to_remove_1,to_remove_2))
        to_keep = np.array([k for k in range(len(matches_filt)) if k not in to_remove])
        if len(to_keep)>0:
            matches_filt = np.array(matches_filt)[to_keep]
            src_pts = src_pts[to_keep,:,:]  
            dst_pts = dst_pts[to_keep,:,:]  
            regions1_filt = [ regions1_filt[k] for k in to_keep]
            regions2_filt = [ regions2_filt[k] for k in to_keep]
            areas1_filt = areas1_filt[to_keep]
            areas2_filt = areas2_filt[to_keep]
        else:
            matches_filt = []

    logging.info(f"Number of matches: {len(matches_filt)}")
    if len(matches_filt)>1:
        # third step: remove points close to each other (i.e. smaller than dmin)
        ctr1 = -1
        ctr2 = 0
        to_remove = np.zeros(0)
        did_break = False
        while ctr1 < ctr2:
            # filter points
            to_keep = np.array([k for k in range(len(matches_filt)) if k not in to_remove])
            tmp_matches_filt = matches_filt[to_keep]
            if len(to_keep) < 5:
                did_break = True
                break

            # first image 
            kdt = KDTree(src_pts.reshape(-1,2)[to_keep], leaf_size=30, metric='euclidean')
            tmp_dist, tmp_indices = kdt.query(src_pts.reshape(-1,2)[to_keep], k=2, return_distance=True)
            tmparr = np.array(list(map(lambda x: np.sort(x), tmp_indices[tmp_dist[:,1]<dmin,:])))
            tmparr = np.unique(tmparr,axis=0)
            to_remove_1 = np.array(list(map(lambda k: tmparr[k,np.argmax([tmp_matches_filt[tmparr[k,0]].distance, tmp_matches_filt[tmparr[k,1]].distance])], range(tmparr.shape[0]))))
            if len(to_remove_1)>0:
                to_remove_1 = to_keep[to_remove_1]

            # second image 
            kdt = KDTree(dst_pts.reshape(-1,2)[to_keep], leaf_size=30, metric='euclidean')
            tmp_dist, tmp_indices = kdt.query(dst_pts.reshape(-1,2)[to_keep], k=2, return_distance=True)
            tmparr = np.array(list(map(lambda x: np.sort(x), tmp_indices[tmp_dist[:,1]<dmin,:])))
            tmparr = np.unique(tmparr,axis=0)
            to_remove_2 = np.array(list(map(lambda k: tmparr[k,np.argmax([tmp_matches_filt[tmparr[k,0]].distance, tmp_matches_filt[tmparr[k,1]].distance])], range(tmparr.shape[0]))))
            if len(to_remove_2)>0:
                to_remove_2 = to_keep[to_remove_2]

            # combine
            to_remove = np.append(to_remove, np.unique(np.append(to_remove_1,to_remove_2)))
            to_remove = np.unique(to_remove)
            ctr1 = ctr2
            ctr2 = len(to_remove)

        # apply filter
        to_keep = np.array([k for k in range(len(matches_filt)) if k not in to_remove])

        if len(to_keep)>0:
            matches_filt = np.array(matches_filt)[to_keep]
            src_pts = src_pts[to_keep,:,:]  
            dst_pts = dst_pts[to_keep,:,:]  
            regions1_filt = [ regions1_filt[k] for k in to_keep]
            regions2_filt = [ regions2_filt[k] for k in to_keep]
            areas1_filt = areas1_filt[to_keep]
            areas2_filt = areas2_filt[to_keep]
        else:
            matches_filt = []

    logging.info(f"Number of matches: {len(matches_filt)}")
    if len(matches_filt)>1:
        kpf1 = np.array([moments1[matches_filt[k].queryIdx,1:3] for k in range(len(matches_filt))])
        kpf2 = np.array([moments2[matches_filt[k].trainIdx,1:3] for k in range(len(matches_filt))])
        dists_real = np.sqrt(np.sum((kpf1/output_spacing-kpf2/output_spacing)**2,axis=1))

        def get_dice(contourA, contourB):
            polygonA = shapely.geometry.Polygon(contourA)
            polygonA = shapely.make_valid(polygonA)
            polygonB = shapely.geometry.Polygon(contourB)
            polygonB = shapely.make_valid(polygonB)
            po = shapely.intersection(polygonA, polygonB)
            return (2*po.area)/(polygonA.area+polygonB.area)

        dices = [get_dice(regions1_filt[k], regions2_filt[k]) for k in range(len(regions1_filt))]

        to_keep = np.array([i for i,d in enumerate(dices) if d>0])
        if len(to_keep)>0:
            matches_filt = np.array(matches_filt)[to_keep]
            regions1_filt = [ regions1_filt[k] for k in to_keep]
            regions2_filt = [ regions2_filt[k] for k in to_keep]
            dists_real = dists_real[to_keep]
            dices = np.array(dices)[to_keep]
        else:
            matches_filt = []
    

    logging.info(f"Number of matches: {len(matches_filt)}")
    if len(matches_filt)>1:

        img1_stacked = np.zeros(img1.shape,dtype=np.uint8)
        for k in range(len(regions1_filt)):
            img1_stacked = cv2.drawContours(
                img1_stacked, 
                [regions1_filt[k]], 
                -1, 
                1,
                -1)
        img2_stacked = np.zeros(img2.shape,dtype=np.uint8)
        for k in range(len(regions2_filt)):
            img2_stacked = cv2.drawContours(
                img2_stacked, 
                [regions2_filt[k]], 
                -1, 
                1,
                -1)
        # fig, ax = plt.subplots(nrows=1, ncols=2)
        # ax[0].imshow(img1_stacked)
        # ax[1].imshow(img2_stacked)
        # plt.show()
        img2_stacked_scaled = cv2.resize(img2_stacked, (img1_stacked.shape[1], img1_stacked.shape[0]), interpolation=cv2.INTER_NEAREST)

        tmpimg = img1_stacked*85+img2_stacked_scaled*170
        cv2.imwrite(snakemake.output["overlap_image"], tmpimg)

        logging.info("Run SITK registration")
        fixed = sitk.GetImageFromArray(img1_stacked.astype(float))
        moving = sitk.GetImageFromArray(img2_stacked.astype(float))

        def command_iteration(method):
            print(
                f"{method.GetOptimizerIteration():3} "
                + f"= {method.GetMetricValue():10.8f} "
                + f": {method.GetOptimizerPosition()}"
            )

        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMeanSquares()
        R.SetMetricSamplingStrategy(R.REGULAR)
        R.SetMetricSamplingPercentage(0.1, seed=1234)
        R.SetInterpolator(sitk.sitkLinear)
        R.SetOptimizerAsGradientDescentLineSearch(
            learningRate=1, numberOfIterations=1000, 
            convergenceMinimumValue=1e-6, convergenceWindowSize=10,
            lineSearchEpsilon=0.001, lineSearchMaximumIterations=50,
        )
        R.SetOptimizerScalesFromPhysicalShift()
        R.SetInitialTransform(sitk.AffineTransform(2))
        R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

        # run registration
        transform = R.Execute(fixed, moving)

        global_x_shift,global_y_shift = -np.array(transform.GetTranslation())
        global_affinemat = transform.GetMatrix()


        transformed_image = sitk.GetArrayFromImage(sitk.Resample(moving, fixed, transform, sitk.sitkNearestNeighbor, 0.0, moving.GetPixelID())).astype(np.uint8)//255

        tmpimg = img1_stacked*85+transformed_image*170
        # plt.imshow(tmpimg,cmap='gray')
        # plt.show()
        n_points_total = len(regions1_filt)
    else: 
        dists_real=[]
        n_points_total = 0 
        global_x_shift, global_y_shift = np.nan, np.nan
        global_affinemat = np.nan*np.ones(4)
        cv2.imwrite(snakemake.output["overlap_image"], np.zeros(img1.shape,dtype=np.uint8))
else: 
    dists_real=[]
    n_points_total = 0 
    global_x_shift, global_y_shift = np.nan, np.nan
    global_affinemat = np.nan*np.ones(4)
    cv2.imwrite(snakemake.output["overlap_image"], np.zeros(img1.shape,dtype=np.uint8))


tc = len(dists_real)>0
mean_error = np.nan if not tc else np.mean(dists_real)
q95_error = np.nan if not tc else np.quantile(dists_real,0.95)
q75_error = np.nan if not tc else np.quantile(dists_real,0.75)
q50_error = np.nan if not tc else np.quantile(dists_real,0.5)
q25_error = np.nan if not tc else np.quantile(dists_real,0.25)
q05_error = np.nan if not tc else np.quantile(dists_real,0.05)
min_error = np.nan if not tc else np.min(dists_real)
max_error = np.nan if not tc else np.max(dists_real)
logging.info(f"median distance: {q50_error:5.3} (min: {min_error:5.3}, max: {max_error:5.3})")

mean_dice = np.nan if not tc else np.mean(dices)
q95_dice = np.nan if not tc else np.quantile(dices,0.95)
q75_dice = np.nan if not tc else np.quantile(dices,0.75)
q50_dice = np.nan if not tc else np.quantile(dices,0.5)
q25_dice = np.nan if not tc else np.quantile(dices,0.25)
q05_dice = np.nan if not tc else np.quantile(dices,0.05)
min_dice = np.nan if not tc else np.min(dices)
max_dice = np.nan if not tc else np.max(dices)
logging.info(f"median dice: {q50_dice:5.3} (min: {min_dice:5.3}, max: {max_dice:5.3})")




comparison_from = os.path.basename(os.path.dirname(microscopy_file_1))
comparison_to = os.path.basename(os.path.dirname(microscopy_file_2))

reg_measure_dic = {
    f"{comparison_from}_to_{comparison_to}_regions_mean_error": str(mean_error),
    f"{comparison_from}_to_{comparison_to}_regions_quantile05_error": str(q05_error),
    f"{comparison_from}_to_{comparison_to}_regions_quantile25_error": str(q25_error),
    f"{comparison_from}_to_{comparison_to}_regions_quantile50_error": str(q50_error),
    f"{comparison_from}_to_{comparison_to}_regions_quantile75_error": str(q75_error),
    f"{comparison_from}_to_{comparison_to}_regions_quantile95_error": str(q95_error),
    f"{comparison_from}_to_{comparison_to}_regions_min_error": str(min_error),
    f"{comparison_from}_to_{comparison_to}_regions_max_error": str(max_error),
    f"{comparison_from}_to_{comparison_to}_regions_mean_dice": str(mean_dice),
    f"{comparison_from}_to_{comparison_to}_regions_quantile05_dice": str(q05_dice),
    f"{comparison_from}_to_{comparison_to}_regions_quantile25_dice": str(q25_dice),
    f"{comparison_from}_to_{comparison_to}_regions_quantile50_dice": str(q50_dice),
    f"{comparison_from}_to_{comparison_to}_regions_quantile75_dice": str(q75_dice),
    f"{comparison_from}_to_{comparison_to}_regions_quantile95_dice": str(q95_dice),
    f"{comparison_from}_to_{comparison_to}_regions_min_dice": str(min_dice),
    f"{comparison_from}_to_{comparison_to}_regions_max_dice": str(max_dice),
    f"{comparison_from}_to_{comparison_to}_regions_global_x_shift": str(global_x_shift),
    f"{comparison_from}_to_{comparison_to}_regions_global_y_shift": str(global_y_shift),
    f"{comparison_from}_to_{comparison_to}_regions_global_affine_matrix_00": str(global_affinemat[0]),
    f"{comparison_from}_to_{comparison_to}_regions_global_affine_matrix_01": str(global_affinemat[1]),
    f"{comparison_from}_to_{comparison_to}_regions_global_affine_matrix_10": str(global_affinemat[2]),
    f"{comparison_from}_to_{comparison_to}_regions_global_affine_matrix_11": str(global_affinemat[3]),
    f"{comparison_from}_to_{comparison_to}_regions_n_points_total": str(n_points_total)
    }

logging.info("Save json")
json.dump(reg_measure_dic, open(snakemake.output["error_stats"],"w"))

