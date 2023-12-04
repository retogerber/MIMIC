import matplotlib.pyplot as plt
import shapely
import re
import wsireg
import numpy as np
import json
import cv2
from image_utils import readimage_crop, convert_and_scale_image, get_image_shape 
from utils import setNThreads, snakeMakeMock
import sys,os
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["pixel_expansion"] = 501
    snakemake.params["min_area"] = 24**2
    snakemake.params["max_area"] = 512**2
    snakemake.params["input_spacing"] = 0.22537
    snakemake.params["input_spacing_IMC_location"] = 0.22537
    snakemake.params["output_spacing"] = 1
    snakemake.params['transform_target'] = "preIMS"
    snakemake.params['transform_type'] = "shape"
    snakemake.input["microscopy_image"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/preIMC/test_split_ims_preIMC.ome.tiff"
    snakemake.input['transform_file'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/registrations/postIMC_to_postIMS/test_split_ims-postIMC_to_postIMS_transformations.json"
    snakemake.input['contours_in'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/preIMC/Cirrhosis-TMA-5_New_Detector_001_preIMC_landmark_regions.json"
    snakemake.input['IMC_location'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/IMC_location/test_split_ims_IMC_mask_on_preIMC_A1.geojson"
    snakemake.output["contours_out"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMC/Cirrhosis-TMA-5_New_Detector_002_postIMC_on_preIMC_landmark_regions.json"
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# params
input_spacing = snakemake.params["input_spacing"]
output_spacing = snakemake.params["output_spacing"]
input_spacing_IMC_location = snakemake.params["input_spacing_IMC_location"]
pixel_expansion = snakemake.params["pixel_expansion"]
min_area = snakemake.params["min_area"]
max_area = snakemake.params["max_area"]
transform_target = snakemake.params["transform_target"]
assert(snakemake.params['transform_type'] in ["image","shape"])

# input
microscopy_file=snakemake.input['microscopy_image']
transform_source = os.path.basename(os.path.dirname(microscopy_file))
assert(transform_source in ["postIMC","preIMC","preIMS"])
transform_file=snakemake.input['transform_file']
IMC_location=snakemake.input["IMC_location"]
if isinstance(IMC_location, list):
    IMC_location = IMC_location[0]
contours_file_in=snakemake.input['contours_in']

# output
contours_file_out=snakemake.output['contours_out']


logging.info("core bounding box extraction")
IMC_geojson = json.load(open(IMC_location, "r"))
if isinstance(IMC_geojson,list):
    IMC_geojson=IMC_geojson[0]
boundary_points = np.array(IMC_geojson['geometry']['coordinates'])[0,:,:]
xmin=np.min(boundary_points[:,1])
xmax=np.max(boundary_points[:,1])
ymin=np.min(boundary_points[:,0])
ymax=np.max(boundary_points[:,0])

s1f = input_spacing/input_spacing_IMC_location
bb1 = [int(xmin/s1f-pixel_expansion/input_spacing),int(ymin/s1f-pixel_expansion/input_spacing),int(xmax/s1f+pixel_expansion/input_spacing),int(ymax/s1f+pixel_expansion/input_spacing)]
imxmax, imymax, _ = get_image_shape(microscopy_file)
imxmax=int(imxmax/input_spacing)
imymax=int(imymax/input_spacing)
bb1[0] = bb1[0] if bb1[0]>=0 else 0
bb1[1] = bb1[1] if bb1[1]>=0 else 0
bb1[2] = bb1[2] if bb1[2]<=imxmax else imxmax
bb1[3] = bb1[3] if bb1[3]<=imymax else imymax
logging.info(f"bounding box whole image: {bb1}")


logging.info("Read regions")
with open(contours_file_in, 'r') as f:
    data = json.load(f)

regionsls = data['regions']
regions = [np.array(reg,dtype=np.uint64) for reg in regionsls]
bboxes = data['bboxes']

logging.info("Check overlap")

def is_contour_inside(contourA: np.ndarray, contourB: np.ndarray) -> bool:
    """
    Check if one contour is inside another.

    Parameters:
    contourA (numpy.ndarray): The first contour (a 2D array of points).
    contourB (numpy.ndarray): The second contour (a 2D array of points).

    Returns:
    bool: True if contourB is inside contourA, False otherwise.
    """
    if np.all(contourA==contourB):
        return True
    polygonA = shapely.geometry.Polygon(contourA)
    polygonA = shapely.make_valid(polygonA)
    polygonB = shapely.geometry.Polygon(contourB)
    polygonB = shapely.make_valid(polygonB)

    return polygonA.contains(polygonB) or polygonA.intersects(polygonB)

overlap_matrix = np.zeros((len(regions),len(regions)),dtype=bool)
for i in range(len(regions)):
    for j in range(i+1,len(regions)):
        contourA=regions[i].astype(int)
        contourB=regions[j].astype(float)
        overlap_matrix[i,j] = is_contour_inside(regions[i], regions[j])

logging.info("Filter regions")

def get_dice(contourA: np.ndarray, contourB: np.ndarray) -> float:
    """
    Calculate the Dice coefficient between two contours.

    Parameters:
    contourA (numpy.ndarray): The first contour (a 2D array of points).
    contourB (numpy.ndarray): The second contour (a 2D array of points).

    Returns:
    float: The Dice coefficient between the two contours.
    """
    polygonA = shapely.geometry.Polygon(contourA)
    polygonA = shapely.make_valid(polygonA)
    polygonB = shapely.geometry.Polygon(contourB)
    polygonB = shapely.make_valid(polygonB)
    po = shapely.intersection(polygonA, polygonB)
    return (2*po.area)/(polygonA.area+polygonB.area)


areas=[cv2.moments(regions[k])['m00'] for k in range(len(regions))]
inset = np.arange(len(regions))
outset = list()
to_remove = np.zeros(len(regions),dtype=bool)
# loop through regions
while np.sum(to_remove)<len(regions):
    k=inset[~to_remove][0]
    # if no overlap with other regions
    if np.sum(overlap_matrix[k,~to_remove])==0:
        outset.append(k)
        to_remove[k]=True
    else:
        qs=np.where(overlap_matrix[k,:])[0]
        dices=[get_dice(regions[k], regions[q]) for q in qs]
        # if large overlap with other regions take largest
        if np.any(np.array(dices)>0.98):
            q = qs[np.array(areas)[qs].argmax()]
            outset.append(q)
            tmptorem = np.concatenate([qs,np.array([k])])
            for t in tmptorem:
                to_remove[t]=True
        # if small overlap with other regions take all
        else:
            tmpregs = np.concatenate([qs,np.array([k])])
            sorted_indices = np.flip(np.argsort(np.array(areas)[tmpregs]))
            for si in tmpregs[sorted_indices]:
                outset.append(si)
                to_remove[si]=True
regions = [regions[k] for k in outset]
overlap_matrix = overlap_matrix[outset,:][:,outset]

logging.info("Create image")
microscopy_image = readimage_crop(microscopy_file, bb1)
microscopy_image = convert_and_scale_image(microscopy_image, input_spacing)
img_stacked = np.zeros((microscopy_image.shape[0], microscopy_image.shape[1], 3), dtype=np.uint16)
for k,reg in enumerate(regions):
    img_stacked = cv2.drawContours(
        img_stacked, 
        [reg], 
        -1, 
        k+1,
        -1)
# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0].imshow(overlap_matrix)
# ax[1].imshow(img_stacked[:,:,0])
# plt.show()


logging.info("Setup transformation for image")
rtsn=wsireg.reg_transforms.reg_transform_seq.RegTransformSeq(transform_file)
rtsn.set_output_spacing((float(output_spacing),float(output_spacing)))
rtls = rtsn.reg_transforms
is_split_transform = len(rtls)==6

if transform_target == "preIMC":
    n_end = 1
elif transform_target == "preIMS":
    n_end = 5 if is_split_transform else 3
elif transform_target == "postIMS":
    n_end = 6 if is_split_transform else 4
else:
    raise ValueError("Unknown transform target: " + transform_target)
if transform_source == "postIMC":
    n_start = 0
elif transform_source == "preIMC":
    n_start = 1
elif transform_source == "preIMS":
    n_start = 5 if is_split_transform else 3
else:
    raise ValueError("Unknown transform source: " + transform_source)

rtls = rtsn.reg_transforms
rtls = rtls[n_start:n_end]
rtsn = wsireg.reg_transforms.reg_transform_seq.RegTransformSeq(rtls, transform_seq_idx=list(range(len(rtls))))
if len(rtls)>0:
    rtsn.set_output_spacing((float(output_spacing),float(output_spacing)))

logging.info("Transform IMC bounding box")
rs = wsireg.reg_shapes.RegShapes(IMC_location, source_res=input_spacing, target_res=output_spacing)
rs.transform_shapes(rtsn)
logging.info(f"Output size: {rtsn.output_size}")

tmpout = rs.transformed_shape_data[0]['array']
logging.info(f"Max extend transformed shapes: ({np.min(tmpout[:,0])},{np.min(tmpout[:,1])}) - ({np.max(tmpout[:,0])},{np.max(tmpout[:,1])})")
xmin = np.min(tmpout[:,1])
assert(xmin>0)
xmax = np.max(tmpout[:,1])
assert(xmax<=rtsn.output_size[1])
ymin = np.min(tmpout[:,0])
assert(ymin>0)
ymax = np.max(tmpout[:,0])
assert(ymax<=rtsn.output_size[0])

s1f = 1
bb1tr = [int(xmin/s1f-pixel_expansion),int(ymin/s1f-pixel_expansion),int(xmax/s1f+pixel_expansion),int(ymax/s1f+pixel_expansion)]
imxmax, imymax = rtsn.output_size
imxmax=int(imxmax/input_spacing)
imymax=int(imymax/input_spacing)
bb1tr[0] = bb1tr[0] if bb1tr[0]>=0 else 0
bb1tr[1] = bb1tr[1] if bb1tr[1]>=0 else 0
bb1tr[2] = bb1tr[2] if bb1tr[2]<=imxmax else imxmax
bb1tr[3] = bb1tr[3] if bb1tr[3]<=imymax else imymax
logging.info(f"bounding box whole image: {bb1tr}")


if snakemake.params['transform_type'] == "image":
    logging.info("Transform shapes using image")

    imgshape = get_image_shape(microscopy_file)
    imgnew = np.zeros(imgshape[:2], dtype=np.uint16)
    img = cv2.resize(img_stacked[:,:,0], (bb1[3]-bb1[1],bb1[2]-bb1[0]), interpolation=cv2.INTER_NEAREST)
    imgnew[bb1[0]:bb1[2],bb1[1]:bb1[3]] = img

    # plt.imshow(imgnew)
    # plt.show()

    logging.info("Transform and save image")
    m = re.search("[a-zA-Z0-9]*(?=.geojson$)",os.path.basename(IMC_location))
    core_name = m.group(0)
    tmp_file = os.path.join(os.path.dirname(microscopy_file), core_name+"_tmp.ome.tiff")
    img_basename = os.path.basename(tmp_file).split(".")[0]
    img_dirname = os.path.dirname(tmp_file)
    # transform and save image
    ri = wsireg.reg_images.loader.reg_image_loader(imgnew, input_spacing)
    writer = wsireg.writers.ome_tiff_writer.OmeTiffWriter(ri, reg_transform_seq=rtsn)
    writer.write_image_by_plane(img_basename, output_dir=img_dirname, tile_size=1024)
    logging.info("Read transformed image")
    trimg = readimage_crop(tmp_file, bb1tr)

    logging.info("Remove temporary file")
    os.remove(tmp_file)

    logging.info("Extract transformed regions")
    regvals = np.arange(len(regions))+1
    transformed_regions=[]
    transformed_bboxes=[]
    for p in range(len(regions)):
        tmptrimg = np.zeros(trimg.shape,dtype=np.uint8)
        tmptrimg[trimg==regvals[p]]=255
        cts,_ = cv2.findContours(tmptrimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for ct in cts:
            ca = cv2.contourArea(ct)*output_spacing**2 
            if ca>=min_area and ca<=max_area:
                ct = (ct.reshape(-1,2)*output_spacing).astype(np.uint64)
                transformed_regions.append(ct)
                transformed_bboxes.append(cv2.boundingRect(ct))

elif snakemake.params['transform_type'] == "shape":
    logging.info("transform shapes using shape")
    # scale regions
    regions_scaled = [reg/input_spacing for reg in regions]
    # translate regions
    regions_scaled_translated = [reg+np.array([bb1[1],bb1[0]]) for reg in regions_scaled]
    # transform regions
    rs = wsireg.reg_shapes.RegShapes(regions_scaled_translated, source_res=input_spacing, target_res=output_spacing)
    rs.transform_shapes(rtsn)

    transformed_regions_regshapes = [rs.transformed_shape_data[k]['array'][:-1,:].astype(int) for k in range(len(regions))]
    # translate regions
    transformed_regions = [(reg-np.array([bb1tr[1],bb1tr[0]])).astype(int) for reg in transformed_regions_regshapes]
    transformed_bboxes = [cv2.boundingRect(reg) for reg in transformed_regions]

else:
    raise ValueError(f"Unknown transform type: {snakemake.params['transform_type']}")


logging.info("Save regions")
# save regions
transformed_regionsls = [reg.tolist() for reg in transformed_regions]
with open(contours_file_out, 'w') as f:
    json.dump({'regions': transformed_regionsls, 'bboxes': transformed_bboxes}, f)



# microscopy_image = readimage_crop(microscopy_file, bb1)
# microscopy_image = convert_and_scale_image(microscopy_image, input_spacing)
# img1_stacked = np.zeros(microscopy_image.shape,dtype=np.uint8)
# for k in range(len(regions)):
#     img1_stacked = cv2.drawContours(
#         img1_stacked, 
#         [regions[k]], 
#         -1, 
#         k+1,
#         -1)
# t2="/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMC/Cirrhosis-TMA-5_New_Detector_002_transformed_on_preIMC.ome.tiff"
# t2="/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/preIMC/Cirrhosis-TMA-5_New_Detector_002_transformed_on_preIMC.ome.tiff"
# get_image_shape(t2)
# microscopy_image2 = readimage_crop(t2, (np.array(bb1tr)).astype(int))
# microscopy_image2 = convert_and_scale_image(microscopy_image2, 1)
# img2_stacked = np.zeros(microscopy_image2.shape,dtype=np.uint8)
# for k in range(len(transformed_regions)):
#     img2_stacked = cv2.drawContours(
#         img2_stacked, 
#         [transformed_regions[k]], 
#         -1, 
#         k+1,
#         -1)
# fig, ax = plt.subplots(nrows=2, ncols=2)
# ax[0,0].imshow(img1_stacked)
# ax[0,1].imshow(microscopy_image, cmap='gray')
# ax[1,0].imshow(img2_stacked)
# ax[1,1].imshow(microscopy_image2, cmap='gray')
# plt.show()
# img3_stacked = np.zeros(microscopy_image2.shape,dtype=np.uint8)
# for k in range(len(transformed_regions)):
#     img3_stacked = cv2.drawContours(
#         img3_stacked, 
#         [transformed_regions_regshapes[k]], 
#         -1, 
#         k+1,
#         -1)
# fig, ax = plt.subplots(nrows=2, ncols=2)
# ax[0,0].imshow(img2_stacked)
# ax[0,1].imshow(microscopy_image2, cmap='gray')
# ax[1,0].imshow(img3_stacked)
# ax[1,1].imshow(microscopy_image2, cmap='gray')
# plt.show()




logging.info("Finished")