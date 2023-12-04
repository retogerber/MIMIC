import matplotlib.pyplot as plt
import numpy as np
import json
from segment_anything import sam_model_registry, SamPredictor
import skimage
import skimage.exposure
import re
import numpy as np
from sklearn.neighbors import KDTree
import cv2
from image_utils import readimage_crop, convert_and_scale_image, saveimage_tile, subtract_postIMS_grid, extract_mask, get_image_shape, sam_core, preprocess_mask
from registration_utils import get_angle
from utils import setNThreads, snakeMakeMock
import sys,os
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["input_spacing_1"] = 1
    snakemake.params["input_spacing_2"] = 0.22537
    snakemake.params["input_spacing_IMC_location"] = 0.22537
    snakemake.params["output_spacing"] = 1
    snakemake.params["max_distance"] = 50
    snakemake.params["min_distance"] = 10
    snakemake.params["pixel_expansion"] = 501
    snakemake.input["sam_weights"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/Misc/sam_vit_h_4b8939.pth"
    snakemake.input['microscopy_image_1'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/preIMC/Cirrhosis-TMA-5_New_Detector_001_transformed_on_preIMS.ome.tiff"
    snakemake.input['microscopy_image_2'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/preIMS/test_split_pre_preIMS.ome.tiff"
    snakemake.input['IMC_location'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMC_location/test_split_pre_IMC_mask_on_preIMS_A1.geojson"
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# params
input_spacing_1 = snakemake.params["input_spacing_1"]
input_spacing_2 = snakemake.params["input_spacing_2"]
input_spacing_IMC_location = snakemake.params["input_spacing_IMC_location"]
output_spacing = snakemake.params["output_spacing"]
# maximum assumed distance between corresponding points
dmax = snakemake.params["max_distance"]
# minimum distance between points on the same image 
dmin = snakemake.params["min_distance"]
pixel_expansion = snakemake.params["pixel_expansion"]

# inputs
microscopy_file_1 = snakemake.input['microscopy_image_1']
microscopy_file_2 = snakemake.input['microscopy_image_2']
IMC_location=snakemake.input["IMC_location"]
if isinstance(IMC_location, list):
    IMC_location = IMC_location[0]
CHECKPOINT_PATH = snakemake.input["sam_weights"]
DEVICE = 'cpu'
MODEL_TYPE = "vit_h"

# outputs
microscopy_file_out_1 = snakemake.output['microscopy_image_out_1']
microscopy_file_out_2 = snakemake.output['microscopy_image_out_2']
matching_points_filename_out = snakemake.output['matching_points']


m = re.search("[a-zA-Z]*(?=.ome.tiff$)",os.path.basename(microscopy_file_1))
comparison_to = m.group(0)
comparison_from = os.path.basename(os.path.dirname(microscopy_file_1))
assert(comparison_to in ["preIMC","preIMS","postIMS"])
assert(comparison_from in ["postIMC","preIMC","preIMS"])


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
microscopy_image_1 = readimage_crop(microscopy_file_1, bb1)
microscopy_image_1 = convert_and_scale_image(microscopy_image_1, input_spacing_1/output_spacing)

logging.info("load microscopy image 2")
microscopy_image_2 = readimage_crop(microscopy_file_2, bb2)
microscopy_image_2 = convert_and_scale_image(microscopy_image_2, input_spacing_2/output_spacing)

logging.info("Extract mask for microscopy image 1")
mask_2 = extract_mask(microscopy_file_1, bb3, rescale = input_spacing_1/output_spacing, is_postIMS=False)[0,:,:]
xb = int((mask_2.shape[0]-microscopy_image_1.shape[0])/2)
yb = int((mask_2.shape[1]-microscopy_image_1.shape[1])/2)
wn = microscopy_image_1.shape[0]
hn = microscopy_image_1.shape[1]
mask_2 = cv2.resize(mask_2[xb:-xb,yb:-yb].astype(np.uint8), (hn,wn), interpolation=cv2.INTER_NEAREST)
mask_2_proportion = np.sum(mask_2)/np.prod(mask_2.shape)
logging.info(f"proportion image 1 covered by mask (rembg): {mask_2_proportion:5.4}")

bb0 = [int(xmin/s1f),int(ymin/s1f),int(xmax/s1f),int(ymax/s1f)]
IMC_mask_proportion = ((bb0[2]-bb0[0])*(bb0[3]-bb0[1]))/((bb1[2]-bb1[0])*(bb1[3]-bb1[1]))
if mask_2_proportion < IMC_mask_proportion:
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    saminp = readimage_crop(microscopy_file_1, bb1)
    # to gray scale, rescale
    saminp = convert_and_scale_image(saminp, input_spacing_1/output_spacing)
    saminp = np.stack([saminp, saminp, saminp], axis=2)
    # run SAM segmentation model
    masks, scores1 = sam_core(saminp, sam)
    # postprocess
    masks = np.stack([skimage.morphology.convex_hull_image(preprocess_mask(msk,1)) for msk in masks ])
    tmpareas = np.array([np.sum(im) for im in masks])
    indmax = np.argmax(tmpareas/(masks.shape[1]*masks.shape[2]))
    mask_2 = masks[indmax,:,:].astype(np.uint8)
    mask_2_proportion = np.sum(mask_2)/np.prod(mask_2.shape)
    logging.info(f"proportion image 1 covered by mask (SAM): {mask_2_proportion:5.4}")
    if mask_2_proportion < IMC_mask_proportion:
        mask_2 = np.ones(microscopy_image_1.shape, dtype=np.uint8)


# px.imshow(mask_2).show()
# px.imshow(microscopy_image_2).show()

xmax = min([microscopy_image_1.shape[0],microscopy_image_2.shape[0]])
ymax = min([microscopy_image_1.shape[1],microscopy_image_2.shape[1]])
imcbbox_outer = [0,0,xmax,ymax]
logging.info(f"imcbbox_outer: {imcbbox_outer}")

print(f"snakemake params remove_postIMS_grid: {snakemake.params['remove_postIMS_grid']}")
if snakemake.params["remove_postIMS_grid"]:
    img = microscopy_image_2[imcbbox_outer[0]:imcbbox_outer[2],imcbbox_outer[1]:imcbbox_outer[3]].copy()
    mask_2_on_2 = cv2.resize(mask_2, (microscopy_image_2.shape[1], microscopy_image_2.shape[0]), interpolation=cv2.INTER_NEAREST)
    cv2.morphologyEx(src=mask_2_on_2, dst=mask_2_on_2, op = cv2.MORPH_DILATE, kernel = skimage.morphology.square(20))
    img[mask_2_on_2==0]=0
    out = subtract_postIMS_grid(img)
    out[mask_2_on_2==0]=0
    microscopy_image_2[imcbbox_outer[0]:imcbbox_outer[2],imcbbox_outer[1]:imcbbox_outer[3]] = out 

def get_regions(img, min_area=96**2,max_area=1024**2,delta=2):
    regions,_ = cv2.MSER_create(min_area=min_area,max_area=max_area,delta=delta).detectRegions(img)

    regionsnew = []
    bboxesnew = []
    for k in range(len(regions)):
        timg = cv2.drawContours(
            np.zeros(img.shape,dtype=np.uint8), 
            [regions[k]], 
            -1, 
            255)
        timg = cv2.morphologyEx(timg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19)))
        ct = cv2.findContours(timg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0].reshape(-1,2)
        bboxesnew.append(cv2.boundingRect(ct))
        regionsnew.append(ct)
    return regionsnew, bboxesnew


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
    # coords = np.zeros((len(regions),2))
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

logging.info("Setup split into segments")
x_segs = np.arange(imcbbox_outer[0],imcbbox_outer[2],(imcbbox_outer[2]-imcbbox_outer[0])/1).astype(int)
x_segs = np.append(x_segs,imcbbox_outer[2])
y_segs = np.arange(imcbbox_outer[1],imcbbox_outer[3],(imcbbox_outer[3]-imcbbox_outer[1])/1).astype(int)
y_segs = np.append(y_segs,imcbbox_outer[3])


 
logging.info("Loop over segments")
dists_real = np.zeros(0)
dists = np.zeros(0)
dist_ratios = np.zeros(0)
angles = np.zeros(0)
kpf1compl = np.zeros((0,2))
kpf2compl = np.zeros((0,2))
ij_array = np.zeros((0,2))
matdic = {}
homography_mask = np.zeros(0)
regions1_all = []
regions2_all = []
areas1_all = np.zeros((0))
areas2_all = np.zeros((0))
n_shift = 1
# n_shift = 3
logging.info("segment | n_found | n_used | dx | dy")
for j in range((len(x_segs)-n_shift)):
    for i in range((len(x_segs)-n_shift)):
        cv2.setRNGSeed(2391)
        
        # descriptor = cv2.SIFT_create(nOctaveLayers=1, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
        descriptor = cv2.BRISK_create(octaves=1)
        maxdim=128


        # extract keypoints and descriptors
        img1 = microscopy_image_1[x_segs[i]:x_segs[i+n_shift],y_segs[j]:y_segs[j+n_shift]]
        img1 = cv2.bitwise_not(img1)
        # img1 = cv2.medianBlur(img1, 3)
        regions1, bboxes1 = get_regions(img1)

        kp1,moments1 = get_descriptors(regions1, bboxes1, maxdim, descriptor, px_exand=3)
        areas1 = moments1[:,0]
        notempty1 = np.array([len(m[0])>0 for m in kp1])
        des1 = np.float32([ m[1][0] for l,m in enumerate(kp1) if notempty1[l]])
        moments1 = moments1[notempty1,:]
        areas1=np.array(areas1)[notempty1]
        regions1 = [ regions1[l] for l in range(len(regions1)) if notempty1[l]]

        img2 = microscopy_image_2[x_segs[i]:x_segs[i+n_shift],y_segs[j]:y_segs[j+n_shift]]
        img2 = cv2.bitwise_not(img2)
        # img2 = cv2.medianBlur(img2, 3)
        regions2, bboxes2 = get_regions(img2)

        kp2, moments2 = get_descriptors(regions2, bboxes2, maxdim, descriptor, px_exand=3)
        areas2 = moments2[:,0]

        notempty2 = np.array([len(m[0])>0 for m in kp2])
        des2 = np.float32([ m[1][0] for l,m in enumerate(kp2) if notempty2[l]])
        moments2 = moments2[notempty2,:]
        areas2=np.array(areas2)[notempty2]
        regions2 = [ regions2[l] for l in range(len(regions2)) if notempty2[l]]

        if moments1.shape[0] < 5 or moments2.shape[0] < 5:
            logging.info(f"{i}_{j}: {0:4} {0:4} {np.nan:8.3} {np.nan:8.3}")
            continue

        # Matching
        # first step: based on physical distance (i.e. smaller than dmax)
        kdt = KDTree(moments1[:,1:3], leaf_size=30, metric='euclidean')
        physical_distances, indices = kdt.query(moments2[:,1:3], k=np.min([500,len(moments1[:,1:3])]), return_distance=True)

        # k=10
        # moments1[indices[k,physical_distances[k,:]<dmax],1:3]
        # moments2[k,1:3]

        def single_match(k):  
            b1 = physical_distances[k,:]<dmax
            area_log = np.abs(np.log10(areas1[indices[k,:]]/areas2[k]))
            b2 = area_log<np.log10(1.5/1)
            bb = np.logical_and(b1,b2)
            filt_indices = indices[k,bb].copy()

            # moments1[filt_indices,1:3]
            # moments2[k,1:3]
            # np.linalg.norm(moments1[filt_indices,1:3] - moments2[k,1:3], axis=1)

            # [ np.linalg.norm(np.mean(regions1[filt_indices[p]],axis=0)-np.mean(regions2[k],axis=0)) for p in range(len(filt_indices)) ]

            distances = np.linalg.norm(des1[filt_indices,:] - des2[k,:], axis=1)
            # distances = np.linalg.norm(moments1[filt_indices,3:] - moments2[k,3:], axis=1)
            # distances = np.linalg.norm(moments1[indices[k,physical_distances[k,:]<dmax],3:] - moments2[k,3:], axis=1)
            if np.sum(bb)<3:
                return [cv2.DMatch(),cv2.DMatch()]
            else:
                ind = np.argpartition(distances, 2)[:2]
                # np.linalg.norm(moments1[filt_indices[ind[1]],3:] - moments2[k,3:])
                return list(map(lambda l: cv2.DMatch(_distance=distances[l], _queryIdx=filt_indices[l], _trainIdx=k), ind))

        matches = list(map(single_match, range(len(des2))))

        # Apply ratio test
        matches_filt = []
        matches_filt_dist_ratio = []
        l=0
        p=0
        for m,n in matches:
            if m.distance < 0.8*n.distance:
                # print(f"{l} {p}")
                matches_filt.append(m)
                matches_filt_dist_ratio.append(m.distance/n.distance)
                l+=1
            p+=1
       
        regions1_filt = [ regions1[m.queryIdx] for m in matches_filt ]
        bboxes1_filt = [ bboxes1[m.queryIdx] for m in matches_filt ]
        areas1_filt = np.array([ areas1[m.queryIdx] for m in matches_filt ])
        regions2_filt = [ regions2[m.trainIdx] for m in matches_filt ]
        bboxes2_filt = [ bboxes2[m.trainIdx] for m in matches_filt ]
        areas2_filt = np.array([ areas2[m.trainIdx] for m in matches_filt ])

        arrowed_microscopy_image_1 = np.stack([img1, img1, img1], axis=2)
        for k in range(len(regions1_filt)):
            arrowed_microscopy_image_1 = cv2.drawContours(
                arrowed_microscopy_image_1, 
                [regions1_filt[k]], 
                -1, 
                255,
                -1)

        arrowed_microscopy_image_2 = np.stack([img2, img2, img2], axis=2)
        for k in range(len(regions2_filt)):
            arrowed_microscopy_image_2 = cv2.drawContours(
                arrowed_microscopy_image_2, 
                [regions2_filt[k]], 
                -1, 
                255,
                -1)

        fig, ax = plt.subplots(nrows=2, ncols=2)
        ax[0,0].imshow(arrowed_microscopy_image_1)
        ax[0,1].imshow(img1, cmap='gray')
        ax[1,0].imshow(arrowed_microscopy_image_2)
        ax[1,1].imshow(img2, cmap='gray')
        plt.show()


        src_pts = np.float32([ moments1[m.queryIdx,1:3] for m in matches_filt ]).reshape(-1,1,2)
        dst_pts = np.float32([ moments2[m.trainIdx,1:3] for m in matches_filt ]).reshape(-1,1,2)

        if src_pts.shape[0] < 5 or dst_pts.shape[0] < 5:
            logging.info(f"{i}_{j}: {0:4} {0:4} {np.nan:8.3} {np.nan:8.3}")
            continue


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
        matches_filt = np.array(matches_filt)[to_keep]
        matches_filt_dist_ratio = np.array(matches_filt_dist_ratio)[to_keep]
        src_pts = src_pts[to_keep,:,:]  
        dst_pts = dst_pts[to_keep,:,:]  
        regions1_filt = [ regions1_filt[k] for k in to_keep]
        regions2_filt = [ regions2_filt[k] for k in to_keep]
        areas1_filt = areas1_filt[to_keep]
        areas2_filt = areas2_filt[to_keep]

        if src_pts.shape[0] < 5 or dst_pts.shape[0] < 5:
            logging.info(f"{i}_{j}: {0:4} {0:4} {np.nan:8.3} {np.nan:8.3}")
            continue

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
        
        if did_break:
            logging.info(f"{i}_{j}: {0:4} {0:4} {np.nan:8.3} {np.nan:8.3}")
            continue

        # apply filter
        to_keep = np.array([k for k in range(len(matches_filt)) if k not in to_remove])
        
        matches_filt = np.array(matches_filt)[to_keep]
        matches_filt_dist_ratio = np.array(matches_filt_dist_ratio)[to_keep]
        src_pts = src_pts[to_keep,:,:]  
        dst_pts = dst_pts[to_keep,:,:]  
        regions1_filt = [ regions1_filt[k] for k in to_keep]
        regions2_filt = [ regions2_filt[k] for k in to_keep]
        areas1_filt = areas1_filt[to_keep]
        areas2_filt = areas2_filt[to_keep]


        if src_pts.shape[0] < 5 or dst_pts.shape[0] < 5:
            logging.info(f"{i}_{j}: {0:4} {0:4} {np.nan:8.3} {np.nan:8.3}")
            continue


        # sort points, and revert sort afterwards (needed for sampling with PROSAC)
        order_matches_filt = np.array(sorted(np.arange(len(matches_filt)), key = lambda x: matches_filt_dist_ratio[x]))


        # based on: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
        # and https://docs.opencv.org/4.x/d0/d74/md__build_4_x-contrib_docs-lin64_opencv_doc_tutorials_calib3d_usac.html
        # Sampling: PROSAC
        # Scoring: MAGSAC with low threshold
        # Error metric: Sampson distance
        # Degeneracy: DEGENSAC
        # Local optimization: Graph-cut RANSAC
        # Solver: Affine2D
        # Graph creation: NEIGH_FLANN_RADIUS
        params = cv2.UsacParams()
        params.confidence = 0.999
        params.sampler = cv2.SAMPLING_PROSAC
        params.score = cv2.SCORE_METHOD_MAGSAC
        # params.score = cv2.SCORE_METHOD_RANSAC
        params.maxIterations = 1000000
        params.neighborsSearch = cv2.NEIGH_FLANN_RADIUS
        params.threshold = 5
        params.loMethod = cv2.LOCAL_OPTIM_GC
        params.loIterations = 100
        # params.loSampleSize = 20 
        # M, mask = cv2.findHomography(src_pts[order_matches_filt,:], dst_pts[order_matches_filt,:], params)
        # M, mask = cv2.findHomography(src_pts[order_matches_filt,:], dst_pts[order_matches_filt,:], cv2.RANSAC,5.0)
        M, mask = cv2.estimateAffine2D(src_pts[order_matches_filt,:], dst_pts[order_matches_filt,:], params)
        
        maskout = mask.copy()*0
        maskout[order_matches_filt]=mask
        mask=maskout
        if np.sum(mask) < 5:
            mask*=0

        matchesMask = mask.ravel()
        # np.sum(matchesMask)

        # img3 = cv2.drawMatches(img1,kp1,img2,kp2,np.array(matches_filt),None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # img3 = cv2.drawMatches(img1,kp1,img2,kp2,np.array(matches_filt)[matchesMask==1],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.imshow(img3)
        # plt.show()

        logging.info(f"{i}_{j}: {len(matches_filt):4} {np.sum(matchesMask):4} {M[0,2]:8.3} {M[1,2]:8.3}")
        print(f"{i}_{j}: {len(matches_filt):4} {np.sum(matchesMask):4} {M[0,2]:8.3} {M[1,2]:8.3}")

        
        # matches_filt = np.array(matches_filt)[matchesMask==1]
        dists = np.append(dists,np.array([m.distance for m in matches_filt]))
        dist_ratios = np.append(dist_ratios,matches_filt_dist_ratio)

        kpf1 = np.array([moments1[matches_filt[k].queryIdx,1:3] for k in range(len(matches_filt))])
        kpf2 = np.array([moments2[matches_filt[k].trainIdx,1:3] for k in range(len(matches_filt))])
        dists_real = np.append(dists_real,np.sqrt(np.sum((kpf1/output_spacing-kpf2/output_spacing)**2,axis=1)))
        angles = np.append(angles,np.array([get_angle([-10000,kpf1[k,1]],kpf1[k,:],kpf2[k,:]) for k in range(kpf1.shape[0])]))

        kpf1 += np.array([y_segs[j],x_segs[i]])
        kpf2 += np.array([y_segs[j],x_segs[i]])
        kpf1compl = np.append(kpf1compl,kpf1,axis=0)
        kpf2compl = np.append(kpf2compl,kpf2,axis=0)
        
        # TODO: add image square offset
        regions1_filt = [rp+np.array([y_segs[j],x_segs[i]]) for rp in regions1_filt]
        regions2_filt = [rp+np.array([y_segs[j],x_segs[i]]) for rp in regions2_filt]
        regions1_all.append(regions1_filt)
        regions2_all.append(regions2_filt)
        areas1_all = np.append(areas1_all,areas1_filt)
        areas2_all = np.append(areas2_all,areas2_filt)

        ij_array = np.append(ij_array,np.vstack([np.zeros(len(kpf1),dtype=int)+i,np.zeros(len(kpf1),dtype=int)+j]).T,axis=0)

        homography_mask = np.append(homography_mask,matchesMask,axis=0)
        matdic[f"{i}_{j}"] = {'matrix': M, 'i':i, 'j':j, 'n': np.sum(matchesMask)}

regions1_all = [r for rs in regions1_all for r in rs]
regions2_all = [r for rs in regions2_all for r in rs]

logging.info(f"Check if points are on tissue")
contours, hierarchy = cv2.findContours(mask_2*255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# c_areas = np.array([cv2.contourArea(c) for c in contours])
# np.argmax(c_areas)
if len(contours)>1:
    contours = [max(contours, key = cv2.contourArea)]

logging.info(f"proportion image 1 covered by contour of mask: {cv2.contourArea(contours[0])/np.prod(mask_2.shape):5.4}")

in_mask_1 = np.zeros(len(kpf1compl),dtype=bool)
for j in range(len(kpf1compl)):
    in_mask_1[j]=np.any(np.array([cv2.pointPolygonTest(contours[i], np.flip(kpf1compl[j,:]), False) for i in range(len(contours))])==1)
in_mask_2 = np.zeros(len(kpf2compl),dtype=bool)
for j in range(len(kpf1compl)):
    in_mask_2[j]=np.any(np.array([cv2.pointPolygonTest(contours[i], np.flip(kpf2compl[j,:]), False) for i in range(len(contours))])==1)
in_mask = np.logical_and(in_mask_1, in_mask_2)

logging.info(f"\t n_points in mask: {np.sum(in_mask)} / {len(in_mask)}")
# px.scatter(x=kpf1compl[in_mask,1],y=-kpf1compl[in_mask,0]).show()
# px.scatter(x=kpf1compl[~in_mask,1],y=-kpf1compl[~in_mask,0]).show()
# ti = cv2.drawContours(microscopy_image_1.astype(np.uint8), contour, -1, (0,255,0), 3)
# px.imshow(ti).show()

combined_output = np.hstack([kpf1compl,kpf2compl,dists.reshape(-1,1),dist_ratios.reshape(-1,1),homography_mask.reshape(-1,1),in_mask.reshape(-1,1),dists_real.reshape(-1,1),angles.reshape(-1,1),ij_array])
logging.info(f"done")
logging.info("Save points")
np.savetxt(matching_points_filename_out,combined_output,header=f"p1x,p1y,p2x,p2y,distance,distance_ratio,homography_mask,on_tissue,distance_physical,angle,i,j",delimiter=',')

logging.info("Apply filter to points")
# homography_mask = areas1_all>np.quantile(areas1_all,0.95)
# homography_mask = np.logical_and(areas1_all>np.quantile(areas1_all,0.95),np.logical_and(ij_array[:,0]==0,ij_array[:,1]==0))
# homography_mask = dists<np.quantile(dists,0.05)
dists_realfilt = dists_real[np.logical_and(homography_mask==1,in_mask)]
ij_arrayfilt = ij_array[np.logical_and(homography_mask==1,in_mask)]
kpf1complfilt = kpf1compl[np.logical_and(homography_mask==1,in_mask)]
kpf2complfilt = kpf2compl[np.logical_and(homography_mask==1,in_mask)]
np.arange(len(regions1_all))[np.logical_and(homography_mask==1,in_mask)]
regions1_allfilt = [regions1_all[k] for k in np.arange(len(regions1_all))[np.logical_and(homography_mask==1,in_mask)]]
regions2_allfilt = [regions2_all[k] for k in np.arange(len(regions2_all))[np.logical_and(homography_mask==1,in_mask)]]


tc = np.sum(np.logical_and(homography_mask==1, in_mask))>=5
if tc:
    params = cv2.UsacParams()
    params.confidence = 0.9999
    params.sampler = cv2.SAMPLING_PROSAC
    params.score = cv2.SCORE_METHOD_MAGSAC
    params.maxIterations = 1000000
    params.neighborsSearch = cv2.NEIGH_FLANN_RADIUS
    params.threshold = 1 
    params.loMethod = cv2.LOCAL_OPTIM_GC
    params.loIterations = 100
    # params.loSampleSize = 20 
    M, mask = cv2.estimateAffine2D(kpf1complfilt, kpf2complfilt, params)
    mask=mask.ravel()

mean_error = np.nan if not tc else np.mean(dists_realfilt)
q95_error = np.nan if not tc else np.quantile(dists_realfilt,0.95)
q75_error = np.nan if not tc else np.quantile(dists_realfilt,0.75)
q50_error = np.nan if not tc else np.quantile(dists_realfilt,0.5)
q25_error = np.nan if not tc else np.quantile(dists_realfilt,0.25)
q05_error = np.nan if not tc else np.quantile(dists_realfilt,0.05)
min_error = np.nan if not tc else np.min(dists_realfilt)
max_error = np.nan if not tc else np.max(dists_realfilt)

logging.info(f"median distance: {q50_error:5.3} (min: {min_error:5.3}, max: {max_error:5.3})")

global_x_shift = np.nan if not tc else M[0,2]
global_y_shift = np.nan if not tc else M[1,2]
global_affinemat = np.zeros((2,2))*np.nan if not tc else M[:2,:2]
n_points_in_global_affine = 0 if not tc else np.sum(mask)
n_points_total = len(mask)

reg_measure_dic = {
    f"{comparison_from}_to_{comparison_to}_mean_error": str(mean_error),
    f"{comparison_from}_to_{comparison_to}_quantile05_error": str(q05_error),
    f"{comparison_from}_to_{comparison_to}_quantile25_error": str(q25_error),
    f"{comparison_from}_to_{comparison_to}_quantile50_error": str(q50_error),
    f"{comparison_from}_to_{comparison_to}_quantile75_error": str(q75_error),
    f"{comparison_from}_to_{comparison_to}_quantile95_error": str(q95_error),
    f"{comparison_from}_to_{comparison_to}_global_x_shift": str(global_x_shift),
    f"{comparison_from}_to_{comparison_to}_global_y_shift": str(global_y_shift),
    f"{comparison_from}_to_{comparison_to}_global_affine_matrix_00": str(global_affinemat[0,0]),
    f"{comparison_from}_to_{comparison_to}_global_affine_matrix_01": str(global_affinemat[0,1]),
    f"{comparison_from}_to_{comparison_to}_global_affine_matrix_10": str(global_affinemat[1,0]),
    f"{comparison_from}_to_{comparison_to}_global_affine_matrix_11": str(global_affinemat[1,1]),
    f"{comparison_from}_to_{comparison_to}_n_points_in_global_affine": str(n_points_in_global_affine),
    f"{comparison_from}_to_{comparison_to}_n_points_total": str(n_points_total)
    }

logging.info("Save json")
json.dump(reg_measure_dic, open(snakemake.output["error_stats"],"w"))


# kpf1complfilt2 = kpf1complfilt[mask.ravel()==1,:]
# kpf2complfilt2 = kpf2complfilt[mask.ravel()==1,:]

# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0].imshow(microscopy_image_2, cmap="gray")
# ax[0].scatter(kpf1complfilt,kpf1complfilt,c="red")
# ax[0].set_ylim([x_segs[0],x_segs[-1]])
# ax[0].set_xlim([y_segs[0],y_segs[-1]])
# ax[1].imshow(microscopy_image_1, cmap="gray")
# ax[1].scatter(kpf2complfilt,kpf2complfilt,c="red")
# ax[1].set_ylim([x_segs[0],x_segs[-1]])
# ax[1].set_xlim([y_segs[0],y_segs[-1]])
# plt.show()


arrowed_microscopy_image_1 = np.stack([microscopy_image_1, microscopy_image_1, microscopy_image_1], axis=2)
for k in range(len(regions1_allfilt)):
    arrowed_microscopy_image_1 = cv2.drawContours(
        arrowed_microscopy_image_1, 
        [regions1_allfilt[k]], 
        -1, 
        255)

arrowed_microscopy_image_2 = np.stack([microscopy_image_2, microscopy_image_2, microscopy_image_2], axis=2)
for k in range(len(regions2_allfilt)):
    arrowed_microscopy_image_2 = cv2.drawContours(
        arrowed_microscopy_image_2, 
        [regions2_allfilt[k]], 
        -1, 
        255)

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].imshow(arrowed_microscopy_image_1[x_segs[0]:x_segs[-1],y_segs[0]:y_segs[-1]])
ax[1].imshow(arrowed_microscopy_image_2[x_segs[0]:x_segs[-1],y_segs[0]:y_segs[-1]])
plt.show()




import matplotlib as mpl
viridis = mpl.colormaps['viridis'].resampled(8)
dists_real_norm = mpl.colors.Normalize()(dists_realfilt)
dists_real_color = (viridis(dists_real_norm)*255).astype(int)[:,:3]

arrowed_microscopy_image_1 = np.stack([microscopy_image_1, microscopy_image_1, microscopy_image_1], axis=2)
for k in range(len(kpf1complfilt)):
    # tc = (int(dists_real_color[k,2]),int(dists_real_color[k,1]),int(dists_real_color[k,0]))
    tc = (int(dists_real_color[k,0]),int(dists_real_color[k,1]),int(dists_real_color[k,2]))
    arrowed_microscopy_image_1 = cv2.arrowedLine(arrowed_microscopy_image_1, pt1=kpf1complfilt[k,:].astype(int), pt2=kpf2complfilt[k,:].astype(int), color=tc, thickness=int(5/(input_spacing_1/output_spacing)), tipLength=0.3)

plt.imshow(arrowed_microscopy_image_1[x_segs[0]:x_segs[-1],y_segs[0]:y_segs[-1]])
plt.show()

saveimage_tile(arrowed_microscopy_image_1[x_segs[0]:x_segs[-1],y_segs[0]:y_segs[-1]], microscopy_file_out_1 ,1)


arrowed_microscopy_image_2 = np.stack([microscopy_image_2, microscopy_image_2, microscopy_image_2], axis=2)
for k in range(len(kpf2complfilt)):
    # tc = (int(dists_real_color[k,2]),int(dists_real_color[k,1]),int(dists_real_color[k,0]))
    tc = (int(dists_real_color[k,0]),int(dists_real_color[k,1]),int(dists_real_color[k,2]))
    arrowed_microscopy_image_2 = cv2.arrowedLine(arrowed_microscopy_image_2, pt1=kpf2complfilt[k,:].astype(int), pt2=kpf1complfilt[k,:].astype(int), color=tc, thickness=int(5/(input_spacing_1/output_spacing)), tipLength=0.3)

saveimage_tile(arrowed_microscopy_image_2[x_segs[0]:x_segs[-1],y_segs[0]:y_segs[-1]], microscopy_file_out_2 ,1)
# px.imshow(arrowed_microscopy_image_1).show()
# px.imshow(arrowed_microscopy_image_2).show()
# plt.imshow(arrowed_microscopy_image_2[x_segs[0]:x_segs[-1],y_segs[0]:y_segs[-1]])
# plt.show()

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].imshow(arrowed_microscopy_image_1[x_segs[0]:x_segs[-1],y_segs[0]:y_segs[-1]])
ax[1].imshow(arrowed_microscopy_image_2[x_segs[0]:x_segs[-1],y_segs[0]:y_segs[-1]])
plt.show()

logging.info("Finished")
