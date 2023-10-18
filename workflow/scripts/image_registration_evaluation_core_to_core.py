import matplotlib.pyplot as plt
import json
import skimage
import skimage.exposure
import re
import numpy as np
from sklearn.neighbors import KDTree
import cv2
from image_registration_IMS_to_preIMS_utils import normalize_image, readimage_crop, prepare_image_for_sam, get_angle, saveimage_tile, subtract_postIMS_grid
import sys,os
import logging, traceback
logging.basicConfig(filename=snakemake.log["stdout"],
                    level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    )
from logging_utils import handle_exception, StreamToLogger
sys.excepthook = handle_exception
sys.stdout = StreamToLogger(logging.getLogger(),logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger(),logging.ERROR)

logging.info("Start")


# microscopy_file_1 = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/preIMS/test_split_pre-preIMS_to_postIMS_registered_reduced.ome.tiff"
# microscopy_file_1 = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMC/Cirrhosis-TMA-5_New_Detector_002_transformed_on_preIMC.ome.tiff"
# microscopy_file_1 = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/preIMC/Cirrhosis-TMA-5_New_Detector_002_transformed_on_preIMS.ome.tiff"
microscopy_file_1 = snakemake.input['microscopy_image_1']

# microscopy_file_2 = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMS/test_split_pre_postIMS.ome.tiff"
# microscopy_file_2 = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/preIMS/test_split_pre_preIMS.ome.tiff"
# microscopy_file_2 = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/preIMC/test_split_pre_preIMC.ome.tiff"
# microscopy_file_2 = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/preIMS/test_split_pre_preIMS.ome.tiff"
microscopy_file_2 = snakemake.input['microscopy_image_2']

# IMC_file="/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMC_mask/Cirrhosis-TMA-5_New_Detector_002_transformed_on_postIMS.ome.tiff"
# IMC_file="/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMC_mask/Cirrhosis-TMA-5_New_Detector_002_transformed_on_preIMS.ome.tiff"
# IMC_file="/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMC_mask/Cirrhosis-TMA-5_New_Detector_002_transformed_on_preIMS.ome.tiff"
IMC_file=snakemake.input["IMC_mask"]

# input_spacing = 0.22537
input_spacing = snakemake.params["input_spacing"]
# rescale = 0.22537
rescale = snakemake.params["downscale"]


# m = re.search("[a-zA-Z]*(?=_registered_reduced.ome.tiff$)",os.path.basename(microscopy_file_1))
m = re.search("[a-zA-Z]*(?=.ome.tiff$)",os.path.basename(microscopy_file_1))
comparison_to = m.group(0)
comparison_from = os.path.basename(os.path.dirname(microscopy_file_1))
assert(comparison_to in ["preIMC","preIMS","postIMS"])
assert(comparison_from in ["postIMC","preIMC","preIMS"])


# maximum assumed distance between corresponding points
dmax = 50/(input_spacing/rescale)
dmax = snakemake.params["max_distance"]
# minimum distance between points on the same image 
dmin = 10/(input_spacing/rescale)
dmin = snakemake.params["min_distance"]

microscopy_file_out = snakemake.output['microscopy_image_out']
matching_points_filename_out = snakemake.output['matching_points']

logging.info("core bounding box extraction")
# read IMC to get bounding box (image was cropped in previous step)
tmpimg = skimage.io.imread(microscopy_file_1)
rp = skimage.measure.regionprops((np.max(tmpimg, axis=2)>0).astype(np.uint8))
bb1 = rp[0].bbox


logging.info("load IMC mask")
IMCw = readimage_crop(IMC_file, bb1)
IMCw = skimage.transform.rescale(IMCw, rescale, preserve_range = True)   
IMCw[IMCw>0]=255
IMCw = IMCw.astype(np.uint8)
imcbbox_inner = skimage.measure.regionprops(IMCw)[0].bbox
IMCw= cv2.morphologyEx(src=IMCw.astype(np.uint8), op = cv2.MORPH_DILATE, kernel = skimage.morphology.square(int(201)))
imcbbox_outer = skimage.measure.regionprops(IMCw)[0].bbox


logging.info("load microscopy image 1")
microscopy_image_1 = readimage_crop(microscopy_file_1, bb1)
microscopy_image_1 = prepare_image_for_sam(microscopy_image_1, rescale)

logging.info("load microscopy image 2")
microscopy_image_2 = readimage_crop(microscopy_file_2, bb1)
microscopy_image_2 = prepare_image_for_sam(microscopy_image_2, rescale)

print(f"snakemake params remove_postIMS_grid: {snakemake.params['remove_postIMS_grid']}")
if snakemake.params["remove_postIMS_grid"]:
    img = microscopy_image_2[imcbbox_outer[0]:imcbbox_outer[2],imcbbox_outer[1]:imcbbox_outer[3]]
    microscopy_image_2[imcbbox_outer[0]:imcbbox_outer[2],imcbbox_outer[1]:imcbbox_outer[3]] = subtract_postIMS_grid(img)



logging.info("Setup split into segments")
x_segs = np.arange(imcbbox_outer[0],imcbbox_outer[2],(imcbbox_outer[2]-imcbbox_outer[0])/8).astype(int)
x_segs = np.append(x_segs,imcbbox_outer[2])
y_segs = np.arange(imcbbox_outer[1],imcbbox_outer[3],(imcbbox_outer[3]-imcbbox_outer[1])/8).astype(int)
y_segs = np.append(y_segs,imcbbox_outer[3])

# ksize = np.round(3*(input_spacing/rescale)).astype(int)
# ksize = ksize+1 if ksize%2==0 else ksize


# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0].imshow(microscopy_image_1, cmap="gray")
# ax[0].set_ylim([x_segs[0],x_segs[-1]])
# ax[0].set_xlim([y_segs[0],y_segs[-1]])
# ax[1].imshow(microscopy_image_2, cmap="gray")
# ax[1].set_ylim([x_segs[0],x_segs[-1]])
# ax[1].set_xlim([y_segs[0],y_segs[-1]])
# plt.show()



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
n_shift = 3
logging.info("segment | n_found | n_used | dx | dy")
for j in range((len(x_segs)-n_shift)):
    for i in range((len(x_segs)-n_shift)):
        cv2.setRNGSeed(2391)
        # extract keypoints and descriptors
        img1 = microscopy_image_1[x_segs[i]:x_segs[i+n_shift],y_segs[j]:y_segs[j+n_shift]]
        # img1 = cv2.medianBlur(img1, ksize)
        # img1 = cv2.createCLAHE().apply(img1)
        # kp1,des1 = cv2.BRISK_create(thresh=50,octaves=n_shift).detectAndCompute(img1,None)
        # kp1,des1 = cv2.ORB_create().detectAndCompute(img1,None)
        # kp1,des1 = cv2.SIFT_create().detectAndCompute(img1,None)
        kp1,des1 = cv2.KAZE_create().detectAndCompute(img1,None)
        pts1 = np.float32([ m.pt for m in kp1 ])

        img2 = microscopy_image_2[x_segs[i]:x_segs[i+n_shift],y_segs[j]:y_segs[j+n_shift]]
        # img2 = cv2.medianBlur(img2, ksize)
        # img2 = cv2.createCLAHE().apply(img2)
        kp2,des2 = cv2.KAZE_create().detectAndCompute(img2,None)
        pts2 = np.float32([ m.pt for m in kp2 ])

        # Matching
        # first step: based on physical distance (i.e. smaller than dmax)
        kdt = KDTree(pts1, leaf_size=30, metric='euclidean')
        physical_distances, indices = kdt.query(pts2, k=np.min([500,len(kp1)]), return_distance=True)

        def single_match(k):  
            distances = np.linalg.norm(des1[indices[k,physical_distances[k,:]<dmax]] - des2[k], axis=1)
            ind = np.argpartition(distances, 2)[:2]
            return list(map(lambda l: cv2.DMatch(_distance=distances[l], _queryIdx=indices[k,:][l], _trainIdx=k), ind))

        matches = list(map(single_match, range(len(pts2))))

        # Apply ratio test
        matches_filt = []
        matches_filt_dist_ratio = []
        for m,n in matches:
            if m.distance < 0.8*n.distance:
                matches_filt.append(m)
                matches_filt_dist_ratio.append(m.distance/n.distance)

        src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches_filt ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches_filt ]).reshape(-1,1,2)


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

        # third step: remove points close to each other (i.e. smaller than dmin)
        ctr1 = -1
        ctr2 = 0
        to_remove = np.zeros(0)
        while ctr1 < ctr2:
            # filter points
            to_keep = np.array([k for k in range(len(matches_filt)) if k not in to_remove])
            tmp_matches_filt = matches_filt[to_keep]

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
        
        matches_filt = np.array(matches_filt)[to_keep]
        matches_filt_dist_ratio = np.array(matches_filt_dist_ratio)[to_keep]
        src_pts = src_pts[to_keep,:,:]  
        dst_pts = dst_pts[to_keep,:,:]  


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

        kpf1 = np.array([kp1[matches_filt[k].queryIdx].pt for k in range(len(matches_filt))])
        kpf2 = np.array([kp2[matches_filt[k].trainIdx].pt for k in range(len(matches_filt))])
        dists_real = np.append(dists_real,np.sqrt(np.sum((kpf1-kpf2)**2,axis=1)))
        angles = np.append(angles,np.array([get_angle([-10000,kpf1[k,1]],kpf1[k,:],kpf2[k,:]) for k in range(kpf1.shape[0])]))

        kpf1 += np.array([y_segs[j],x_segs[i]])
        kpf2 += np.array([y_segs[j],x_segs[i]])
        kpf1compl = np.append(kpf1compl,kpf1,axis=0)
        kpf2compl = np.append(kpf2compl,kpf2,axis=0)
        
        ij_array = np.append(ij_array,np.vstack([np.zeros(len(kpf1),dtype=int)+i,np.zeros(len(kpf1),dtype=int)+j]).T,axis=0)

        homography_mask = np.append(homography_mask,matchesMask,axis=0)
        matdic[f"{i}_{j}"] = {'matrix': M, 'i':i, 'j':j, 'n': np.sum(matchesMask)}

combined_output = np.hstack([kpf1compl,kpf2compl,dists.reshape(-1,1),dist_ratios.reshape(-1,1),homography_mask.reshape(-1,1),dists_real.reshape(-1,1),angles.reshape(-1,1),ij_array])
np.savetxt(matching_points_filename_out,combined_output,header=f"p1x,p1y,p2x,p2y,distance,distance_ratio,homography_mask,distance_physical,angle,i,j",delimiter=',')

dists_realfilt = dists_real[homography_mask==1]
ij_arrayfilt = ij_array[homography_mask==1]
kpf1complfilt = kpf1compl[homography_mask==1]
kpf2complfilt = kpf2compl[homography_mask==1]

logging.info(f"median distance: {np.median(dists_realfilt):5.3} (min: {np.min(dists_realfilt):5.3}, max: {np.max(dists_realfilt):5.3})")


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

mean_error = np.mean(dists_realfilt)
q95_error = np.quantile(dists_realfilt,0.95)
q75_error = np.quantile(dists_realfilt,0.75)
q50_error = np.quantile(dists_realfilt,0.5)
q25_error = np.quantile(dists_realfilt,0.25)
q05_error = np.quantile(dists_realfilt,0.05)

global_x_shift = M[0,2]
global_y_shift = M[1,2]
global_affinemat = M[:2,:2]
n_points_in_global_affine= np.sum(mask)
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


import matplotlib as mpl
viridis = mpl.colormaps['viridis'].resampled(8)
dists_real_norm = mpl.colors.Normalize()(dists_real)
dists_real_color = (viridis(dists_real_norm)*255).astype(int)[:,:3]

arrowed_microscopy_image_1 = np.stack([microscopy_image_1, microscopy_image_1, microscopy_image_1], axis=2)
for k in range(len(kpf1complfilt)):
    # tc = (int(dists_real_color[k,2]),int(dists_real_color[k,1]),int(dists_real_color[k,0]))
    tc = (int(dists_real_color[k,0]),int(dists_real_color[k,1]),int(dists_real_color[k,2]))
    arrowed_microscopy_image_1 = cv2.arrowedLine(arrowed_microscopy_image_1, pt1=kpf1complfilt[k,:].astype(int), pt2=kpf2complfilt[k,:].astype(int), color=tc, thickness=int(5/(input_spacing/rescale)), tipLength=0.3)

# plt.imshow(arrowed_microscopy_image_1[x_segs[0]:x_segs[-1],y_segs[0]:y_segs[-1]])
# plt.show()

saveimage_tile(arrowed_microscopy_image_1[x_segs[0]:x_segs[-1],y_segs[0]:y_segs[-1]], microscopy_file_out ,1)


arrowed_microscopy_image_2 = np.stack([microscopy_image_2, microscopy_image_2, microscopy_image_2], axis=2)
for k in range(len(kpf2complfilt)):
    # tc = (int(dists_real_color[k,2]),int(dists_real_color[k,1]),int(dists_real_color[k,0]))
    tc = (int(dists_real_color[k,0]),int(dists_real_color[k,1]),int(dists_real_color[k,2]))
    arrowed_microscopy_image_2 = cv2.arrowedLine(arrowed_microscopy_image_2, pt1=kpf2complfilt[k,:].astype(int), pt2=kpf1complfilt[k,:].astype(int), color=tc, thickness=int(5/(input_spacing/rescale)), tipLength=0.3)

# plt.imshow(arrowed_microscopy_image_2[x_segs[0]:x_segs[-1],y_segs[0]:y_segs[-1]])
# plt.show()

# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0].imshow(arrowed_microscopy_image_1[x_segs[0]:x_segs[-1],y_segs[0]:y_segs[-1]])
# ax[1].imshow(arrowed_microscopy_image_2[x_segs[0]:x_segs[-1],y_segs[0]:y_segs[-1]])
# plt.show()

