# from wsireg.writers.ome_tiff_writer import OmeTiffWriter
# from wsireg.reg_transforms.reg_transform_seq import RegTransform, RegTransformSeq
# from wsireg.parameter_maps.transformations import BASE_RIG_TFORM
# from wsireg.reg_images.loader import reg_image_loader
from rembg import remove, new_session
from segment_anything import sam_model_registry, SamPredictor
import torch
import cv2
import skimage
import numpy as np
# from imc_to_ims_workflow.workflow.scripts.image_registration_IMS_to_preIMS_utils import *
from image_registration_IMS_to_preIMS_utils import *
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

# prepare model for rembg
model_name = "isnet-general-use"
os.environ["OMP_NUM_THREADS"] = str(snakemake.threads)
rembg_session = new_session(model_name)
# CHECKPOINT_PATH = "/home/retger/Downloads/sam_vit_h_4b8939.pth"
# CHECKPOINT_PATH = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/sam_weights/sam_vit_h_4b8939.pth"
CHECKPOINT_PATH = snakemake.input["sam_weights"]
DEVICE = 'cpu'
MODEL_TYPE = "vit_h"
torch.set_num_threads(snakemake.threads)

# parameters
#stepsize = 30
stepsize = float(snakemake.params["IMS_pixelsize"])
#pixelsize = 24
pixelsize = stepsize*float(snakemake.params["IMS_shrink_factor"])
#resolution = 1
# resolution=0.22537
resolution = float(snakemake.params["microscopy_pixelsize"])

# postIMS_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMS/test_split_pre_postIMS_reduced.ome.tiff"
# postIMS_file = "/home/retger/Downloads/Lipid_TMA_3781_postIMS_reduced.ome.tiff"
postIMS_file = snakemake.input["postIMS_downscaled"]
# postIMSr_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMS/test_split_pre_postIMS_reduced_mask.ome.tiff"
postIMSr_file = snakemake.output["postIMSmask_downscaled"]

# imc_mask_files = [f"/home/retger/Downloads/Lipid_TMA_37819_0{i}_transformed.ome.tiff" for i in ["09"] + list(range(10,43))]
# imc_mask_files = [ f"/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMC_mask/Cirrhosis-TMA-5_New_Detector_00{i}_transformed.ome.tiff" for i in [1,2]]
imc_mask_files = snakemake.input["IMCmask"]
logging.info("Read and process image")
# read postIMS image
postIMS = skimage.io.imread(postIMS_file)
postIMS = prepare_image_for_sam(postIMS, 1)
# postIMS = prepare_image_for_sam(postIMS, resolution)

# postIMSmpre = skimage.filters.median(postIMS, skimage.morphology.disk( np.floor(((stepsize-pixelsize)/resolution)/3)))
# postIMS2r = skimage.filters.median(postIMS, skimage.morphology.disk(int((1/resolution) * (pixelsize/4))))
# postIMS2r = np.stack([postIMS, postIMS, postIMS], axis=2)



logging.info("Read IMC masks")
imcbboxls = list()
for imcmaskfile in imc_mask_files:
    imc=skimage.io.imread(imcmaskfile)
    imc[imc>0]=255
    imc = imc.astype(np.uint8)
    imcbboxls.append(skimage.measure.regionprops(imc)[0].bbox)

ksize = int((1/resolution) * (stepsize/2))
ksize = ksize+1 if ksize%2==0 else ksize
logging.info(f"Median filter postIMS with radius: {ksize}px")
# postIMS2r = skimage.filters.median(postIMS, skimage.morphology.disk(med_radius))
# postIMS2r = np.stack([postIMS2r, postIMS2r, postIMS2r], axis=2)
postIMS2r = cv2.medianBlur(postIMS, ksize)

logging.info("Remove background")
# remove background, i.e. detect cores
postIMSr = remove(postIMS2r, only_mask=True, session=rembg_session,post_process_mask=True)
postIMSr = postIMSr>127
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0].imshow(postIMS2r, cmap='gray')
# ax[0].set_title("postIMS")
# ax[1].imshow(postIMSr, cmap='gray')
# ax[1].set_title("postIMSmask")
# plt.show()


# fill holes in tissue mask
logging.info("Fill holes in mask")
postIMSr = skimage.morphology.remove_small_holes(postIMSr,1000**2*np.pi*(1/resolution))

logging.info("Check mask")
IMC_filled = list()
for i in range(len(imcbboxls)):
    xmin = int(imcbboxls[i][0])
    ymin = int(imcbboxls[i][1])
    xmax = int(imcbboxls[i][2])
    ymax = int(imcbboxls[i][3])
    IMC_filled.append(np.min(postIMSr[xmin:xmax,ymin:ymax]) == 1)
# IMC_filled

# if np.sum(IMC_filled)<len(IMC_filled):
# logging.info("Mask detection doesn't contain all IMC regions")
logging.info("Remove background individually for each IMC location with rembg")
postIMSstitch = postIMS2r[:,:]*0
postIMSmask = postIMS2r[:,:]*0
# img_ls = []
rembg_mask_areas = []
for i in range(len(imcbboxls)):
    #for expand in [750,1000,1250,1500,1750]:
    for expand in [int(r/resolution) for r in [1500]]:
        xmin = int(imcbboxls[i][0]-expand)
        xmin = xmin if xmin>=0 else 0
        ymin = int(imcbboxls[i][1]-expand)
        ymin = ymin if ymin>=0 else 0
        xmax = int(imcbboxls[i][2]+expand)
        xmax = xmax if xmax<=postIMS2r.shape[0] else postIMS2r.shape[0]
        ymax = int(imcbboxls[i][3]+expand)
        ymax = ymax if ymax<=postIMS2r.shape[1] else postIMS2r.shape[1]
        print(f"i: {i}, {os.path.basename(imc_mask_files[i])}, coords:[{xmin}:{xmax},{ymin}:{ymax}]")
        postIMSmask[(xmin+expand):(xmax-expand),(ymin+expand):(ymax-expand)] = 255
        tmpimg = remove(postIMS2r[xmin:xmax,ymin:ymax], only_mask=True, session=rembg_session,post_process_mask=True)
        # plt.imshow(tmpimg>0)
        # plt.show()
        tmpimg2 = np.zeros((tmpimg.shape[0]+10,tmpimg.shape[1]+10))
        tmpimg2[5:(tmpimg.shape[0]+5),5:(tmpimg.shape[1]+5)] = tmpimg
        tmpimg = skimage.morphology.remove_small_holes(tmpimg2>0,500**2*np.pi*(1/resolution))[5:(tmpimg.shape[0]+5),5:(tmpimg.shape[1]+5)]
        tmpimg = tmpimg.astype(np.uint8)*255
        tmpimg = preprocess_mask(tmpimg,1)
        tmpimg = tmpimg.astype(np.uint8)*255
        tmpimg2 = np.zeros(np.array(list(tmpimg.shape))+20, dtype=np.uint8)
        tmpimg2[10:(tmpimg2.shape[0]-10),10:(tmpimg2.shape[1]-10)] = tmpimg.astype(np.uint8)
        tmpimg2 = skimage.morphology.convex_hull_image(tmpimg2>127)
        tmpimg2 = tmpimg2.astype(np.uint8)*255
        tmpimg = tmpimg2[10:(tmpimg2.shape[0]-10),10:(tmpimg2.shape[1]-10)]
        # img_ls.append(tmpimg)
        rembg_mask_areas.append(np.sum(tmpimg>0))
        # TODO: check if touches border
        # max_border_value = np.max([np.max(tmpimg[0,:]),np.max(tmpimg[-1,:]),np.max(tmpimg[:,0]),np.max(tmpimg[:,-1])])
        # if max_border_value == 0:
        postIMSstitch[xmin:xmax,ymin:ymax] = np.max(np.stack([postIMSstitch[xmin:xmax,ymin:ymax],tmpimg], axis=0),axis=0)
# import matplotlib.pyplot as plt
# ind = 0
# fig, ax = plt.subplots(nrows=3, ncols=3)
# for i in range(3):
#     for j in range(3):
#         ax[i,j].imshow(img_ls[ind], cmap='gray')
#         ax[i,j].set_title(f"{ind}")
#         ind+=1
# plt.show()

postIMSrs = postIMSstitch>0
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0].imshow(postIMSstitch, cmap='gray')
# ax[0].set_title("postIMS")
# ax[1].imshow(postIMSr, cmap='gray')
# ax[1].set_title("postIMSmask")
# plt.show()


logging.info("Check mask")
IMCrs_filled = list()
for i in range(len(imcbboxls)):
    xmin = int(imcbboxls[i][0])
    ymin = int(imcbboxls[i][1])
    xmax = int(imcbboxls[i][2])
    ymax = int(imcbboxls[i][3])
    IMCrs_filled.append(np.min(postIMSrs[xmin:xmax,ymin:ymax]) == 1)
# IMC_filled

    # if np.sum(IMC_filled)<len(IMC_filled):
# inds = np.array(list(range(len(imcbboxls))))[np.logical_not(np.array(IMC_filled))]
postIMSstitch = postIMS2r[:,:]*0
inds = np.array(list(range(len(imcbboxls))))
# logging.info(f"Number of incomplete masks found: {len(inds)}")
logging.info("Run segment anything model on IMC locations")
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_mask_areas = []
for i in inds:
    xmin = int(imcbboxls[i][0]-1500/resolution)
    xmin = xmin if xmin>=0 else 0
    ymin = int(imcbboxls[i][1]-1500/resolution)
    ymin = ymin if ymin>=0 else 0
    xmax = int(imcbboxls[i][2]+1500/resolution)
    xmax = xmax if xmax<=postIMS2r.shape[0] else postIMS2r.shape[0]
    ymax = int(imcbboxls[i][3]+1500/resolution)
    ymax = ymax if ymax<=postIMS2r.shape[1] else postIMS2r.shape[1]
    print(f"i: {i}, {os.path.basename(imc_mask_files[i])}, coords:[{xmin}:{xmax},{ymin}:{ymax}]")
    saminp = np.stack([postIMS2r[xmin:xmax,ymin:ymax],postIMS2r[xmin:xmax,ymin:ymax],postIMS2r[xmin:xmax,ymin:ymax]], axis=2)
    postIMSmasks, scores1 = sam_core(saminp, sam)
    postIMSmasks = np.stack([preprocess_mask(msk,1) for msk in postIMSmasks ])
    tmpareas = np.array([np.sum(im) for im in postIMSmasks])
    imcarea = (imcbboxls[i][2]-imcbboxls[i][0])*(imcbboxls[i][3]-imcbboxls[i][1])
    tmpinds = np.array(list(range(3)))
    tmpinds = tmpinds[tmpareas > imcarea*1.02]
    tmpind = tmpinds[scores1[tmpinds]==np.max(scores1[tmpinds])]
    tmpimg = postIMSmasks[tmpind,:,:][0,:,:].astype(np.uint8)*255
    sam_mask_areas.append(np.sum(tmpimg>0))
    postIMSstitch[xmin:xmax,ymin:ymax] = np.max(np.stack([postIMSstitch[xmin:xmax,ymin:ymax],tmpimg], axis=0),axis=0)

    postIMSsamr = postIMSstitch>0
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(nrows=1, ncols=3)
    # ax[0].imshow(postIMSr, cmap='gray')
    # ax[0].set_title("postIMS")
    # ax[1].imshow(postIMSrs, cmap='gray')
    # ax[1].set_title("postIMSmask")
    # ax[2].imshow(postIMSsamr, cmap='gray')
    # ax[2].set_title("postIMSmask")
    # plt.show()

logging.info("Check mask")
IMCrsam_filled = list()
for i in range(len(imcbboxls)):
    xmin = int(imcbboxls[i][0])
    ymin = int(imcbboxls[i][1])
    xmax = int(imcbboxls[i][2])
    ymax = int(imcbboxls[i][3])
    IMCrsam_filled.append(np.min(postIMSsamr[xmin:xmax,ymin:ymax]) == 1)
# IMC_filled


tmpbool = np.logical_not(np.logical_and(IMCrs_filled,IMCrsam_filled))
inds = np.array(list(range(len(imcbboxls))))[tmpbool]
if len(inds)>0:
    logging.info(f"The following masks were not found in both rembg and sam:")
    for i in inds:
        logging.info(f"{os.path.basename(imc_mask_files[i])};\trembg: {IMCrs_filled[i]};\tSAM:{IMCrsam_filled[i]}")

ratio_sam_to_rembg = [np.log10(sam_mask_areas[i]/rembg_mask_areas[i]) for i in range(len(imcbboxls))]
logging.info(f"Difference of the mask areas:")
logging.info(f"SAM-rembg\t,(SAM-rembg)/(0.5*(SAM+rembg))\t,Name")
for i in range(len(imcbboxls)):
    logging.info(f"{sam_mask_areas[i]-rembg_mask_areas[i]}\t, {(sam_mask_areas[i]-rembg_mask_areas[i])/(0.5*(sam_mask_areas[i]+rembg_mask_areas[i])):.4f}\t, {os.path.basename(imc_mask_files[i])}")
    # if (ratio_sam_to_rembg[i]<np.log10(1/1.1)) or (ratio_sam_to_rembg[i]>np.log10(1.1/1)):
    # logging.info(f"Difference of the mask areas (SAM-rembg) for {os.path.basename(imc_mask_files[i])}: {sam_mask_areas[i]-rembg_mask_areas[i]} ((SAM-rembg)/(0.5*(SAM+rembg)): {(sam_mask_areas[i]-rembg_mask_areas[i])/(0.5*(sam_mask_areas[i]+rembg_mask_areas[i])):.4f}")



postIMSr = postIMSsamr

logging.info(f"Get convex hull")
lbs = skimage.measure.label(postIMSr)
rps = skimage.measure.regionprops(lbs)
cvi = lbs*0
for i in range(len(rps)):
    tbb = rps[i].bbox
    ti = skimage.morphology.convex_hull_image(lbs[tbb[0]:tbb[2],tbb[1]:tbb[3]]==rps[i].label)
    cvi[tbb[0]:tbb[2],tbb[1]:tbb[3]] = np.logical_or(ti,cvi[tbb[0]:tbb[2],tbb[1]:tbb[3]])
# plt.imshow(cvi)
# plt.show()
logging.info("Save mask")

saveimage_tile(cvi, postIMSr_file, resolution)
# empty_transform = BASE_RIG_TFORM
# empty_transform['Spacing'] = (str(resolution),str(resolution))
# empty_transform['Size'] = (cvi.shape[1], cvi.shape[0])
# rt = RegTransform(empty_transform)
# rts = RegTransformSeq(rt,[0])
# ri = reg_image_loader(cvi.astype(np.uint8), resolution)
# writer = OmeTiffWriter(ri, reg_transform_seq=rts)
# img_basename = os.path.basename(postIMSr_file).split(".")[0]
# img_dirname = os.path.dirname(postIMSr_file)
# writer.write_image_by_plane(img_basename, output_dir=img_dirname, tile_size=1024)


logging.info("Finished")
