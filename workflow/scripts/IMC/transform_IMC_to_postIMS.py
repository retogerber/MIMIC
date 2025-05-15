import sys,os
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "..","..","workflow","scripts","utils")))
from image_utils import get_image_shape, readimage_crop, saveimage_tile
import numpy as np
from wsireg.reg_transforms.reg_transform import RegTransform
from wsireg.reg_transforms.reg_transform_seq import RegTransformSeq
from shapely.geometry import shape
import json
import SimpleITK as sitk
import tifffile
from utils import setNThreads, snakeMakeMock
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["input_spacing"] = 1 
    snakemake.params["TMA_location_spacing"] = 0.22537
    snakemake.params["output_spacing"] = 0.22537
    snakemake.params["transform_target"] = "postIMC"

    snakemake.input["IMC"] = [f"/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMC_mask/{sample}.tiff" for sample in ["Cirrhosis-TMA-5_New_Detector_001","Cirrhosis-TMA-5_New_Detector_002"]]
    snakemake.input["IMC_to_postIMC_transform"] = [f'results/test_split_pre/registrations/IMC_to_postIMC/test_split_pre_{core}/test_split_pre_{core}-IMC_to_postIMC_transformations.json' for core in ["A1","B1"]]
    snakemake.input["postIMC_to_postIMS_transform"] = [f"/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/registrations/postIMC_to_postIMS/{core}/test_split_pre_{core}-postIMC_to_postIMS_transformations_mod.json" for core in ["A1","B1"]]
    snakemake.input["TMA_location_target"] = [f"/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/TMA_location/test_split_pre_TMA_location_on_postIMC_{core}.geojson" for core in ["A1","B1"]]
    snakemake.input["microscopy_target_image"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMC/test_split_pre_postIMC.ome.tiff"

    snakemake.output["IMC_transformed"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMC/test_split_pre_IMC_on_postIMC.ome.tiff"
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)


# params
input_spacing = snakemake.params["input_spacing"]
output_spacing = snakemake.params["output_spacing"]
TMA_location_spacing = snakemake.params["TMA_location_spacing"]
transform_target = snakemake.params["transform_target"]

# inputs
transform_file_postIMC_to_postIMS=snakemake.input["postIMC_to_postIMS_transform"]
if isinstance(transform_file_postIMC_to_postIMS, str):
    transform_file_postIMC_to_postIMS = [transform_file_postIMC_to_postIMS]
transform_file_IMC_to_postIMC=snakemake.input["IMC_to_postIMC_transform"]
if isinstance(transform_file_IMC_to_postIMC, str):
    transform_file_IMC_to_postIMC = [transform_file_IMC_to_postIMC]
imc_file = snakemake.input["IMC"]
if isinstance(imc_file, str):
    imc_file = [imc_file]
TMA_target_geojson_file=snakemake.input['TMA_location_target']
if isinstance(TMA_target_geojson_file, str):
    TMA_target_geojson_file = [TMA_target_geojson_file]
microscopy_target_image=snakemake.input["microscopy_target_image"]


# outputs
img_filename_out = snakemake.output["IMC_transformed"]
if isinstance(img_filename_out, list):
    img_filename_out = img_filename_out[0]
img_basename = os.path.basename(img_filename_out).split(".")[0]
img_dirname = os.path.dirname(img_filename_out)

# get info of IMC location
logging.info("Create bounding box")
max_shape = get_image_shape(microscopy_target_image)
logging.info(f"microscopy image size: {max_shape}")
max_shape = [int(np.ceil(x/(output_spacing/TMA_location_spacing))) for x in max_shape]
logging.info(f"microscopy image size scaled: {max_shape}")
bb_target_ls = list()
for single_TMA_geojson_file in TMA_target_geojson_file:
    TMA_geojson = json.load(open(single_TMA_geojson_file, "r"))
    if isinstance(TMA_geojson,list):
        TMA_geojson=TMA_geojson[0]
    TMA_geojson_polygon = shape(TMA_geojson['geometry'])
    logging.info(f"Setup bounding box for {os.path.basename(single_TMA_geojson_file)}")
    # bounding box
    bb1 = TMA_geojson_polygon.bounds
    logging.info(f"\tbbox before scaling: {bb1}")
    # reorder axis
    bb1 = np.array([bb1[1],bb1[0],bb1[3],bb1[2]])/(output_spacing/TMA_location_spacing)
    bb1 = bb1.astype(int)
    if bb1[0]<0:
        logging.info(f"\tbb1[0] < 0: {bb1[0]}")
        bb1[0]=0
    if bb1[1]<0:
        logging.info(f"\tbb1[1] < 0: {bb1[1]}")
        bb1[1]=0
    if bb1[2]>max_shape[0]:
        logging.info(f"\tbb1[2] > max_shape[0]: {bb1[2]} > {max_shape[0]}")
        bb1[2]=max_shape[0]
    if bb1[3]>max_shape[1]:
        logging.info(f"\tbb1[3] > max_shape[1]: {bb1[3]} > {max_shape[1]}")
        bb1[3]=max_shape[1]
    logging.info(f"\tbbox after scaling: {bb1}")
    bb_target_ls.append(bb1)


logging.info("Load transform")
rtlsls = list()
for single_transform_file_postIMC_to_postIMS in transform_file_postIMC_to_postIMS:
    logging.info(f"Setup transformation for {os.path.basename(single_transform_file_postIMC_to_postIMS)}")
    if transform_target == "postIMC":
        rtlsls.append(RegTransformSeq(None)) 
        continue
    else:
        # setup transformation sequence
        rtsn=RegTransformSeq(single_transform_file_postIMC_to_postIMS)
        rtsn.set_output_spacing((float(output_spacing),float(output_spacing)))

        rtls = rtsn.reg_transforms
        all_linear = np.array([r.is_linear for r in rtls]).all()
        if all_linear:
            assert(len(rtls)==5 or len(rtls)==3)
            is_split_transform = len(rtls)==5
        else:
            # len=4 : direct registration
            # len=6 : additional separate registration between preIMC and preIMS
            assert(len(rtls)==6 or len(rtls)==4)
            is_split_transform = len(rtls)==6


        if transform_target == "preIMC":
            n_end = 1
        elif transform_target == "preIMS":
            if all_linear:
                n_end = 4 if is_split_transform else 2
            else:
                n_end = 5 if is_split_transform else 3
        elif transform_target == "postIMS":
            if all_linear:
                n_end = 5 if is_split_transform else 3
            else:
                n_end = 6 if is_split_transform else 4
        else:
            raise ValueError("Unknown transform target: " + transform_target)

        # start is always postIMC
        n_start = 0

        rtls = rtsn.reg_transforms
        rtls = rtls[n_start:n_end]
        rtsn = RegTransformSeq(rtls, transform_seq_idx=list(range(len(rtls))))
        if len(rtls)>0:
            rtsn.set_output_spacing((float(output_spacing),float(output_spacing)))
        rtlsls.append(rtsn)


from wsireg.reg_transforms.reg_transform_seq import RegTransform
logging.info("Load transform IMC to postIMC")
rtimclsls = list()
for single_transform_file_IMC_to_postIMC in transform_file_IMC_to_postIMC:
    rtimclsls.append(RegTransform(json.load(open(single_transform_file_IMC_to_postIMC,"r"))))


def get_resampler(transform, fixed, default_value=0.0):
    resampler = sitk.ResampleImageFilter()
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetTransform(transform)
    resampler.SetReferenceImage(fixed)
    resampler.SetOutputSpacing(fixed.GetSpacing())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return resampler

def resample_image(resampler, moving):
    return sitk.GetArrayFromImage(resampler.Execute(moving))

tmp_imc_dim = get_image_shape(imc_file[0])
if len(tmp_imc_dim)==2:
    n_channels = 1
else:
    chind = np.argmin(tmp_imc_dim)
    n_channels = tmp_imc_dim[chind]


img_dtype = readimage_crop(imc_file[0], [0,0,1,1]).dtype
# get output shape
try:
    output_shape = get_image_shape(microscopy_target_image)
    output_shape = [n_channels, int(np.ceil(output_shape[0]/(output_spacing/TMA_location_spacing))),int(np.ceil(output_shape[1]/(output_spacing/TMA_location_spacing)))]
    logging.info("Size from microscopy image")
except:
    if rtlsls[0].composite_transform is None:
        output_shape = [n_channels, np.max([bb[2] for bb in bb_target_ls]), np.max([bb[3] for bb in bb_target_ls])]
        logging.info("Size from max bb_target_ls")
    else:
        output_shape = [n_channels, rtsn.output_size[0], rtsn.output_size[1]]
        logging.info("Size from composite transform")

# init output image
out_image = np.zeros(output_shape, dtype=img_dtype)    
logging.info(f"Output shape: {output_shape}")
# loop over imc images
for i in range(len(rtlsls)):
    logging.info(f"Transform image {i}")
    logging.info(f"\tbb_target: {bb_target_ls[i]}")
    logging.info(f"\tcreate composite transform")
    rtsn = rtlsls[i]
    rtimcsn = rtimclsls[i]

    # init composite transform
    composite = sitk.CompositeTransform(2)
    # add IMC to postIMC transform
    composite.AddTransform(rtimcsn.itk_transform)
    # add all transforms between source and target
    if rtsn.composite_transform is None:
        trls = []
    else:
        trls = [rtsn.composite_transform.GetNthTransform(j) for j in range(rtsn.composite_transform.GetNumberOfTransforms())]
    for j,trl in enumerate(trls):
        composite.AddTransform(trl)
    # add translation to target
    composite.AddTransform(sitk.TranslationTransform(2, [float(bb_target_ls[i][1])*output_spacing, float(bb_target_ls[i][0])*output_spacing]))


    logging.info(f"\tnumber of transforms: {composite.GetNumberOfTransforms()}")
    logging.info(f"\ttransforms:")
    trls = [composite.GetNthTransform(j) for j in range(composite.GetNumberOfTransforms())]
    for j,trl in enumerate(trls):
        if trl.IsLinear():
            logging.info(f"\t\t{j}: linear, {trl.GetParameters()}")
        else:
            logging.info(f"\t\t{j}: non-linear")

    logging.info(f"\tcreate resampler")
    if rtsn.composite_transform is None:
        tmpimg = sitk.Image([1,1], sitk.sitkFloat32)
        tmpimg.SetSpacing([output_spacing,output_spacing])
        resampler = get_resampler(composite, tmpimg)
    else:
        resampler = rtsn.resampler

    logging.info(f"\torigin: {resampler.GetOutputOrigin()}")
    logging.info(f"\tspacing: {resampler.GetOutputSpacing()}")
    logging.info(f"\tprevious size: {resampler.GetSize()}")
    newsize = np.array([bb_target_ls[i][3]-bb_target_ls[i][1], bb_target_ls[i][2]-bb_target_ls[i][0]], dtype='int').tolist()
    # newsize=np.array(output_shape[:2], dtype=int).tolist()
    resampler.SetSize(newsize)
    logging.info(f"\tcropped size: {resampler.GetSize()}")
    resampler.SetTransform(composite)

    logging.info(f"\tread image")
    moving_np = tifffile.imread(imc_file[i])
    logging.info(f"\timage shape before swap: {moving_np.shape}")
    if len(moving_np.shape)==2:
        moving_np_swap = moving_np
    else:
        moving_np_swap = np.moveaxis(moving_np,chind,-1)
    logging.info(f"\timage shape after swap: {moving_np_swap.shape}")


    imc_xmax = moving_np_swap.shape[0]
    imc_ymax = moving_np_swap.shape[1]
    logging.info(f"\tIMC cornerpoints: {[[0,0],[imc_xmax,0],[0,imc_ymax],[imc_xmax,imc_ymax]]}")
    cornerpoints = np.stack([np.array(composite.GetInverse().TransformPoint([x,y])) for x,y in [[0,0],[imc_xmax,0],[0,imc_ymax],[imc_xmax,imc_ymax]]])
    logging.info(f"\tIMC transformed cornerpoints: {cornerpoints}")
    assert np.all(cornerpoints>=0)

    logging.info(f"\tloop over channels")
    for ch in range(n_channels):
        logging.info(f"\t\tchannel {ch}")
        if n_channels==1:
            moving = sitk.GetImageFromArray(moving_np_swap)
        else:
            moving = sitk.GetImageFromArray(moving_np_swap[:,:,ch])
        moving.SetSpacing([input_spacing, input_spacing])

        logging.info(f"\t\tresample image")
        source_image_trans = resample_image(resampler, moving)
        prop_nonzero = np.sum(source_image_trans>0)/np.prod(source_image_trans.shape)
        logging.info(f"\t\tproportion of non-zero pixels: {prop_nonzero}")
        assert prop_nonzero>0

        logging.info(f"\t\tadd image to output")
        out_image[ch, bb_target_ls[i][0]:bb_target_ls[i][2], bb_target_ls[i][1]:bb_target_ls[i][3]] = source_image_trans


# postimc=tifffile.imread("/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMC/test_split_pre_postIMC.ome.tiff")
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1,3)
# ax[0].imshow(out_image[:,:,0]>0, interpolation="none")
# ax[1].imshow(postimc[:,:,0], interpolation="none")
# ax[2].imshow((out_image[:,:,0]>0)*127+postimc[:,:,0]/2, interpolation="none")
# fig.set_size_inches(30,10)
# plt.savefig(f"test.png")

saveimage_tile(out_image, filename= img_filename_out, resolution=output_spacing, dtype=img_dtype, is_rgb=False)




































# # params
# input_spacing = snakemake.params["input_spacing"]
# output_spacing = snakemake.params["output_spacing"]
# transform_target = snakemake.params["transform_target"]

# # inputs
# transform_file_IMC_to_preIMC=snakemake.input["IMC_to_preIMC_transform"]
# orig_size_tform_IMC_to_preIMC=snakemake.input["preIMC_orig_size_transform"]
# transform_file_preIMC_to_postIMS=snakemake.input["preIMC_to_postIMS_transform"]
# img_file = snakemake.input["IMC"]

# # outputs
# img_out = snakemake.output["IMC_transformed"]
# img_basename = os.path.basename(img_out).split('.')[0]
# img_dirname = os.path.dirname(img_out)

# logging.info("Read Transformation 1")
# if os.path.getsize(transform_file_IMC_to_preIMC)>0:
# # transform sequence IMC to preIMC
#     try:
#         rts = RegTransformSeq(transform_file_IMC_to_preIMC)
#         read_rts_error=False
#     except:
#         read_rts_error=True
#     try:
#         tmptform = json.load(open(transform_file_IMC_to_preIMC, "r"))
#         print("tmptform")
#         print(tmptform)
#         tmprt = RegTransform(tmptform)
#         rts=RegTransformSeq([tmprt], transform_seq_idx=[0])
#         read_rts_error=False
#     except:
#         read_rts_error=True
#     if read_rts_error:
#         exit("Could not read transform data transform_file_IMC_to_preIMC: " + transform_file_IMC_to_preIMC)

# else:
#     print("Empty File!")
#     rts = RegTransformSeq()
    
# logging.info("Read Transformation 2")
# # read transform sequence preIMC
# osize_tform = json.load(open(orig_size_tform_IMC_to_preIMC, "r"))
# osize_tform_rt = RegTransform(osize_tform)
# osize_rts = RegTransformSeq([osize_tform_rt], transform_seq_idx=[0])
# rts.append(osize_rts)

# rts.set_output_spacing((float(output_spacing),float(output_spacing)))
  
# logging.info("Read Transformation 3")
# # read in transformation sequence from preIMC to postIMS
# rtsn=RegTransformSeq(transform_file_preIMC_to_postIMS)
# rtsn.set_output_spacing((float(output_spacing),float(output_spacing)))

# logging.info("Setup transformation")
# # setup transformation sequence
# rtsn=RegTransformSeq(transform_file_preIMC_to_postIMS)
# rtsn.set_output_spacing((float(output_spacing),float(output_spacing)))

# if transform_target != "postIMS":

#     rtls = rtsn.reg_transforms
#     all_linear = np.array([r.is_linear for r in rtls]).all()
#     if all_linear:
#         assert(len(rtls)==4 or len(rtls)==2)
#         is_split_transform = len(rtls)==4
#     else:
#         # len=4 : direct registration
#         # len=6 : additional separate registration between preIMC and preIMS
#         assert(len(rtls)==5 or len(rtls)==3)
#         is_split_transform = len(rtls)==5

#     if transform_target == "preIMC":
#         n_end = 0
#     elif transform_target == "preIMS":
#         if all_linear:
#             n_end = 3 if is_split_transform else 1
#         else:
#             n_end = 4 if is_split_transform else 2
#     else:
#         raise ValueError("Unknown transform target: " + transform_target)

#     rtls = rtsn.reg_transforms
#     rtls = rtls[:n_end]
#     rtsn = RegTransformSeq(rtls, transform_seq_idx=list(range(len(rtls))))
#     if len(rtls)>0:
#         rtsn.set_output_spacing((float(output_spacing),float(output_spacing)))

# # combine transformations
# rts.append(rtsn)
# rts.set_output_spacing((float(output_spacing),float(output_spacing)))

# logging.info("Transform and save image")
# ri = reg_image_loader(img_file, float(input_spacing))#,preprocessing=ipp)
# writer = OmeTiffWriter(ri, reg_transform_seq=rts)
# writer.write_image_by_plane(img_basename, output_dir=img_dirname, tile_size=1024)

logging.info("Finished")
