import sys,os
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "..","..","workflow","scripts","utils")))
import numpy as np
import json
from shapely.geometry import shape
import re
import SimpleITK as sitk
import pandas as pd
import h5py
from image_utils import get_image_shape, saveimage_tile
from utils import setNThreads, snakeMakeMock
import logging, traceback
import logging_utils
import pandas as pd

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["IMS_pixelsize"] = 30
    snakemake.params["microscopy_pixelsize"] = 0.22537
    snakemake.params["output_pixelsize"] = 1
    snakemake.params["IMS_rotation_angle"] = [180,180]
    snakemake.params["sample_names"] = ["Cirrhosis-TMA-5_New_Detector_001","Cirrhosis-TMA-5_New_Detector_002"]
    snakemake.params["use_bbox"] = False
    snakemake.input["imzml_peaks"] = "results/test_split_pre/data/IMS/IMS_test_split_pre_peaks.h5","results/test_split_pre/data/IMS/IMS_test_split_pre_peaks.h5"
    snakemake.input["imzml_coords"] = ["results/test_split_pre/data/IMS/postIMS_to_IMS_test_split_pre-IMSML-coords.h5","results/test_split_pre/data/IMS/postIMS_to_IMS_test_split_pre-Cirrhosis-TMA-5_New_Detector_002-IMSML-coords.h5"] 
    # snakemake.input["imzml_coords"] = ["results/test_split_pre/data/IMS/postIMS_to_IMS_test_split_pre-Cirrhosis-TMA-5_New_Detector_001-IMSML-coords.h5","results/test_split_pre/data/IMS/postIMS_to_IMS_test_split_pre-Cirrhosis-TMA-5_New_Detector_002-IMSML-coords.h5"] 
    snakemake.input["ims_meta"] = "results/test_split_pre/data/IMS/postIMS_to_IMS_test_split_pre-IMSML-meta.json","results/test_split_pre/data/IMS/postIMS_to_IMS_test_split_pre-Cirrhosis-TMA-5_New_Detector_002-IMSML-meta.json"
    # snakemake.input["ims_meta"] = "results/test_split_pre/data/IMS/postIMS_to_IMS_test_split_pre-Cirrhosis-TMA-5_New_Detector_001-IMSML-meta.json","results/test_split_pre/data/IMS/postIMS_to_IMS_test_split_pre-Cirrhosis-TMA-5_New_Detector_002-IMSML-meta.json"
    snakemake.input["postIMS"] = "results/test_split_pre/data/postIMS/test_split_pre_postIMS.ome.tiff"
    snakemake.input["TMA_location_target"] = [f"/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/TMA_location/test_split_pre_TMA_location_on_postIMS_{core}.geojson" for core in ["A1","B1"]]
    # snakemake.input["TMA_location_target"] = [f"results/test_split_pre/data/registration_metric/{sample}_step1_metadata.json" for sample in ["Cirrhosis-TMA-5_New_Detector_001","Cirrhosis-TMA-5_New_Detector_002"]]
    snakemake.output["IMS_transformed"] = "results/test_split_pre/data/IMS/test_split_pre_IMS_on_postIMS.ome.tiff"
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# params
ims_spacing = snakemake.params["IMS_pixelsize"]
microscopy_spacing = snakemake.params["microscopy_pixelsize"]
output_spacing = snakemake.params["output_pixelsize"]
ims_rotation_angle = snakemake.params["IMS_rotation_angle"]
if isinstance(ims_rotation_angle, str) or isinstance(ims_rotation_angle, int):
    ims_rotation_angle = list(ims_rotation_angle)
for angle in ims_rotation_angle:
    assert angle in [0,90,180,270], "IMS rotation angle must be 0, 90, 180 or 270"
sample_names = snakemake.params['sample_names']
if isinstance(sample_names, str):
    sample_names = [sample_names]
use_bbox = snakemake.params["use_bbox"]

# inputs
imsml_peaks = snakemake.input["imzml_peaks"]
if isinstance(imsml_peaks, str):
    imsml_peaks = [imsml_peaks]
imsml_coords = snakemake.input["imzml_coords"]
if isinstance(imsml_coords, str):
    imsml_coords = [imsml_coords]
ims_meta_file = snakemake.input["ims_meta"]
if isinstance(ims_meta_file, str):
    ims_meta_file = [ims_meta_file]
TMA_target_geojson_file=snakemake.input['TMA_location_target']
if isinstance(TMA_target_geojson_file, str):
    TMA_target_geojson_file = [TMA_target_geojson_file]
postIMS_file = snakemake.input["postIMS"]

assert len(imsml_coords) == len(imsml_peaks), "number of coords and peaks files must be the same"
assert len(imsml_coords) == len(ims_meta_file), "number of coords and meta files must be the same"
assert len(imsml_coords) == len(ims_rotation_angle), "number of coords and rotation angles must be the same"
# outputs
ims_out_dir = snakemake.output["IMS_transformed"]

if not os.path.exists(ims_out_dir):
    os.makedirs(ims_out_dir)

# def check_is_manual_registration(imsml_coords, sample_name):
#     project_name = re.search(r"results/(.*)/data", imsml_coords).group(1)
#     regex = re.compile(fr"^postIMS_to_IMS_{project_name}-{sample_name}-IMSML-coords.h5$")
#     if regex.search(os.path.basename(imsml_coords)):
#         return False
#     else:
#         return True

# is_manual_registration = [check_is_manual_registration(imsml_coords[i], sample_names[i]) for i in range(len(imsml_coords))]
def get_peaks(imsml_peaks, mz=None):
    logging.info(f"\tRead h5 peaks file: {imsml_peaks}")
    with h5py.File(imsml_peaks, "r") as f:
        coords = f["coord"][:]
        coords = coords[:2,:].T
        data = {
            "coord_0": coords[:,0].tolist(),
            "coord_1": coords[:,1].tolist()
        }
        mzs = f["mzs"][:][0]
        if not mz is None:
            dists = np.array([abs(float(mz) - float(m)) for m in mzs])
            mz_ind = np.argmin(dists)
            data[f"peak_{mz_ind}_mz{mz}"] = f["peaks"][:,mz_ind].tolist()
        else:
            peaks = f["peaks"][:]
            for i in range(peaks.shape[1]):
                data[f"peak_{i}_mz{mzs[i]}"] = peaks[:, i].tolist()

    return pd.DataFrame(data)

def get_directions(imsml_coords):
    logging.info(f"\tReading h5 coords file: {imsml_coords}")
    with h5py.File(imsml_coords, "r") as f:
        # if in imsmicrolink IMS was the target
        if "xy_micro_physical" in [key for key, val in f.items()]:
            direction = "IMS_to_postIMS"
        # if the microscopy image was the target
        else:
            direction = "postIMS_to_IMS"
    return direction

def get_coords(imsml_coords):
    logging.info(f"\tReading h5 coords file: {imsml_coords}")
    with h5py.File(imsml_coords, "r") as f:
        df_1 = pd.DataFrame({
            "coord_0_padded": f["xy_padded"][:,0].tolist(),
            "coord_1_padded": f["xy_padded"][:,1].tolist(),
            "coord_0": f["xy_original"][:,0].tolist(),
            "coord_1": f["xy_original"][:,1].tolist()
        })
    return df_1

def merge_dfs(df_1, df_2, mz=None):
    logging.info(f"\tMerge dataframes")
    if not mz is None:
        peak_col = [col for col in df_2.columns if "peak" in col]
        no_peak_col = [col for col in df_2.columns if not "peak" in col]
        regex = re.compile(r"^peak_[0-9]*_mz")
        peak_names = list()
        for pc in peak_col:
            peak_names.append(regex.sub("", pc))
        dists = np.array([abs(float(mz) - float(m)) for m in peak_names])
        mz_ind = np.argmin(dists)
        merged_df = df_1.merge(df_2[no_peak_col+[peak_col[mz_ind]]], on=["coord_0", "coord_1"], how="inner")
    else:
        merged_df = df_1.merge(df_2, on=["coord_0", "coord_1"], how="inner")
    return merged_df

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

# get info of IMC location
logging.info("Create bounding box")
bb_target_ls = list()
for single_TMA_geojson_file in TMA_target_geojson_file:
    TMA_geojson = json.load(open(single_TMA_geojson_file, "r"))
    if isinstance(TMA_geojson,list):
        TMA_geojson=TMA_geojson[0]
    if 'postIMS_bbox' in TMA_geojson.keys():
        bb1 = np.array(TMA_geojson['postIMS_bbox'])/(output_spacing/microscopy_spacing)
        bb1 = bb1.astype(int)
    else:
        # bounding box
        bb1 = shape(TMA_geojson['geometry']).bounds
        # reorder axis
        bb1 = np.array([bb1[1],bb1[0],bb1[3],bb1[2]])/(output_spacing/microscopy_spacing)
        bb1 = bb1.astype(int)
    bb_target_ls.append(bb1)

logging.info(f"Get coordinates")
df1_dic = {f: get_coords(f) for f in np.unique(imsml_coords)}
directions = {f: get_directions(f) for f in np.unique(imsml_coords)}
logging.info(f"Get peaks")
df2_dic = {f: get_peaks(f) for f in np.unique(imsml_peaks)}
logging.info(f"Done reading data")

merged_df = merge_dfs(df1_dic[imsml_coords[0]], df2_dic[imsml_peaks[0]])
# number of peaks + index
peak_col = [col for col in merged_df.columns if "peak" in col] + ["indices","qc"]
regex = re.compile(r"^peak_[0-9]*_mz")
peak_names = list()
for pc in peak_col:
    peak_names.append(regex.sub("", pc))

ims_out = [f"{ims_out_dir}/mz_{peak}.ome.tiff" for peak in peak_names]
logging.info(f"Output files, base dir:{os.path.basename(ims_out_dir)}")
logging.info(f"Output files, base names:{[os.path.basename(ims_out[i]) for i in range(len(ims_out))]}")

image_shape = get_image_shape(postIMS_file)
output_image_shape = [len(peak_col), int(image_shape[0]/(output_spacing/microscopy_spacing)), int(image_shape[1]/(output_spacing/microscopy_spacing))]
out_image = np.zeros(output_image_shape, dtype=np.float32)    
for ch in range(output_image_shape[0]):
    logging.info(f"Channel: {peak_names[ch]}, file: {ims_out[ch]}")
    out_image = np.zeros([1,output_image_shape[1],output_image_shape[2]], dtype=np.float32)    
    logging.info(f"Ouput image shape single channel: {out_image.shape}")
    for sample_id in range(len(imsml_coords)):
        logging.info(f"Sample: {sample_names[sample_id]}")
        # logging.info(f"\tis manual registration: {is_manual_registration[sample_id]}")
        logging.info(f"\tIMS rotation angle: {ims_rotation_angle[sample_id]}")
        reg_direction = directions[imsml_coords[sample_id]]
        merged_df = merge_dfs(df1_dic[imsml_coords[sample_id]], df2_dic[imsml_peaks[sample_id]], mz=peak_names[ch])
        logging.info(f"\tMetadata file: {ims_meta_file[sample_id]}")
        logging.info(f"\tdirection of registration: {reg_direction}")

        coords = np.array(merged_df[['coord_0_padded','coord_1_padded']])
        ims_x_min = np.min(coords[:,1])
        ims_x_max = np.max(coords[:,1])
        ims_y_min = np.min(coords[:,0])
        ims_y_max = np.max(coords[:,0])


        tmp_peak_col = [col for col in merged_df.columns if "peak" in col]
        assert len(tmp_peak_col) == 1
        tmp_peak_col = tmp_peak_col[0]

        if peak_names[ch] == "indices":
            peaks = np.array(merged_df.index)
        elif peak_names[ch] == "qc":
            peaks = np.array([0,0])
        else:
            peaks = np.array(merged_df[tmp_peak_col])
        assert peaks.ndim == 1
        metadata = json.load(open(ims_meta_file[sample_id], "r"))


        logging.info(f"\tLoad Affine transformation matrix")
        aff_mat = np.array(metadata["Affine transformation matrix (yx,microns)"])
        IMS_to_postIMS_transform = sitk.AffineTransform(2)
        IMS_to_postIMS_transform.SetMatrix(aff_mat[:2,:2].flatten())
        IMS_to_postIMS_transform.SetTranslation(aff_mat[:2,2])
        if reg_direction == "IMS_to_postIMS":
            logging.info(f"\tInvert affine transform")
            IMS_to_postIMS_transform = IMS_to_postIMS_transform.GetInverse()
        logging.info(f"\taffine transform params: {IMS_to_postIMS_transform.GetParameters()}")
        transform = IMS_to_postIMS_transform

        logging.info(f"\tGet bounding box")
        corner_pts = np.array([[ims_x_min, ims_y_min], [ims_x_max, ims_y_min], [ims_x_max, ims_y_max], [ims_x_min, ims_y_max]])
        corner_pts_scaled = np.array([transform.TransformPoint([float(x*ims_spacing), float(y*ims_spacing)]) for x,y in corner_pts])/(output_spacing/microscopy_spacing)
        bbox = np.array([np.min(corner_pts_scaled, axis=0), np.max(corner_pts_scaled, axis=0)]).flatten().astype(int)
        bbox[0] = (np.floor(bbox[0])-ims_spacing)
        offset_x = 0 if bbox[0]>=0 else -bbox[0]
        bbox[0] = 0 if bbox[0]<0 else bbox[0]
        bbox[1] = (np.floor(bbox[1])-ims_spacing)
        offset_y = 0 if bbox[1]>=0 else -bbox[1]
        bbox[1] = 0 if bbox[1]<0 else bbox[1]
        bbox[2] = (np.ceil(bbox[2])+ims_spacing)
        bbox[2] = image_shape[0] if bbox[2]>image_shape[0] else bbox[2]
        bbox[3] = (np.ceil(bbox[3])+ims_spacing)
        bbox[3] = image_shape[1] if bbox[3]>image_shape[1] else bbox[3]
        bbox = bbox.astype(int)
        logging.info(f"\tbbox on postIMS: {bbox}")

        logging.info(f"\tCreate IMS image")
        # create empty image
        # image = np.zeros((output_image_shape[0],ims_x_max+1, ims_y_max+1))
        image = np.zeros((1,ims_x_max+1, ims_y_max+1))
        if peak_names[ch] == "qc":
        # checkerboard pattern for quality control
            for i, (xr, yr) in enumerate(zip(coords[:,1].T, coords[:,0].T)):
                image[0,xr, yr] = 1
            image[0,::2, :][image[-1,::2,:]!=0] = 2
            image[0,:,::2][image[-1,:,::2]!=0] = 2
            image[0,::2,::2][image[-1,::2,::2]!=0] = 1
        else:
            # Fill the image 
            for i, (xr, yr) in enumerate(zip(coords[:,1].T, coords[:,0].T)):
                image[0,xr, yr] = peaks[i]


        logging.info(f"\tImage shape: {image.shape}")

        logging.info(f"\tCreate composite transform")
        composite = sitk.CompositeTransform(2)

        composite.AddTransform(transform)
        if use_bbox:
            composite.AddTransform(sitk.TranslationTransform(2, [float((bbox[0])*output_spacing), float((bbox[1])*output_spacing)]))
        else:
            composite.AddTransform(sitk.TranslationTransform(2, [float(bb_target_ls[sample_id][0])*output_spacing, float(bb_target_ls[sample_id][1])*output_spacing]))

        logging.info(f"\tnumber of transforms: {composite.GetNumberOfTransforms()}")
        logging.info(f"\ttransforms:")
        trls = [composite.GetNthTransform(j) for j in range(composite.GetNumberOfTransforms())]
        for j,trl in enumerate(trls):
            if trl.GetName() == "CompositeTransform":
                trls2 = [trl.GetNthTransform(j) for j in range(trl.GetNumberOfTransforms())]
                for jj,trl2 in enumerate(trls2):
                    logging.info(f"\t\t{j}{jj}: linear, {trl2.GetParameters()}")
            else:
                logging.info(f"\t\t{j}: linear, {trl.GetParameters()}")


        tmpimg = sitk.Image([1,1], sitk.sitkFloat32)
        tmpimg.SetSpacing([output_spacing,output_spacing])
        resampler = get_resampler(composite, tmpimg)
        if use_bbox:
            newsize = np.array([bbox[2]-bbox[0],bbox[3]-bbox[1]], dtype='int').tolist()
        else:
            newsize = np.array([bb_target_ls[sample_id][2]-bb_target_ls[sample_id][0],bb_target_ls[sample_id][3]-bb_target_ls[sample_id][1]], dtype='int').tolist()
        resampler.SetSize(newsize)
        resampler.SetTransform(composite)

        logging.info(f"\tResample image")
        moving = sitk.GetImageFromArray(np.swapaxes(image[0,:,:],0,1))
        moving.SetSpacing([ims_spacing, ims_spacing])

        source_image_trans = resample_image(resampler, moving)
        np.max(source_image_trans)

        if use_bbox:
            out_image[0,bbox[0]:bbox[2], bbox[1]:bbox[3]] = np.swapaxes(source_image_trans,0,1)
        else:
            out_image[0,bb_target_ls[sample_id][0]:bb_target_ls[sample_id][2], bb_target_ls[sample_id][1]:bb_target_ls[sample_id][3]] = np.swapaxes(source_image_trans,0,1)


    logging.info(f"Save image")
    saveimage_tile(out_image, filename = ims_out[ch], resolution=output_spacing, dtype=np.float32, is_rgb=False, channel_names = peak_names[ch], compression="default")

# import tifffile
# postIMS = tifffile.imread(postIMS_file)
# # downsample postIMS
# stepsize_px = int(ims_spacing / microscopy_spacing)
# postIMS = postIMS[::stepsize_px, ::stepsize_px]
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1,3, figsize=(15,10))
# # ax[0].imshow(out_image[0,::stepsize_px, ::stepsize_px])
# ax[0].imshow(np.swapaxes(source_image_trans[::stepsize_px, ::stepsize_px],0,1))
# ax[1].imshow(postIMS)
# ax[2].imshow(postIMS[:,:,0]/255 + out_image[0,::stepsize_px, ::stepsize_px])
# plt.savefig("test.png")


logging.info("Finished")