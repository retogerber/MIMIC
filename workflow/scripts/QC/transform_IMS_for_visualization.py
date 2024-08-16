import sys,os
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "..","..","workflow","scripts","utils")))
import numpy as np
import cv2
import json
import pandas as pd
import h5py
from image_utils import get_image_shape, readimage_crop
from utils import setNThreads, snakeMakeMock
import logging, traceback
import logging_utils
import pandas as pd

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["IMS_pixelsize"] = 30
    snakemake.params["microscopy_pixelsize"] = 0.22537
    snakemake.input["imzml_peaks"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/cirrhosis_TMA/data/IMS/cirrhosis_TMA_IMS_peaks.h5"
    snakemake.input["imzml_coords"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/cirrhosis_TMA/data/IMS/postIMS_to_IMS_cirrhosis_TMA-Cirrhosis_TMA_5_01262022_004-IMSML-coords.h5"
    snakemake.input['IMCmask_on_postIMS'] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/cirrhosis_TMA/data/IMC_mask/Cirrhosis_TMA_5_01262022_004_transformed_on_postIMS.ome.tiff"
    snakemake.input['IMC_location'] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/cirrhosis_TMA/data/IMC_location/cirrhosis_TMA_IMC_mask_on_postIMS_E8.geojson"
    snakemake.output["IMS_transformed"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/cirrhosis_TMA/data/IMS/cirrhosis_TMA_Cirrhosis_TMA_5_01262022_004_IMS_transformed.ome.tiff"
    snakemake.output["IMC_mask_transformed"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/cirrhosis_TMA/data/IMC_mask/Cirrhosis_TMA_5_01262022_004_transformed_on_postIMS_cropped.ome.tiff"
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# params
ims_spacing = snakemake.params["IMS_pixelsize"]
microscopy_spacing = snakemake.params["microscopy_pixelsize"]
# inputs
imsml_peaks = snakemake.input["imzml_peaks"]
imsml_coords = snakemake.input["imzml_coords"]
IMCmask_on_postIMS = snakemake.input['IMCmask_on_postIMS']
IMC_location = snakemake.input['IMC_location']
if isinstance(IMC_location, list):
    IMC_location = IMC_location[0]
# outputs
ims_out = snakemake.output["IMS_transformed"]
imc_mask_out = snakemake.output["IMC_mask_transformed"]


logging.info("core bounding box extraction")
IMC_geojson = json.load(open(IMC_location, "r"))
if isinstance(IMC_geojson,list):
    IMC_geojson=IMC_geojson[0]
boundary_points = np.array(IMC_geojson['geometry']['coordinates'])[0,:,:]
xmin=np.min(boundary_points[:,1])
xmax=np.max(boundary_points[:,1])
ymin=np.min(boundary_points[:,0])
ymax=np.max(boundary_points[:,0])

bb1 = [int(xmin-901/microscopy_spacing),int(ymin-901/microscopy_spacing),int(xmax+901/microscopy_spacing),int(ymax+901/microscopy_spacing)]

m2full_shape = get_image_shape(IMCmask_on_postIMS)
bb1[0] = bb1[0] if bb1[0]>=0 else 0
bb1[1] = bb1[1] if bb1[1]>=0 else 0
bb1[2] = bb1[2] if bb1[2]<=m2full_shape[0] else m2full_shape[0]
bb1[3] = bb1[3] if bb1[3]<=m2full_shape[1] else m2full_shape[1]
logging.info(f"bounding box mask whole image 1: {bb1}")


logging.info("Read h5 coords file")
with h5py.File(imsml_coords, "r") as f:
    # if in imsmicrolink IMS was the target
    if "xy_micro_physical" in [key for key, val in f.items()]:
        xy_micro_physical = f["xy_micro_physical"][:]

        coords_micro = xy_micro_physical
    # if the microscopy image was the target
    else:
        padded = f["xy_padded"][:]
        
        coords_micro = (padded * ims_spacing)
    
    coords_original = f["xy_original"][:]

    data = {
        "coord_0": coords_original[:,0].tolist(),
        "coord_1": coords_original[:,1].tolist(),
        "coord_micro_0": coords_micro[:,0].tolist(),
        "coord_micro_1": coords_micro[:,1].tolist()
    }

    df_1 = pd.DataFrame(data)

logging.info("Read h5 peaks file")
with h5py.File(imsml_peaks, "r") as f:
    coords = f["coord"][:]
    coords = coords[:2,:].T
    peaks = f["peaks"][:]


    data = {
        "coord_0": coords[:,0].tolist(),
        "coord_1": coords[:,1].tolist()
    }

    for i in range(peaks.shape[1]):
        data[f"peak_{i}"] = peaks[:, i].tolist()

    df_2 = pd.DataFrame(data)

logging.info("Merge dataframes")
merged_df = df_1.merge(df_2, on=["coord_0", "coord_1"], how="inner")


# transform microscopy coords to pixels
coords = np.array(merged_df[['coord_micro_0','coord_micro_1']])
coords=coords/microscopy_spacing
coords = coords.astype(int)
# get peaks as array
peaks = np.array(merged_df[[f"peak_{i}" for i in range(peaks.shape[1])]])
peaks[np.isnan(peaks)] = 0


logging.info("Create image")
image_shape = [bb1[3]-bb1[1],bb1[2]-bb1[0]]
coords = coords-bb1[:2][::-1]
# Calculate ranges
stepsize_px =  int(ims_spacing / microscopy_spacing)
stepsize_px_half = int(stepsize_px/2)
x_ranges = np.clip([coords[:,1] - stepsize_px_half, coords[:,1] + stepsize_px_half], 0, image_shape[1]-1)
y_ranges = np.clip([coords[:,0] - stepsize_px_half, coords[:,0] + stepsize_px_half], 0, image_shape[0]-1)
# create empty image
image = np.zeros((image_shape[0],image_shape[1],peaks.shape[1]))
# Fill the image 
for i, (xr, yr) in enumerate(zip(x_ranges.T, y_ranges.T)):
    image[xr[0]:(xr[1]+1), yr[0]:(yr[1]+1)] = peaks[i,:]

# from: https://forum.image.sc/t/creating-a-multi-channel-pyramid-ome-tiff-with-tiffwriter-in-python/76424
def write_tiff(im, filename, subifds=3):
    from tifffile import TiffWriter

    nchannels = im.shape[-1]

    ome_xml = '''<?xml version="1.0" encoding="UTF-8"?>
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06
    http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
    <Image ID="Image:0" Name="Image0">
    <Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint8"
    SizeX="2048" SizeY="2048" SizeC="2" SizeZ="1" SizeT="1">
    <Channel ID="Channel:0:0" SamplesPerPixel="1"><LightPath/></Channel>
    <Channel ID="Channel:0:1" SamplesPerPixel="1"><LightPath/></Channel>
    <TiffData IFD="0" PlaneCount="2"/>
    </Pixels>
    </Image>
    </OME>'''

    with TiffWriter(filename, ome=False, bigtiff=True) as tif:
        for channeli in range(nchannels):
            tif.write(
                im[:, :, channeli],
                description=ome_xml,
                subifds=subifds,
                metadata=False,  # do not write tifffile metadata
                tile=(1024, 1024),
                photometric='minisblack',
                compression='LZW',
                resolution = (microscopy_spacing, microscopy_spacing),
                resolutionunit = "MICROMETER",
            )
            for i in range(subifds):
                res = 2 ** (i + 1)
                tif.write(
                    im[::res, ::res, channeli],  # in production use resampling
                    subfiletype=1,
                    metadata=False,
                    tile=(1024, 1024),
                    photometric='minisblack',
                    compression='LZW',
                    resolution = (microscopy_spacing, microscopy_spacing),
                    resolutionunit = "MICROMETER",
                )

logging.info("Downscale image")
wn = int(image.shape[0]*microscopy_spacing)
hn = int(image.shape[1]*microscopy_spacing)
image = cv2.resize(image, (hn,wn), interpolation=cv2.INTER_NEAREST)

logging.info("Save IMS image")
write_tiff(image.astype(np.float16), ims_out)

logging.info("Read IMC mask image")
imcimg = readimage_crop(IMCmask_on_postIMS, bb1)
imcimg = cv2.resize(imcimg, (hn,wn), interpolation=cv2.INTER_NEAREST)
imcimg = imcimg.reshape(imcimg.shape[0],imcimg.shape[1],1)

logging.info("Save IMC mask image")
write_tiff(imcimg, imc_mask_out)



logging.info("Finished")
# import matplotlib.pyplot as plt
# plt.imshow(image[:,:,0])
# plt.imshow(imcimg)
# plt.show()

