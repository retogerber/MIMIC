import numpy as np
import pandas as pd
import h5py
from image_utils import get_image_shape, saveimage_tile
from utils import setNThreads, snakeMakeMock
import sys,os
import logging, traceback
import logging_utils
import pandas as pd

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["IMS_pixelsize"] = 30
    snakemake.params["microscopy_pixelsize"] = 0.22537
    snakemake.input["imzml_peaks"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/IMS/IMS_test_combined_peaks.h5"
    snakemake.input["imzml_coords"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/IMS/postIMS_to_IMS_test_combined-Cirrhosis-TMA-5_New_Detector_002-IMSML-coords.h5"
    snakemake.input['postIMS'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/postIMS/test_combined_postIMS.ome.tiff"
    snakemake.output["IMS_transformed"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/IMS/test_combined_Cirrhosis-TMA-5_New_Detector_002_IMS_transformed.ome.tiff"
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
postIMS = snakemake.input['postIMS']
# outputs
ims_out = snakemake.output["IMS_transformed"]

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
image_shape = get_image_shape(postIMS)[:2]
# Calculate ranges
stepsize_px =  int(ims_spacing / microscopy_spacing)
stepsize_px_half = int(stepsize_px/2)
x_ranges = np.clip([coords[:,0] - stepsize_px_half, coords[:,0] + stepsize_px_half], 0, image_shape[0]-1)
y_ranges = np.clip([coords[:,1] - stepsize_px_half, coords[:,1] + stepsize_px_half], 0, image_shape[1]-1)
# create empty image
image = np.zeros((image_shape[0],image_shape[1],peaks.shape[1]))
# Fill the image 
for i, (xr, yr) in enumerate(zip(x_ranges.T, y_ranges.T)):
    image[xr[0]:(xr[1]+1), yr[0]:(yr[1]+1)] = peaks[i,:]

logging.info("Save image")
# from: https://forum.image.sc/t/creating-a-multi-channel-pyramid-ome-tiff-with-tiffwriter-in-python/76424
from tifffile import TiffWriter

image=image.astype(np.uint8)
nchannels = image.shape[-1]
subifds = 3

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

with TiffWriter(ims_out, ome=False, bigtiff=True) as tif:
    for channeli in range(nchannels):
        tif.write(
            image[:, :, channeli],
            description=ome_xml,
            subifds=subifds,
            metadata=False,  # do not write tifffile metadata
            tile=(1024, 1024),
            photometric='minisblack',
            compression='jpeg',
            # resolution = ...,
            # resolutionunit = ...,
        )
        for i in range(subifds):
            res = 2 ** (i + 1)
            tif.write(
                image[::res, ::res, channeli],  # in production use resampling
                subfiletype=1,
                metadata=False,
                tile=(1024, 1024),
                photometric='minisblack',
                compression='jpeg',
                # resolution = ...,
                # resolutionunit = ...,
            )


logging.info("Finished")
# import matplotlib.pyplot as plt
# plt.imshow(image[:,:,0])
# plt.show()

