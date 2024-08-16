import sys,os
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "..","..","workflow","scripts","utils")))
import numpy as np
import pandas as pd
import h5py
from utils import setNThreads, snakeMakeMock
import logging, traceback
import logging_utils
import pandas as pd

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["IMS_pixelsize"] = 30
    snakemake.input["imzml_peaks"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/IMS/IMS_test_combined_peaks.h5"
    snakemake.output["IMS_image"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/IMS/IMS_test_combined.ome.tiff"
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# params
IMS_pixelsize = snakemake.params["IMS_pixelsize"]
# inputs
imsml_peaks = snakemake.input["imzml_peaks"]
# outputs
ims_out = snakemake.output["IMS_image"]

logging.info("Read h5 peaks file")
with h5py.File(imsml_peaks, "r") as f:
    coords = f["coord"][:]
    coords = coords[:2,:].T
    mzs = f["mzs"][:].flatten()
    peaks = f["peaks"][:]


    data = {
        "coord_0": coords[:,0].tolist(),
        "coord_1": coords[:,1].tolist()
    }

    for i in range(peaks.shape[1]):
        data[f"peak_{i}"] = peaks[:, i].tolist()

    merged_df = pd.DataFrame(data)


# transform microscopy coords to pixels
coords = np.array(merged_df[['coord_0','coord_1']])
coords = coords.astype(int)
coords = coords - np.min(coords, axis=0)

# get peaks as array
peaks = np.array(merged_df[[f"peak_{i}" for i in range(peaks.shape[1])]])
peaks[np.isnan(peaks)] = 0


logging.info("Create image")
image_shape = np.max(coords, axis=0)[::-1] + 1
# create empty image
image = np.zeros((image_shape[0],image_shape[1],peaks.shape[1]))
# Fill the image 
for i, (xr, yr) in enumerate(zip(coords[:,1], coords[:,0])):
    image[xr, yr] = peaks[i,:]

# from: https://forum.image.sc/t/creating-a-multi-channel-pyramid-ome-tiff-with-tiffwriter-in-python/76424
def write_tiff(im, filename, channelnames, subifds=3):
    from tifffile import TiffWriter

    nchannels = im.shape[-1]

    s1 = '''<?xml version="1.0" encoding="UTF-8"?>
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06
    http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
    <Image ID="Image:0" Name="Image0">'''
    s2 = f'<Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint8" SizeX="{im.shape[1]}" SizeY="{im.shape[0]}" SizeC="{nchannels}" SizeZ="1" SizeT="1">'
    s3 = ''.join([f'<Channel ID="Channel:0:{i}" Name="{channelnames[i]}" SamplesPerPixel="1"><LightPath/></Channel>' for i in range(nchannels)])
    s4 = f'''<TiffData IFD="0" PlaneCount="{nchannels}"/>
    </Pixels>
    </Image>
    </OME>'''

    ome_xml = s1 + s2 + s3 + s4

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
                resolution = (IMS_pixelsize, IMS_pixelsize),
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
                    resolution = (IMS_pixelsize, IMS_pixelsize),
                    resolutionunit = "MICROMETER",
                )

logging.info("Save IMS image")
write_tiff(image.astype(np.float16), ims_out, mzs)

logging.info("Finished")
