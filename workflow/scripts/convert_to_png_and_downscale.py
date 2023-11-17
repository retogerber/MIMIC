import cv2
import json
import numpy as np
from image_registration_IMS_to_preIMS_utils import readimage_crop,get_image_shape 
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

cv2.setNumThreads(snakemake.threads)

input_image = snakemake.input["input_image"]
try:
    geojson = snakemake.input["geojson"]
except:
    geojson = None
downscale_factor = snakemake.params["downscale_factor"]

logging.info(f"input_image: {input_image}")
logging.info(f"geojson: {geojson}")
logging.info(f"downscale_factor: {downscale_factor}")


def get_bbox(geojson):
    IMC_geojson = json.load(open(geojson, "r"))
    if isinstance(IMC_geojson,list):
        IMC_geojson=IMC_geojson[0]
    boundary_points = np.array(IMC_geojson['geometry']['coordinates'])[0,:,:]
    xmin=int(np.min(boundary_points[:,1]))
    xmax=int(np.max(boundary_points[:,1]))
    ymin=int(np.min(boundary_points[:,0]))
    ymax=int(np.max(boundary_points[:,0]))
    bbox = np.array([xmin,ymin,xmax,ymax])
    return bbox

logging.info(f"Read bbox")
if not geojson is None and os.path.exists(geojson):
    bbox = get_bbox(geojson)
else:
    imsh = get_image_shape(input_image)
    bbox = np.array([0,0,imsh[1],imsh[0]])
logging.info(f"bbox: {bbox}")

logging.info(f"Read image")
img = readimage_crop(input_image,bbox)
logging.info(f"Image shape: {img.shape}")

logging.info("Downscale image")
if downscale_factor != 1:
    wn = int(img.shape[0]/downscale_factor)
    hn = int(img.shape[1]/downscale_factor)
    img = cv2.resize(img, (hn,wn), interpolation=cv2.INTER_NEAREST)

logging.info(f"Image shape: {img.shape}")

logging.info("Write image")
if len(img.shape)>2:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGRA)

cv2.imwrite(snakemake.output["output_image"],img)


logging.info("Finished")