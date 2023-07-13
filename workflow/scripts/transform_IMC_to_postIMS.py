from wsireg.writers.ome_tiff_writer import OmeTiffWriter
from wsireg.reg_transforms.reg_transform import RegTransform
from wsireg.reg_transforms.reg_transform_seq import RegTransformSeq
from wsireg.reg_images.loader import reg_image_loader
from tifffile import imread
import json
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

# read transformations
transform_file_IMC_to_preIMC=snakemake.input["IMC_to_preIMC_transform"]
orig_size_tform_IMC_to_preIMC=snakemake.input["preIMC_orig_size_transform"]
#rts_pickle_in = snakemake.input["preIMC_to_postIMS_transform_inverted"]
transform_file_preIMC_to_postIMS=snakemake.input["preIMC_to_postIMS_transform"]

# cell mask file
img_file = snakemake.input["IMC"]

img_out = snakemake.output["IMC_transformed"]
img_basename = os.path.basename(img_out).split(".")[0]
img_dirname = os.path.dirname(img_out)

logging.info("Read image")
# read in IMC image
img=imread(img_file)

logging.info("Read Transformation 1")
if os.path.getsize(transform_file_IMC_to_preIMC)>0:
# transform sequence IMC to preIMC
    try:
        rts = RegTransformSeq(transform_file_IMC_to_preIMC)
        read_rts_error=False
    except:
        read_rts_error=True
    try:
        tmptform = json.load(open(transform_file_IMC_to_preIMC, "r"))
        print("tmptform")
        print(tmptform)
        tmprt = RegTransform(tmptform)
        rts=RegTransformSeq([tmprt], transform_seq_idx=[0])
        read_rts_error=False
    except:
        read_rts_error=True
    if read_rts_error:
        exit("Could not read transform data transform_file_IMC_to_preIMC: " + transform_file_IMC_to_preIMC)

else:
    print("Empty File!")
    rts = RegTransformSeq()
    
#rts.set_output_spacing((1.0,1.0))
#ri = reg_image_loader(img, 1.0)#,preprocessing=ipp)
#writer = OmeTiffWriter(ri, reg_transform_seq=rts)
#writer.write_image_by_plane(img_basename+"_1", output_dir=img_dirname, tile_size=1024)
    

logging.info("Read Transformation 2")
# read transform sequence preIMC
osize_tform = json.load(open(orig_size_tform_IMC_to_preIMC, "r"))
osize_tform_rt = RegTransform(osize_tform)
osize_rts = RegTransformSeq([osize_tform_rt], transform_seq_idx=[0])
rts.append(osize_rts)

rts.set_output_spacing((1.0,1.0))
#ri = reg_image_loader(img, 1.0)#,preprocessing=ipp)
#writer = OmeTiffWriter(ri, reg_transform_seq=rts)
#writer.write_image_by_plane(img_basename+"_2", output_dir=img_dirname, tile_size=1024)
  
logging.info("Read Transformation 3")
# read in transformation sequence from preIMC to postIMS
rtsn=RegTransformSeq(transform_file_preIMC_to_postIMS)
rtsn.set_output_spacing((1.0,1.0))


## read in transformation sequence from preIMC to postIMS
#with open(rts_pickle_in, "rb") as f:
#    rtls=pickle.load(f)
#
#rtsn = RegTransformSeq(rtls, transform_seq_idx=list(range(len(rtls))))
#rtsn.set_output_spacing((1.0,1.0))

# combine transformations
rts.append(rtsn)
rts.set_output_spacing((1.0,1.0))

logging.info("Transform and save image")
ri = reg_image_loader(img, 1.0)#,preprocessing=ipp)
writer = OmeTiffWriter(ri, reg_transform_seq=rts)
writer.write_image_by_plane(img_basename, output_dir=img_dirname, tile_size=1024)

logging.info("Finished")