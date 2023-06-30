from wsireg.reg_shapes import RegShapes
from wsireg.reg_transforms.reg_transform import RegTransform
from wsireg.reg_transforms.reg_transform_seq import RegTransformSeq
import sys
import os

#mask_file = sys.argv[1]
mask_file = snakemake.input["IMC_location_on_postIMC"]
#mask_file = "/home/retger/IMC/data/image_coregistration/IMC_geojson/cirrhosis_TMA/cirrhosis_TMA_E1_unreg.geojson"
#transform_file = sys.argv[2]
transform_file = snakemake.input["postIMC_to_postIMS_transform"]
#transform_file = "/home/retger/IMC/analysis/image_coregistration/cirrhosis_TMA/post_IMC_to_post_IMS/cirrhosis_TMA-postIMC_to_postIMS_transformations.json"
#out_mask = sys.argv[3]
out_mask = snakemake.output["IMC_location_on_postIMS"]
#out_mask = os.path.splitext(mask_file)[0]+"_transform"+os.path.splitext(mask_file)[1]

#microscopy_pixelsize = sys.argv[4]
microscopy_pixelsize = snakemake.params["microscopy_pixelsize"]

# read in mask
rs = RegShapes(mask_file)

# read in transform
rts = RegTransformSeq(transform_file)
#rts.set_output_spacing((1.0, 1.0)) # only do this for .ome.tiff converted data, comment for .ndpi file
rts.set_output_spacing((float(microscopy_pixelsize), float(microscopy_pixelsize))) # only do this for .ome.tiff converted data, comment for .ndpi file

# do transformation
rs.transform_shapes(rts)

rs.save_shape_data(out_mask)
