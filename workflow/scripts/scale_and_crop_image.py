from wsireg.writers.ome_tiff_writer import OmeTiffWriter
from wsireg.reg_transforms.reg_transform_seq import RegTransform, RegTransformSeq
from wsireg.parameter_maps.transformations import BASE_AFF_TFORM
from wsireg.reg_images.loader import reg_image_loader
from image_utils import get_image_shape 
from utils import setNThreads
import json
import numpy as np
import sys, os, getopt
from ome_types import from_tiff

def main(argv):
    inputfile = ''
    outputfile = ''
    geojsonfile = ''
    output_spacing = ''
    nthreads = 1
    try:
        opts, args = getopt.getopt(argv,"hi:o:gsn")
    except getopt.GetoptError:
        print ('scale_and_crop_image.py -i <inputfile> -o <outputfile> -g <geojsonfile> -s <output_spacing> -n <nthreads>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('scale_and_crop_image.py -i <inputfile> -o <outputfile> -g <geojsonfile> -s <output_spacing> -n <nthreads>')
            sys.exit()
        elif opt in ("-i"):
            inputfile = arg
        elif opt in ("-o"):
            outputfile = arg
        elif opt in ("-g"):
            geojsonfile = arg
        elif opt in ("-s"):
            output_spacing = arg
        elif opt in ("-n"):
            nthreads = arg
    if inputfile == '' or outputfile == '':
        print ('scale_and_crop_image.py -i <inputfile> -o <outputfile> -g <geojsonfile> -s <output_spacing> -n <nthreads>')
        sys.exit(2)

    setNThreads(nthreads)
    ome = from_tiff(inputfile) 
    input_spacing = ome.images[0].pixels.physical_size_x
    imgshape = get_image_shape(inputfile)
    print(f"imgshape: {imgshape}")

    if output_spacing == '':
        output_spacing = input_spacing

    img_basename = os.path.basename(outputfile).split(".")[0]
    img_dirname = os.path.dirname(outputfile)

    if geojsonfile != '':
        postIMS_geojson = json.load(open(geojsonfile, "r"))
        if isinstance(postIMS_geojson,list):
            postIMS_geojson=postIMS_geojson[0]
        boundary_points = np.array(postIMS_geojson['geometry']['coordinates'])[0,:,:].astype(float)
        print("Assuming the scale of the geojson is in pixels, downscaling to microns.")
        boundary_points*=input_spacing
        xmin=np.min(boundary_points[:,1])
        xmax=np.max(boundary_points[:,1])
        ymin=np.min(boundary_points[:,0])
        ymax=np.max(boundary_points[:,0])

        # if in micron scale:
        output_size = (int((ymax-ymin)/output_spacing), int((xmax-xmin)/output_spacing))
    else:
        output_size = (int(imgshape[0]/output_spacing), int(imgshape[1]/output_spacing))
        ymin=0
        xmin=0

    # setup transformation sequence
    empty_transform = BASE_AFF_TFORM.copy()
    empty_transform['Spacing'] = (str(input_spacing),str(input_spacing))
    empty_transform['Size'] = output_size
    empty_transform['TransformParameters'] = (1,0,0,1,ymin,xmin)
    rt = RegTransform(empty_transform)
    rt.output_size = output_size
    rt.output_spacing = (output_spacing, output_spacing)
    rts = RegTransformSeq(rt,[0])

    # transform and save image
    ri = reg_image_loader(inputfile, input_spacing)
    writer = OmeTiffWriter(ri, reg_transform_seq=rts)
    writer.write_image_by_plane(img_basename, output_dir=img_dirname, tile_size=1024)

    # ome = from_tiff(inputfile) 
    # omeout = from_tiff(outputfile) 
    # ome.images[0]
    # omeout.images[0]

    # get_image_shape(inputfile)
    # get_image_shape(outputfile)


if __name__ == "__main__":
   main(sys.argv[1:])
