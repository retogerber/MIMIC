from wsireg.reg_shapes import RegShapes
from wsireg.parameter_maps import transformations
import numpy as np
import os
from ome_types import from_tiff
from tifffile import imread
import sys
import json
from math import ceil


imc_file = snakemake.input["IMC_aggr"]
mask_file = snakemake.input["IMC_location_on_preIMC"]
#mask_file = os.path.join("/home/retger/imc_to_ims_workflow/results/cirrhosis_TMA/data/IMC_location/cirrhosis_TMA_IMC_mask_on_preIMC_A2.geojson")
ome_file = snakemake.input["preIMC"]
#ome_file = os.path.join("/home/retger/imc_to_ims_workflow/results/cirrhosis_TMA/data/preIMC/cirrhosis_TMA_preIMC.ome.tiff")

IMC_to_preIMC_transform = snakemake.output["IMC_to_preIMC_transform"]
#IMC_to_preIMC_image = snakemake.output["IMC_to_preIMC_image"]
#imcache_dir = snakemake.output["imcache_dir"]


# read imc
imc = imread(imc_file)
imc_size = imc.shape
spacing_IMC = snakemake.params["IMC_pixelsize"]
# base rigid transformation
rot_tform=transformations.BASE_RIG_TFORM.copy()
# set spacing
rot_tform["Spacing"] = [str(spacing_IMC), str(spacing_IMC)]

# set transformation parameters, i.e. 180 degree rotation + slight rotation
# read geojson
rs = RegShapes(mask_file)
arr=rs.shape_data[0]["array"]
def get_angle(arr,i,j):
    """Calculate smallest angle between consecutive poins in rectangle relative to axis"""
    xl=np.sqrt(np.square(arr[j,0]-arr[i,0]))
    yl=np.sqrt(np.square(arr[j,1]-arr[i,1]))
    xyl=np.sqrt(np.square(arr[j,0]-arr[i,0])+np.square(arr[j,1]-arr[i,1]))
    if yl>xl:
        return np.arccos(yl/xyl)
    else:
        return np.arccos(xl/xyl)
    
def get_average_angle(arr, ii, jj):
    angles=[]
    for k in range(len(ii)):
        angles.append(get_angle(arr, ii[k], jj[k]))
    return np.mean(angles)

ii=[0,1,2,3]
jj=[1,2,3,0]
shape_rotation = get_average_angle(arr,ii,jj)
total_rotation=np.pi-shape_rotation
rot_tform["TransformParameters"]=[str(total_rotation),"0.000","0.000"]


arr=rs.shape_data[0]["array"]
new_size = [ceil((np.max(arr[:,0])-np.min(arr[:,0]))*0.22537),ceil((np.max(arr[:,1])-np.min(arr[:,1]))*0.22537)]
new_size_str = [str(s) for s in new_size]
# set final image size
#rot_tform["Size"] = [str(imc_size[0]), str(imc_size[1])]
rot_tform["Size"] = new_size_str

CenterOfRotationPoint = [str((p-1)/2) for p in new_size]

#rot_tform["CenterOfRotationPoint"]=[str(int(imc_size[0]/2)),str(int(imc_size[1]/2))]
rot_tform["CenterOfRotationPoint"]=CenterOfRotationPoint

json.dump(rot_tform, open(IMC_to_preIMC_transform,"w"))



# 
osize_tform = snakemake.output["preIMC_orig_size_transform"]
#osize_tform = "/home/retger/imc_to_ims_workflow/results/cirrhosis_TMA/registrations/IMC_to_preIMC/cirrhosis_TMA_A2/.imcache_cirrhosis_TMA_A2/preIMC_orig_size_tform.json.test"

spacing = snakemake.params["microscopy_pixelsize"]
#spacing=0.22537


# find topleft corner
arr=rs.shape_data[0]["array"].copy()
# normalize per axis
arr[:,0]-=np.min(arr[:,0])
arr[:,1]-=np.min(arr[:,1])
# sum axis ~ distance to origin
arr_rowsum = np.sum(arr,axis=1)
# find minimum
ind = np.where(arr_rowsum == np.min(arr_rowsum))[0][0]
print("IMC_location")
print(rs.shape_data[0]["array"]*spacing)
print(ind)
transform_params=rs.shape_data[0]["array"][ind,:].tolist()
#transform_params=[np.min(rs.shape_data[0]["array"][:,0]),np.min(rs.shape_data[0]["array"][:,1])]
#print(transform_params)

# read ome metadata for size
ome=from_tiff(ome_file)
img_size=[ome.images[0].pixels.size_x,ome.images[0].pixels.size_y]

# base rigid transformation
tform=transformations.BASE_RIG_TFORM.copy()
# set spacing
tform["Spacing"] = [str(spacing), str(spacing)]
# set final image size
tform["Size"] = [str(img_size[0]), str(img_size[1])]
# set transformation parameters, i.e. translation in x and y axis
transform_params_scaled = [str(0.000000)]+[ str(-tp * spacing ) for tp in transform_params]
tform["TransformParameters"]=transform_params_scaled
print(tform)

json.dump(tform, open(osize_tform,"w"))


