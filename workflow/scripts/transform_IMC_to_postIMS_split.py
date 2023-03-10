from wsireg.writers.ome_tiff_writer import OmeTiffWriter
from wsireg.reg_transforms.reg_transform import RegTransform
from wsireg.reg_transforms.reg_transform_seq import RegTransformSeq
from wsireg.reg_images.loader import reg_image_loader
import json

# read transformations
transform_file_IMC_to_preIMC="/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/registrations/IMC_to_preIMC/NASH_HCC_TMA_A1/NASH_HCC_TMA_A1-precise_IMC_to_preIMC_transformations.json"
orig_size_tform_IMC_to_preIMC="/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/registrations/IMC_to_preIMC/NASH_HCC_TMA_A1/NASH_HCC_TMA_A1_precise_preIMC_orig_size_tform.json"


transform_file_preIMC_to_preIMS="/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/registrations/preIMC_to_preIMS/1/NASH_HCC_TMA_1-preIMC_to_preIMS_transformations.json"
orig_size_tform_preIMC_to_preIMS="/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/registrations/preIMC_to_preIMS/1/.imcache_NASH_HCC_TMA_1/preIMS_orig_size_tform.json"

transform_file_preIMS_to_postIMS="/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/registrations/postIMC_to_postIMS/NASH_HCC_TMA-preIMS_to_postIMS_transformations.json"

img = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_workflow/results/masks/NASH_HCC_TMA-2_031.tiff"
img_dirname="/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/IMC_mask"


# IMC to preIMC
tr_IMC_to_preIMC = json.load(open(transform_file_IMC_to_preIMC, "r"))
tmprt = RegTransform(tr_IMC_to_preIMC)
rts=RegTransformSeq([tmprt], transform_seq_idx=[0])

# to preIMC size
os_IMC_to_preIMC = json.load(open(orig_size_tform_IMC_to_preIMC, "r"))
osize_tform_rt = RegTransform(os_IMC_to_preIMC)
osize_rts = RegTransformSeq([osize_tform_rt], transform_seq_idx=[0])
rts.append(osize_rts)

# preIMC to preIMS
#rts2 = RegTransformSeq(transform_file_preIMC_to_preIMS)
#rts.append(rts2)

# to preIMS size
#os_preIMC_to_preIMS = json.load(open(orig_size_tform_preIMC_to_preIMS, "r"))
# set correct spacing (because downsampling=2 was used for registration)
#os_preIMC_to_preIMS['Spacing'] = [ str(float(e)/2) for e in os_preIMC_to_preIMS['Spacing']]
#osize_tform_rt = RegTransform(os_preIMC_to_preIMS)
#osize_rts = RegTransformSeq([osize_tform_rt], transform_seq_idx=[0])
#rts.append(osize_rts)

# preIMS to postIMS
#tr_preIMS_to_postIMS = RegTransformSeq(transform_file_preIMS_to_postIMS)
#tr_preIMS_to_postIMS.set_output_spacing((1.0,1.0))
#rts.append(tr_preIMS_to_postIMS)



outt_file="/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/registrations/preIMC_to_preIMS/1/NASH_HCC_TMA_1-preIMC_to_postIMS_transformations.json"
rts2 = RegTransformSeq(outt_file)
rts.append(rts2)



# set spacing to 1um
rts.set_output_spacing((1.0,1.0))
ri = reg_image_loader(img, 1.0)
writer = OmeTiffWriter(ri, reg_transform_seq=rts)
writer.write_image_by_plane("testout", output_dir=img_dirname, tile_size=1024)
