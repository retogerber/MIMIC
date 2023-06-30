import sys
import json

# transform_file_postIMC_to_preIMC = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/registrations/postIMC_to_postIMS/test_split_pre-postIMC_to_postIMS_transformations.json"
transform_file_postIMC_to_preIMC = snakemake.input["postIMC_to_preIMC_transform"]
# transform_file_preIMC_to_preIMS = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/registrations/preIMC_to_preIMS/A1/test_split_pre_A1-preIMC_to_preIMS_transformations.json"
transform_file_preIMC_to_preIMS = snakemake.input["preIMC_to_preIMS_transform"]
# orig_size_tform_preIMC_to_preIMS = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/registrations/preIMC_to_preIMS/A1/.imcache_test_split_pre_A1/preIMS_orig_size_tform.json"
orig_size_tform_preIMC_to_preIMS=snakemake.input["preIMS_orig_size_transform"]
# transform_file_preIMS_to_postIMS = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/registrations/postIMC_to_postIMS/test_split_pre-preIMS_to_postIMS_transformations.json"
transform_file_preIMS_to_postIMS=snakemake.input["preIMS_to_postIMS_transform"]

# postIMC_to_postIMS_transform = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/registrations/postIMC_to_postIMS/A1/test_split_pre_A1-postIMC_to_postIMS_transformations.json"
postIMC_to_postIMS_transform = snakemake.output["postIMC_to_postIMS_transform"]

# load all transforms
j0 = json.load(open(transform_file_postIMC_to_preIMC, "r"))
j0 = {'000-to-preIMC':j0['000-to-preIMC']}
j1 = json.load(open(transform_file_preIMC_to_preIMS, "r"))
j2 = json.load(open(orig_size_tform_preIMC_to_preIMS, "r"))
# set correct spacing because of used downsampling=2 in registration
j2['Spacing'] = [ str(float(e)/2) for e in j2['Spacing']]
# j2 is a single transform, add a name to it
j2_new = {"orig_size_tform_preIMC_to_preIMS":j2}

j3 = json.load(open(transform_file_preIMS_to_postIMS, "r"))

# combine all transforms
j0.update(j1)
j0.update(j2_new)
j0.update(j3)

json.dump(j0, open(postIMC_to_postIMS_transform,"w"))