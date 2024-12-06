import sys,os
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "..","..","workflow","scripts","utils")))
import wsireg
import json
from wsireg.reg_shapes import RegShapes
from wsireg.reg_transforms.reg_transform_seq import RegTransformSeq
import numpy as np
from utils import setNThreads, snakeMakeMock
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["input_spacing"] = 0.22537
    snakemake.params["output_spacing"] = 0.22537
    snakemake.params["transform_target"] = "postIMS"
    # snakemake.input["postIMC_to_postIMS_transform"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/registrations/postIMC_to_postIMS/test_combined-postIMC_to_postIMS_transformations_mod.json"
    snakemake.input["postIMC_to_postIMS_transform"] = [f"/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/registrations/postIMC_to_postIMS/{core}/test_split_pre_{core}-postIMC_to_postIMS_transformations_mod.json" for core in ["A1","A1","B1"]]
    snakemake.input['IMC_location_on_postIMC'] = [f"/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMC_location/test_split_pre_IMC_mask_on_postIMC_{core}.geojson" for core in ["A1","A1","B1"]]
    # snakemake.input['IMC_location_on_postIMC'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMC_location/test_split_pre_IMC_mask_on_postIMC_A1.geojson"
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# params
output_spacing = snakemake.params["output_spacing"]
input_spacing = snakemake.params["input_spacing"]
transform_target = snakemake.params["transform_target"]
assert(transform_target in ["preIMC", "preIMS", "postIMS"])

# inputs
transform_file_postIMC_to_postIMS=snakemake.input["postIMC_to_postIMS_transform"]
if isinstance(transform_file_postIMC_to_postIMS, str):
    transform_file_postIMC_to_postIMS = [transform_file_postIMC_to_postIMS]
IMC_geojson_file_ls=snakemake.input['IMC_location_on_postIMC']
if isinstance(IMC_geojson_file_ls, str):
    IMC_geojson_file_ls = [IMC_geojson_file_ls]

# outputs
IMC_geojson_transformed_file = snakemake.output["IMC_location_transformed"]
if isinstance(IMC_geojson_transformed_file, list):
    IMC_geojson_transformed_file = IMC_geojson_transformed_file[0]

assert len(transform_file_postIMC_to_postIMS)==len(IMC_geojson_file_ls)

def construct_transform_seq(transform_file):
    # setup transformation sequence
    rtsn=RegTransformSeq(transform_file)
    rtsn.set_output_spacing((float(output_spacing),float(output_spacing)))
    rtls = rtsn.reg_transforms
    logging.info(f"Number of transforms: {len(rtls)}")

    all_linear = np.array([r.is_linear for r in rtls]).all()
    if all_linear:
        assert(len(rtls)==5 or len(rtls)==3)
        is_split_transform = len(rtls)==5
    else:
        # len=4 : direct registration
        # len=6 : additional separate registration between preIMC and preIMS
        assert(len(rtls)==6 or len(rtls)==4)
        is_split_transform = len(rtls)==6


    logging.info("Setup transformation for image")
    if transform_target == "preIMC":
        n_end = 1
    elif transform_target == "preIMS":
        if all_linear:
            n_end = 4 if is_split_transform else 2
        else:
            n_end = 5 if is_split_transform else 3
    elif transform_target == "postIMS":
        if all_linear:
            n_end = 5 if is_split_transform else 3
        else:
            n_end = 6 if is_split_transform else 4
    else:
        raise ValueError("Unknown transform target: " + transform_target)

    rtls = rtsn.reg_transforms
    rtls = rtls[:n_end]

    rtsngeo = RegTransformSeq(rtls, transform_seq_idx=list(range(len(rtls))))
    assert(len(rtls)>0)
    rtsngeo.set_output_spacing((float(output_spacing),float(output_spacing)))

    not_linear = np.array([not r.is_linear for r in rtsngeo.reg_transforms])
    assert np.sum(not_linear)<=1
    if np.sum(not_linear)==1:
        nonlintr = np.array(rtsngeo.reg_transforms)[not_linear][0]
        pseudohash = ";".join([f"{v:.2f}" for v in nonlintr.itk_transform.GetParameters()])
        nonlinind = np.where(not_linear)[0][0]
    else:
        pseudohash = None
        nonlinind = None
    return rtsngeo, pseudohash, nonlinind

logging.info("Load transform")
rtsngeo_ls = []
pseudohash_ls = []
nonlinind_ls = []
for transform_file in transform_file_postIMC_to_postIMS:
    rtsngeo, pseudohash, nonlinind = construct_transform_seq(transform_file)
    rtsngeo_ls.append(rtsngeo)
    pseudohash_ls.append(pseudohash)
    nonlinind_ls.append(nonlinind)

logging.info(f"nonlinind_ls: {nonlinind_ls}")
if np.any([pseudohash is not None for pseudohash in pseudohash_ls]):
    logging.info("Check for duplicate nonlinear transforms")
    matchls = []
    all_match_ls = []
    for i in range(len(pseudohash_ls)):
        for j in range(i+1,len(pseudohash_ls)):
            if pseudohash_ls[i]==pseudohash_ls[j] and pseudohash_ls[i] is not None and pseudohash_ls[j] is not None:
                logging.info(f"Found duplicate nonlinear transform {i} and {j}")
                matchls.append([i,j])
                added_to_set = False
                for k in range(len(all_match_ls)):
                    tmphashes = [pseudohash_ls[match] for match in all_match_ls[k]]
                    if pseudohash_ls[i] in tmphashes or pseudohash_ls[j] in tmphashes:
                        all_match_ls[k].add(i)
                        all_match_ls[k].add(j)
                        added_to_set = True
                        break
                if not added_to_set:
                    all_match_ls.append(set([i,j]))
    all_match_ls = np.array(all_match_ls)

    logging.info("Compute inverse nonlinear transforms")
    has_inverse = list()
    for i in range(len(rtsngeo_ls)):
        if not nonlinind_ls[i] is None:
            has_same = np.array([i in k for k in all_match_ls])
            do_compute_inverse = False
            if np.any(has_same):
                has_same_and_has_inverse = np.array([j in has_inverse for j in list(all_match_ls[has_same].item())])
                print(has_same_and_has_inverse)
                if np.any(has_same_and_has_inverse):
                    ind_same_inverse = np.array(list(all_match_ls[has_same].item()))[has_same_and_has_inverse][0]
                    rtsngeo_ls[i].reg_transforms_itk_order[nonlinind_ls[i]] = rtsngeo_ls[ind_same_inverse].reg_transforms_itk_order[nonlinind_ls[ind_same_inverse]]
                else:
                    do_compute_inverse = True
            else:
                do_compute_inverse = True
            if do_compute_inverse:
                logging.info(f"Compute inverse for {i}")
                rtsngeo_ls[i].reg_transforms_itk_order[nonlinind_ls[i]].compute_inverse_nonlinear()
                has_inverse.append(i)

logging.info("Read json, transform and create shape")
geojson_out_dict = dict()
for i in range(len(IMC_geojson_file_ls)):
    logging.info(f"Read IMC geojson {i}")
    rs = RegShapes(IMC_geojson_file_ls[i], source_res=input_spacing, target_res=output_spacing)
    rs.transform_shapes(rtsngeo_ls[i])

    tmpout = rs.transformed_shape_data[0]['array']
    xmin = np.min(tmpout[:,0])
    assert(xmin>0)
    xmax = np.max(tmpout[:,0])
    assert(xmax<=rtsngeo.output_size[0])
    ymin = np.min(tmpout[:,1])
    assert(ymin>0)
    ymax = np.max(tmpout[:,1])
    assert(ymax<=rtsngeo.output_size[1])

    # rs.save_shape_data(IMC_geojson_transformed_file_ls[i])
    geojson_out_dict[rs.shape_data_gj[0]["properties"]["name"]] = wsireg.reg_shapes.reg_shapes.insert_transformed_pts_gj(
        rs.shape_data_gj, rs.transformed_shape_data
    )

json.dump(
    geojson_out_dict,
    open(
        IMC_geojson_transformed_file,
        "w",
    ),
    indent=1,
)
logging.info(f"Finished")