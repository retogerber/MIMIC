import pandas as pd
import numpy as np
from snakemake.utils import validate
import json
import os

def read_sample_metadata(filename="config/sample_metadata.csv", validator=os.path.abspath("workflow/sample_metadata.schema.yaml")):
    if not os.path.isfile(validator):
        sys.exit(f"Validator file {validator} not found!")

    if os.path.isfile(filename):
        with open(filename, 'r') as fil:
            sample_metadata_df = pd.read_csv(fil)
            validate(sample_metadata_df, validator)
            return sample_metadata_df
    else:
        sys.exit("File config/sample_metadata.csv not found!")


def get_column_entry_from_metadata(
    cond_name, 
    column_out,   
    cond_column="sample_name",
    sample_metadata_df = read_sample_metadata(),
    return_all=False):
    """filter metadata df according to entry and column"""
    sample_metadata_df = sample_metadata_df.fillna('')
    df_sub = sample_metadata_df.loc[sample_metadata_df[cond_column] == cond_name]
    if return_all:
        return df_sub[column_out].tolist()
    else:
        return df_sub[column_out].tolist()[0]


def get_column_entry_from_metadata_two_conditions(
    cond_name_1,
    cond_name_2,
    column_out,
    cond_column_1="sample_name",
    cond_column_2="project_name",
    sample_metadata_df = read_sample_metadata(),
    return_all=False):
    """filter metadata df according to entry and column"""
    sample_metadata_df = sample_metadata_df.fillna('')
    inds_arr = np.logical_and(sample_metadata_df[cond_column_1] == cond_name_1, sample_metadata_df[cond_column_2] == cond_name_2)
    df_sub = sample_metadata_df.loc[inds_arr]
    if return_all:
        return df_sub[column_out].tolist()
    else:
        return df_sub[column_out].tolist()[0]


def IMC_aggr_sample_name_from_core_name(wildcards):
    sample_name = get_column_entry_from_metadata_two_conditions(wildcards.core, wildcards.project_name, "sample_name", "core_name", "project_name", read_sample_metadata(config["sample_metadata"]))

    return os.path.join("results",wildcards.project_name,"data","IMC",f"{sample_name}_aggr.ome.tiff")

def IMC_location_from_project_name(wildcards):
    core_name_l = get_column_entry_from_metadata(wildcards.project_name, "core_name", "project_name", read_sample_metadata(config["sample_metadata"]), return_all=True)
    files = [ f'results/{wildcards.project_name}/data/IMC_location/{wildcards.project_name}_IMC_mask_on_preIMC_{c}.geojson' for c in core_name_l ]
    return files

def IMC_location_from_project_name_and_sample_name(wildcards, target=""):
    assert(target in ["postIMC","preIMC","preIMS","postIMS"])
    core=get_column_entry_from_metadata_two_conditions(
        wildcards.project_name,
        wildcards.sample,
        "core_name",
        "project_name",
        "sample_name",
        read_sample_metadata(config["sample_metadata"]),
        return_all=False,
    )
    return f"results/{wildcards.project_name}/data/IMC_location/{wildcards.project_name}_IMC_mask_on_{target}_{core}.geojson",

def TMA_location_from_project_name_and_sample_name(wildcards, target=""):
    assert(target in ["postIMC","preIMC","preIMS","postIMS"])
    core=get_column_entry_from_metadata_two_conditions(
        wildcards.project_name,
        wildcards.sample,
        "core_name",
        "project_name",
        "sample_name",
        read_sample_metadata(config["sample_metadata"]),
        return_all=False,
    )
    return f"results/{wildcards.project_name}/data/TMA_location/{wildcards.project_name}_TMA_location_on_{target}_{core}.geojson",


def IMC_to_preIMC_transform_pkl_core_name_from_sample_name(wildcards):
    core_name = get_column_entry_from_metadata_two_conditions(wildcards.sample, wildcards.project_name, "core_name", "sample_name", "project_name", read_sample_metadata(config["sample_metadata"]))
    reg_type = get_column_entry_from_metadata(wildcards.project_name, "IMS_pixel_size", "project_name", read_sample_metadata(config["sample_metadata"])),
    if reg_type == "register":
        return os.path.join("results",wildcards.project_name,"registrations","IMC_to_preIMC",f"{wildcards.project_name}_{core_name}",f"{wildcards.project_name}_{core_name}-IMC_to_preIMC_transformations.json")
    else:
        return os.path.join("results",wildcards.project_name,"registrations","IMC_to_preIMC",f"{wildcards.project_name}_{core_name}",f"{wildcards.project_name}_{core_name}-precise_IMC_to_preIMC_transformations.json")


def preIMC_orig_size_transform_core_name_from_sample_name(wildcards):
    core_name = get_column_entry_from_metadata_two_conditions(wildcards.sample, wildcards.project_name, "core_name", "sample_name", "project_name", read_sample_metadata(config["sample_metadata"]))
    reg_type = get_column_entry_from_metadata(wildcards.project_name, "IMS_pixel_size", "project_name", read_sample_metadata(config["sample_metadata"])),
    if reg_type == "register":
        return os.path.join("results",wildcards.project_name,"registrations","IMC_to_preIMC",f"{wildcards.project_name}_{core_name}",f".imcache_{wildcards.project_name}_{core_name}","preIMC_orig_size_tform.json")
    else:
        return os.path.join("results",wildcards.project_name,"registrations","IMC_to_preIMC",f"{wildcards.project_name}_{core_name}",f"{wildcards.project_name}_{core_name}_precise_preIMC_orig_size_tform.json")



def decide_use_direct_preIMC_to_postIMS_transform(wildcards):
    if isinstance(wildcards, dict):
        project_name = wildcards["project_name"]
    else:
        project_name = wildcards.project_name
    table_file = checkpoints.create_preIMS_mask_table.get(project_name=project_name).output['table']
    if os.stat(table_file).st_size == 0:
        return True
    else:
        return False

def choose_preIMC_to_postIMS_transform(wildcards):
    if isinstance(wildcards, dict):
        project_name = wildcards["project_name"]
        if 'core' in wildcards.keys():
            core = wildcards["core"]
        if 'sample' in wildcards.keys():
            sample = wildcards["sample"]
    else:
        project_name = wildcards.project_name
        if 'core' in wildcards.keys():
            core = wildcards.core
        if 'sample' in wildcards.keys():
            sample = wildcards.sample
    use_direct = decide_use_direct_preIMC_to_postIMS_transform(wildcards)
    if use_direct:
        outfile = f'results/{project_name}/registrations/postIMC_to_postIMS/{project_name}-preIMC_to_postIMS_transformations_mod.json'
        return outfile
    else:
        match_preIMC_location_with_IMC_location_file = checkpoints.match_preIMC_location_with_IMC_location.get(project_name=project_name).output['matching']
        with match_preIMC_location_with_IMC_location_file.open() as f:
            df = pd.read_csv(f) 
            if not 'core' in wildcards.keys():
                core = get_column_entry_from_metadata_two_conditions(sample, project_name, "core_name", "sample_name", "project_name", read_sample_metadata(config["sample_metadata"]))
            part=df.loc[df["core"] == core]["preIMC_location"].tolist()[0]
            outfile = f'results/{project_name}/registrations/preIMC_to_preIMS/{part}/{project_name}_{part}-preIMC_to_postIMS_transformations_mod.json'
            return outfile



def choose_postIMC_to_postIMS_transform(wildcards):
    if isinstance(wildcards, dict):
        project_name = wildcards["project_name"]
        if 'core' in wildcards.keys():
            core = wildcards["core"]
        if 'sample' in wildcards.keys():
            sample = wildcards["sample"]
    else:
        project_name = wildcards.project_name
        if 'core' in wildcards.keys():
            core = wildcards.core
        if 'sample' in wildcards.keys():
            sample = wildcards.sample
    use_direct = decide_use_direct_preIMC_to_postIMS_transform(wildcards)
    if use_direct:
        outfile = f'results/{project_name}/registrations/postIMC_to_postIMS/{project_name}-postIMC_to_postIMS_transformations_mod.json'
        return outfile
    else:
        match_preIMC_location_with_IMC_location_file = checkpoints.match_preIMC_location_with_IMC_location.get(project_name=project_name).output['matching']
        with match_preIMC_location_with_IMC_location_file.open() as f:
            df = pd.read_csv(f) 
            if not 'core' in wildcards.keys():
                core = get_column_entry_from_metadata_two_conditions(sample, project_name, "core_name", "sample_name", "project_name", read_sample_metadata(config["sample_metadata"]))
            part=df.loc[df["core"] == core]["preIMC_location"].tolist()[0]
            outfile = f'results/{project_name}/registrations/postIMC_to_postIMS/{part}/{project_name}_{part}-postIMC_to_postIMS_transformations_mod.json'
            return outfile

def choose_postIMC_to_postIMS_transform_ls(project_name, core_names, transform_target=None):
    transform_files = list()
    for core_name in core_names:
        if transform_target == "preIMC":
            transform_files.append(f"results/{project_name}/registrations/postIMC_to_postIMS/{project_name}-postIMC_to_postIMS_transformations_mod.json")
        else:
            transform_files.append(choose_postIMC_to_postIMS_transform({"project_name": project_name, "core": core_name}))
    return transform_files


def choose_postIMC_to_postIMS_transform_all(wildcards, transform_target=None):
    project_name = wildcards.project_name
    core_names = get_column_entry_from_metadata(project_name, "core_name", "project_name", read_sample_metadata(config["sample_metadata"]), return_all=True)
    return choose_postIMC_to_postIMS_transform_ls(project_name, core_names, transform_target)

#def choose_imsml_coordsfile(wildcards):
#        filename = get_column_entry_from_metadata_two_conditions(wildcards.sample, wildcards.project_name, "coords_filename", "sample_name", "project_name", read_sample_metadata(config["sample_metadata"]))
#        filename = str(filename).strip()
#        if filename == "":
#            filename_out = f"results/{wildcards.project_name}/data/IMS/postIMS_to_IMS_{wildcards.project_name}-{wildcards.sample}-IMSML-coords.h5"
#        else:
#            filename_out = f"results/{wildcards.project_name}/data/IMS/{filename}"
#        return filename_out

def choose_imsml_coordsfile_base(sample, project_name):
    filename = get_column_entry_from_metadata_two_conditions(sample, project_name, "coords_filename", "sample_name", "project_name", read_sample_metadata(config["sample_metadata"]))
    filename = str(filename).strip()
    if filename == "":
        filename_out = f"results/{project_name}/data/IMS/postIMS_to_IMS_{project_name}-{sample}-IMSML-coords.h5"
    else:
        filename_out = f"results/{project_name}/data/IMS/{filename}"
    return filename_out

def choose_imsml_coordsfile(wildcards):
    return choose_imsml_coordsfile_base(wildcards.sample, wildcards.project_name)

def choose_all_imsml_coordsfile_from_project(wildcards):
    sample_names = get_column_entry_from_metadata(wildcards.project_name, "sample_name", "project_name", read_sample_metadata(config["sample_metadata"]), return_all = True)
    return [choose_imsml_coordsfile_base(sample, wildcards.project_name) for sample in sample_names]

def choose_imsml_metafile_base(sample, project_name):
    filename = get_column_entry_from_metadata_two_conditions(sample, project_name, "coords_filename", "sample_name", "project_name", read_sample_metadata(config["sample_metadata"]))
    filename = str(filename).strip().replace("-coords.h5","-meta.json")
    if filename == "":
        filename_out = f"results/{project_name}/data/IMS/postIMS_to_IMS_{project_name}-{sample}-IMSML-meta.json"
    else:
        filename_out = f"results/{project_name}/data/IMS/{filename}"
    return filename_out

def choose_all_imsml_metafile_from_project(wildcards):
    sample_names = get_column_entry_from_metadata(wildcards.project_name, "sample_name", "project_name", read_sample_metadata(config["sample_metadata"]), return_all = True)
    return [choose_imsml_metafile_base(sample, wildcards.project_name) for sample in sample_names]


def imzml_peaks_from_sample_and_project(wildcards):
    imzml_file=get_column_entry_from_metadata_two_conditions(
            wildcards.project_name,
            wildcards.sample,
            "imzml_filename",
            "project_name",
            "sample_name",
            read_sample_metadata(config["sample_metadata"]),
        )
    imzml_base = imzml_file.replace(".imzML","")
    return f"results/{wildcards.project_name}/data/IMS/{imzml_base}_peaks.h5"

def imzml_peaks_from_project(wildcards):
    sample_names = get_column_entry_from_metadata(wildcards.project_name, "sample_name", "project_name", read_sample_metadata(config["sample_metadata"]), return_all = True)
    imzml_files=[get_column_entry_from_metadata_two_conditions(
            wildcards.project_name,
            sample,
            "imzml_filename",
            "project_name",
            "sample_name",
            read_sample_metadata(config["sample_metadata"])
        ) for sample in sample_names]
    imzml_bases = [imzml_file.replace(".imzML","") for imzml_file in imzml_files]
    peak_files = [f"results/{wildcards.project_name}/data/IMS/{imzml_base}_peaks.h5" for imzml_base in imzml_bases]
    return peak_files


def choose_IMS_to_postIMS_svg(wildcards):
        sample_names = get_column_entry_from_metadata(wildcards.project_name, "sample_name", "project_name", read_sample_metadata(config["sample_metadata"]), return_all = True)
        filenames_out = list()
        for sample in sample_names:
            filename = get_column_entry_from_metadata_two_conditions(sample, wildcards.project_name, "coords_filename", "sample_name", "project_name", read_sample_metadata(config["sample_metadata"]))
            filename = str(filename).strip()
            if filename == "":
                filenames_out.append(f"results/{wildcards.project_name}/data/registration_metric/report/{wildcards.project_name}_{sample}_IMS_to_postIMS_combined_registration_all.svg")
        return filenames_out

def choose_IMS_to_postIMS_png(wildcards):
        sample_names = get_column_entry_from_metadata(wildcards.project_name, "sample_name", "project_name", read_sample_metadata(config["sample_metadata"]), return_all = True)
        filenames_out = list()
        for sample in sample_names:
            filename = get_column_entry_from_metadata_two_conditions(sample, wildcards.project_name, "coords_filename", "sample_name", "project_name", read_sample_metadata(config["sample_metadata"]))
            filename = str(filename).strip()
            if filename == "":
                filenames_out.append(f"results/{wildcards.project_name}/data/registration_metric/report/{sample}_IMS_to_postIMS_reg_metrics_auto.png")
        return filenames_out



def choose_imsml_metafile(wildcards):
        filename = get_column_entry_from_metadata_two_conditions(wildcards.sample, wildcards.project_name, "coords_filename", "sample_name", "project_name", read_sample_metadata(config["sample_metadata"]))
        filename = str(filename).strip()
        if filename == "":
            filename_out = f"results/{wildcards.project_name}/data/IMS/postIMS_to_IMS_{wildcards.project_name}-{wildcards.sample}-IMSML-coords.h5"
        else:
            filename_out = f"results/{wildcards.project_name}/data/IMS/{filename}"
        return filename_out.replace("-coords.h5","-meta.json")


def decide_IMS_to_postIMS_reg_metrics_auto_or_not(wildcards):
        samples=get_column_entry_from_metadata(wildcards.project_name, "sample_name", "project_name", read_sample_metadata(config["sample_metadata"]), return_all = True)
        filename = [get_column_entry_from_metadata_two_conditions(s, wildcards.project_name, "coords_filename", "sample_name", "project_name", read_sample_metadata(config["sample_metadata"])) for s in samples]
        filename = [str(f).strip() for f in filename]
        out = ["_auto" if f == ""  else "" for f in filename]
        return out


def choose_imsml_coordsfile_from_imzml(wildcards):
        sample_names = get_column_entry_from_metadata_two_conditions(f"{wildcards.imzml_base}.imzML", wildcards.project_name, "sample_name", "imzml_filename", "project_name", read_sample_metadata(config["sample_metadata"]), return_all=True)
        filenames = []
        for sample in sample_names:
            filename = get_column_entry_from_metadata_two_conditions(sample, wildcards.project_name, "coords_filename", "sample_name", "project_name", read_sample_metadata(config["sample_metadata"]))
            filename = str(filename).strip()
            if filename == "":
                filename_out = f"results/{wildcards.project_name}/data/IMS/postIMS_to_IMS_{wildcards.project_name}-{sample}-IMSML-coords.h5"
            else:
                filename_out = f"results/{wildcards.project_name}/data/IMS/{filename}"
            filenames.append(filename_out)
        return filenames


def sample_core_names(wildcards):
    sample_names = get_column_entry_from_metadata_two_conditions(f"{wildcards.imzml_base}.imzML", wildcards.project_name, "sample_name", "imzml_filename", "project_name", read_sample_metadata(config["sample_metadata"]), return_all=True)
    core_names = get_column_entry_from_metadata_two_conditions(f"{wildcards.imzml_base}.imzML", wildcards.project_name, "core_name", "imzml_filename", "project_name", read_sample_metadata(config["sample_metadata"]), return_all=True)
    return f'{"|-|-|".join(str(sample_names[i])+"|-_-|"+str(core_names[i]) for i in range(len(sample_names)))}'


def n_threads_for_register_IMS_to_postIMS_single_core_1(wildcards, sample_metadata_df, n_threads_max=1):
    IMS_to_postIMS_n_splits=get_column_entry_from_metadata_two_conditions(
        wildcards.project_name,
        wildcards.sample,
        "IMS_to_postIMS_n_splits",
        "project_name",
        "sample_name",
        read_sample_metadata(sample_metadata_df),
    )
    IMS_to_postIMS_init_gridsearch=get_column_entry_from_metadata_two_conditions(
        wildcards.project_name,
        wildcards.sample,
        "IMS_to_postIMS_init_gridsearch",
        "project_name",
        "sample_name",
        read_sample_metadata(sample_metadata_df),
    )
    return int(min([n_threads_max,max([IMS_to_postIMS_n_splits/2,4])]))


def is_linear_transform_single(transform):
    try:
        tr =json.load(open(transform))
        trls = list()
        for k in tr.keys():
            for i in range(len(tr[k])):
                trls.append(tr[k][i]["Transform"][0])
        if "BSplineTransform" in trls:
            return False
        else:
            return True
    except:
        return False

def is_linear_transform(transform):
    if isinstance(transform, list):
        return any([is_linear_transform_single(t) for t in transform])
    else:
        return is_linear_transform_single(transform)

def decide_postIMSpreIMSmask(wildcards, type):
    assert type in ["preIMS","postIMS"]
    postIMSpreIMSmask=get_column_entry_from_metadata(
        wildcards.project_name,
        "postIMSpreIMSmask",
        "project_name",
        read_sample_metadata(config["sample_metadata"]),
    )
    assert postIMSpreIMSmask in ["none","bbox","segment"]
    table_file = checkpoints.create_postIMSpreIMS_mask_table.get(project_name=wildcards.project_name).output['table']
    df = pd.read_csv(table_file) 
    geojson_exists=df.loc[df["type"] == type]["exists"].tolist()[0]!=0
    if geojson_exists and postIMSpreIMSmask != "none":
        filename=f"results/{wildcards.project_name}/data/{type}/{wildcards.project_name}_{type}_mask_for_reg_geojson.ome.tiff",
    elif postIMSpreIMSmask == "segment" and not geojson_exists:
        filename=f"results/{wildcards.project_name}/data/{type}/{wildcards.project_name}_{type}_mask_for_reg_nogeojson.ome.tiff",
    else:
        filename = f"results/{wildcards.project_name}/data/{type}/{wildcards.project_name}_{type}.ome.tiff"
    return filename

def return_file_or_generic(file, bool, generic = config["generic_input"]):
    if bool:
        return file
    else:
        return generic


def checkpoint_input_file_exists_or_generic(file, generic = config["generic_input"]):
    bool = os.path.isfile(file)
    if bool:
        return file
    else:
        return generic