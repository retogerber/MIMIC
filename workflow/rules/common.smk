import pandas as pd
import numpy as np
from snakemake.utils import validate
import snakemake.workflow

def read_sample_metadata(filename="config/sample_metadata.csv", validator=snakemake.workflow.srcdir("../sample_metadata.schema.yaml")):
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
    table_file = checkpoints.create_preIMS_mask_table.get(project_name=wildcards.project_name).output['table']
    if os.stat(table_file).st_size == 0:
        return True
    else:
        return False

def choose_preIMC_to_postIMS_transform(wildcards):
    use_direct = decide_use_direct_preIMC_to_postIMS_transform(wildcards)
    if use_direct:
        outfile = f'results/{wildcards.project_name}/registrations/postIMC_to_postIMS/{wildcards.project_name}-preIMC_to_postIMS_transformations.json'
        return outfile
    else:
        match_preIMC_location_with_IMC_location_file = checkpoints.match_preIMC_location_with_IMC_location.get(project_name=wildcards.project_name).output['matching']
        with match_preIMC_location_with_IMC_location_file.open() as f:
            df = pd.read_csv(f) 
            if 'core' in wildcards.keys():
                core = wildcards.core
            else:
                core = get_column_entry_from_metadata_two_conditions(wildcards.sample, wildcards.project_name, "core_name", "sample_name", "project_name", read_sample_metadata(config["sample_metadata"]))
            part=df.loc[df["core"] == core]["preIMC_location"].tolist()[0]
            outfile = f'results/{wildcards.project_name}/registrations/preIMC_to_preIMS/{part}/{wildcards.project_name}_{part}-preIMC_to_postIMS_transformations.json'
            return outfile



def choose_postIMC_to_postIMS_transform(wildcards):
    use_direct = decide_use_direct_preIMC_to_postIMS_transform(wildcards)
    if use_direct:
        outfile = f'results/{wildcards.project_name}/registrations/postIMC_to_postIMS/{wildcards.project_name}-postIMC_to_postIMS_transformations.json'
        return outfile
    else:
        match_preIMC_location_with_IMC_location_file = checkpoints.match_preIMC_location_with_IMC_location.get(project_name=wildcards.project_name).output['matching']
        with match_preIMC_location_with_IMC_location_file.open() as f:
            df = pd.read_csv(f) 
            if 'core' in wildcards.keys():
                core = wildcards.core
            else:
                core = get_column_entry_from_metadata_two_conditions(wildcards.sample, wildcards.project_name, "core_name", "sample_name", "project_name", read_sample_metadata(config["sample_metadata"]))
            part=df.loc[df["core"] == core]["preIMC_location"].tolist()[0]
            outfile = f'results/{wildcards.project_name}/registrations/postIMC_to_postIMS/{part}/{wildcards.project_name}_{part}-postIMC_to_postIMS_transformations.json'
            return outfile


def choose_imsml_coordsfile(wildcards):
        filename = get_column_entry_from_metadata_two_conditions(wildcards.sample, wildcards.project_name, "coords_filename", "sample_name", "project_name", read_sample_metadata(config["sample_metadata"]))
        filename = str(filename).strip()
        if filename == "":
            filename_out = f"results/{wildcards.project_name}/data/IMS/postIMS_to_IMS_{wildcards.project_name}-{wildcards.sample}-IMSML-coords.h5"
        else:
            filename_out = f"results/{wildcards.project_name}/data/IMS/{filename}"
        return filename_out

def choose_imsml_coordsfile(wildcards):
        filename = get_column_entry_from_metadata_two_conditions(wildcards.sample, wildcards.project_name, "coords_filename", "sample_name", "project_name", read_sample_metadata(config["sample_metadata"]))
        filename = str(filename).strip()
        if filename == "":
            filename_out = f"results/{wildcards.project_name}/data/IMS/postIMS_to_IMS_{wildcards.project_name}-{wildcards.sample}-IMSML-coords.h5"
        else:
            filename_out = f"results/{wildcards.project_name}/data/IMS/{filename}"
        return filename_out

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
        sample_names = get_column_entry_from_metadata_two_conditions(wildcards.imzml_base+".imzML", wildcards.project_name, "sample_name", "imzml_filename", "project_name", read_sample_metadata(config["sample_metadata"]), return_all=True)
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
    sample_names = get_column_entry_from_metadata_two_conditions(wildcards.imzml_base+".imzML", wildcards.project_name, "sample_name", "imzml_filename", "project_name", read_sample_metadata(config["sample_metadata"]), return_all=True)
    core_names = get_column_entry_from_metadata_two_conditions(wildcards.imzml_base+".imzML", wildcards.project_name, "core_name", "imzml_filename", "project_name", read_sample_metadata(config["sample_metadata"]), return_all=True)
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



