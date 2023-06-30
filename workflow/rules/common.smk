import pandas as pd
import numpy as np

def read_sample_metadata(filename="config/sample_metadata.csv"):
    if os.path.isfile(filename):
        with open(filename, 'r') as fil:
            sample_metadata_df = pd.read_csv(fil)
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
            #core=core_name_from_sample_name(wildcards)
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
            #core=core_name_from_sample_name(wildcards)
            core = get_column_entry_from_metadata_two_conditions(wildcards.sample, wildcards.project_name, "core_name", "sample_name", "project_name", read_sample_metadata(config["sample_metadata"]))
            part=df.loc[df["core"] == core]["preIMC_location"].tolist()[0]
            outfile = f'results/{wildcards.project_name}/registrations/postIMC_to_postIMS/{part}/{wildcards.project_name}_{part}-postIMC_to_postIMS_transformations.json'
            return outfile




