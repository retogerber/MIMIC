
# Software requirements

Requirements:  
 - This repo, clone with:
```
git clone https://github.com/retogerber/imc_to_ims_workflow.git && cd imc_to_ims_workflow
```
Additional requirements:
- conda and snakemake [Install instructions here](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html#installation)
- singularity/apptainer [Install instructions here](http://apptainer.org/docs/admin/1.1/installation.html#installing-apptainer)
- QuPath [Install instructions here](https://github.com/qupath/qupath/wiki/Installing-QuPath)

# Workflow

## raw data / results structure

For each project (slide) create a project result directory with (replace `PROJECT_NAME`)
```
bash workflow/scripts/setup_project_results_directory.sh -p PROJECT_NAME
```
which will create the following data directory:

```
results/
├── {PROJECT_NAME}
│   ├── data
│   │   ├── cell_overlap
│   │   ├── IMC
│   │   ├── IMC_location
│   │   ├── IMC_mask
│   │   ├── IMS
│   │   ├── postIMC
│   │   ├── postIMS
│   │   ├── preIMC
│   │   ├── preIMC_location
│   │   ├── preIMC_location_combined
│   │   ├── preIMS
│   │   ├── preIMS_location
│   │   ├── preIMS_location_combined
│   │   ├── registration_metric
│   │   └── TMA_location
│   └── registrations
│       ├── IMC_to_IMS
│       ├── IMC_to_preIMC
│       ├── postIMC_to_postIMS
│       └── preIMC_to_preIMS
```
The `data` subdirectory contains all the input data and after workflow execution also the output data.
The `registrations` subdirectory will contain all the output from the registration process.


Three example projects are available in the results directory: 
1. `test_combined`  
    - normal use case
    - one imzML file per project
    - registration of preIMC to preIMS successful in one step (whole slide registration)
2. `test_split_ims`  
    - multiple imzML files per project
    - registration of preIMC to preIMS successful in one step (whole slide registration)
3. `test_split_pre`  
    - one imzML file per project
    - registration of preIMC to preIMS **not** successful in one step, register subsets of slides to each other

use cases 2. and 3. can also be combined.

For all three cases do:

## Setup

The following steps to setup your own projects are required: (as guidance you can also use the directory `results/test_combined`)

### Input data

- .imzml and .ibd files 
    - to `results/{PROJECT_NAME}/data/IMS`
- microscopy images 
    - to their respective folders (e.g. postIMS into `results/{PROJECT_NAME}/data/postIMS`)
    - the expected names are: `{PROJECT_NAME}_{FILETYPE}.ome.tiff` , e.g. for project name `test_combined`: `test_combined_preIMS.ome.tiff`
- IMC images (as single, multi-channel tiff files) 
    - to `results/{PROJECT_NAME}/data/IMC`
- IMC segmentation mask image (as single-channel tiff files) 
    - to `results/{PROJECT_NAME}/data/IMC_mask`
    - IMC images and IMC segmentation mask should have the same name
- IMC summary panels (as csv files)
    - to `results/{PROJECT_NAME}/data/IMC_summary_panel`
    - expected name is `{IMC_NAME}_summary.csv` (replace `{IMC_NAME}` with the corresponding IMC file basename (no extension), e.g. Cirrhosis-TMA-5_New_Detector_001_summary.csv)
    - the structure follows the [steinbock Panel file](https://bodenmillergroup.github.io/steinbock/latest/file-types/) naming scheme: Two columns are required: `channel` (Unique channel ID) and `name` (Unique channel name)


### Linking IMS pixels to postIMS 

Two options are possible, automatically or manually:

#### Automatically

Requirements: 
- Stepsize higher than pixelsize so that individual pixels are clearly separate (depends on experimental setup).
- (Maybe: More than 2 layers of IMS pixels outside of tissue around the whole tissue.)

Adapt file `config/sample_metadata.csv`: (see also section [add metadata](#add-metadata))
- `coords_filename`: empty ()
- `IMS_rotation_angle`: initial rotation angle in degrees of IMS relative to postIMS, should be one of {-270,-180.-90,0,90,180,270}. Default: 0
- `IMS_to_postIMS_n_splits`: number of combinations of weights to detect ablation marks, should be one of {3,5,7,9,11,13,15,17,19}. Higher number means longer runtime but potentially better detection. If pixels are clearly separate a value of 5 should be sufficient. Default: 19
- `IMS_to_postIMS_init_gridsearch`: number of rounds of inital grid search, should be one of {0,1,2,3}. 0 Means no additiional grid search. If pixels are clearly separate a value of 1 should be sufficient. Default: 3
- `within_IMC_fine_registration`: should final fine registration be performed where points in IMC location are upweigthed? Often this is not needed and can be set to False. Should be one of {True, False}. Default: True


#### Manually using napari-imsmicrolink

First create a conda environment with (to speed up this process consider using `mamba`):
```
conda env create -f workflow/env/napari_imsmicrolink.yaml
```
then run 
```
conda run -n napari_imsmicrolink napari
```
This should open `napari`. In `napari` go to `Plugins` -> `IMS MicroLink` which should open `napari-imsmicrolink` on the righthand side. 
Interactively register the two modalities by choosing corresponding pixels on the two images. For more information check out [the github repository](https://github.com/NHPatterson/napari-imsmicrolink).
Two possibilities for registration are possible: IMS to postIMS or postIMS to IMS (preferred) which you can choosing by specifying the target modality in the napari plugin  
- save the output to `results/{PROJECT_NAME}/data/IMS`
- if output generates .ome.tiff (if register postIMS to IMS) move the created `.ome.tiff` file (which should be the postIMS microscopy image registered to IMS) to the directory `results/{PROJECT_NAME}/data/postIMS`, replace the existing file

Then adapt file `config/sample_metadata.csv`: (see also section [add metadata](#add-metadata))
- `coords_filename`: the name of the just generated `.coords` file
- `IMS_rotation_angle`: initial rotation angle in degrees of IMS relative to postIMS, should be one of {-270,-180.-90,0,90,180,270}. Enter the same value as used during interactive registration. Default: 0
 
 If all TMA cores are registred in one step enter the same `coords_filename` for each sample. If registration should be done per TMA core individually (can be more precise) repeat the above procedure for each TMA core.


### Select IMC location on postIMC scan (QuPath)

In QuPath: 
- open postIMC microscopy image
- draw quadrangles around ablation marks from IMC, if automatic registration should be done this can be very approximate and just has to include the whole IMC (see parameter `do_register_IMC_location` in section [add metadata](#add-metadata), essentially set to True if automatic registration should be done). If manual registration is required draw the quadrangles as precicely as possible.
- name each quadrangle by right click -> `Annotations` -> `Set properties` -> `Name`. Keep this name in mind since you will need to fill it into file `config/sample_metadata.csv` (see [below](#add-metadata)). 
- Do this for all IMC ablated areas. 
- Then go to `File` -> `Export objects as GeoJSON`, **Unselect** `Export as FeatureCollection`, then confirm.  Save as `results/{PROJECT_NAME}/data/IMC_location/{PROJECT_NAME}_IMC_mask_on_postIMC.geojson`

The names of annotations are important for correctly matching IMC images to locations.

### Add metadata

Adapt file `config/sample_metadata.csv`:
- `sample_name`: name of sample (i.e. name of IMC image without file ending)
- `project_name`: name of project (above: `{PROJECT_NAME}`)
- `core_name`: name of GeoJSON object from step above, e.g. A1
- (optional) `reg_type`: one of "precise" (default, i.e. registration of IMC to postIMC as described above). Currently only "precise" is suppported.
- (optional) `do_register_IMC_location`: bool, default=True, should automatic IMC to postIMC be performed? This setting means that the manually drawn IMC location mask doesn't need to be precise.
- (optional) `register_IMC_location_with_pycpd`: bool, default=True, in case `do_register_IMC_location` is True, should an additional registration step using CPD be performed. Might or might not improve precision.
- `IMS_pixel_size`: size of IMS pixel in micrometer (stepsize)
- `IMS_shrink_factor`: ratio of IMS stepsize to pixelsize (e.g. if stepsize is 30 and pixelsize (ablation area) is 15 then the ratio is 15/30=0.5). Set to 1 if stepsize is equal to pixelsize. This is mainly used to calculate cell to IMS pixel overlaps. 
- `IMC_pixel_size`: size of IMC pixel in micrometer. For IMC this is usually 1.
- `microscopy_pixel_size`: size of microscopy pixels in micrometer. 
- `coords_filename`: name of `-coords.h5` file as obtained from step `Manually using napari-imsmicrolink` or empty. If present the file should be placed in `results/{PROJECT_NAME}/data/IMS`
- `imzml_filename`: name of `.imzML` file (in directory `results/data/{PROJECT_NAME}/IMS` ), `.ibd` file with same base name has to be present in the same directory
- (optional) `postIMSpreIMSmask`: type of mask extraction for registration between preIMS and postIMS. Default is None (no mask), other available options are "bbox" and "segment". Sometimes the registration between postIMS and preIMS is difficult because of the MALDI matrix on the postIMS image. Therefore to guide the registration the optimization of the transformation can be restricted to areas containing a TMA core. 
- (optional) `IMS_rotation_angle`: int, default=0, one of 0,90,180,270. Initial IMS rotation angle relative to postIMS microscopy image.
- (optional) `IMC_rotation_angle`: int, default=0, one of 0,90,180,270. Initial IMC rotation angle relative to postIMC microscopy image.
- (optional) `IMS_to_postIMS_n_splits`: int, default=19, should be a whole number between 3 and 19, number of splits for thresholding to find IMS ablation marks.
- (optional) `IMS_to_postIMS_init_gridsearch`: int, default=3, should be a whole number between 0 and 3, number of rounds of gridsearch for IMS ablation mark to IMS pixel initial registration.
- (optional) `within_IMC_fine_registration`: bool, default=True, should an additional registration step IMS to postIMS be done only taking into account IMS pixels within the location of IMC.
- (optional) `min_index_length`: int, default=10, minimum number of neighbors to take into account for matching IMS ablation marks with IMS pixels.
- (optional) `max_index_length`: int, default=30, maximum number of neighbors to take into account for matching IMS ablation marks with IMS pixels.

### Update config

Adapt file `config/config.yaml`:
- `IMC_channels_for_aggr`: specify which IMC channels should be used. Names should match values in column `name` in IMC summary panel files.
- (optional) `QC_metrics`: specify which QC metrics should be calculated. Default: all
- (optional) `QC_steps`: specify for which registration step QC metrics should be calculated. Default: all

### Register preIMC to preIMS (case 3. only)

For case 3. additionally the regions that should be independently registered have to be selected.  
In QuPath: 
- open preIMC microscopy image
- draw quadrangles around regions to be registered independently. Should contain at least one full core.
- name each quadrangle by right click -> `Annotations` -> `Set properties` -> `Name`. Keep this name in mind since the corresponding core on the preIMS microscopy image has to have the same name. 
- Do this for all regions/cores where the registration didn't work properly. 
- Then go to `File` -> `Export objects as GeoJSON`, **Unselect** `Export as FeatureCollection`, then confirm.  Save as `results/{PROJECT_NAME}/data/preIMC_location_combined/{PROJECT_NAME}_reg_mask_on_preIMC.geojson`

- Do the same for the preIMS microscopy image
- Make sure the names of the corresponding regions are the same
- Save as `results/{PROJECT_NAME}/data/preIMS_location_combined/{PROJECT_NAME}_reg_mask_on_preIMS.geojson`

### Data obtained from the same slide

In the case that the data was generated from the same slide the preIMS and preIMC microscopy images are effectively the same (or only one of the two exists). In this scenario add one of the two images (preIMS or preIMC) to its respective location and then create a symlink (or a copy) for the other image. The workflow will check if the preIMS and the preIMC images are the same, and if they are, the registration step between the two is dropped.

## Run workflow

To run the workflow:
```
snakemake --use-conda --use-singularity -c 1
```
If there are links to files outside of the project directory adding the flag ` --singularity-args "--bind $HOME"` might be needed:
```
snakemake --use-conda --use-singularity -c 1  --singularity-args "--bind $HOME"
```
For all snakemake cli options see [here](https://snakemake.readthedocs.io/en/stable/executing/cli.html).

### Run individual parts of the workflow

Individual parts of the workflow can be run separately with the following:

```
snakemake --use-conda --use-singularity -c 1 -R PART
```

where `PART` is one of `IMS`, `regIMS`, `regIMC`, `Microscopy`, `overlap`, `QC`.

### Specify which QC metrics to calculate

In the file `config/config.yaml` adapt the variables `QC_metrics` and `QC_steps` to only evaluate certain registration steps of the workflow. This can be useful for faster iteration if a certain step needs to be optimized.



## Check output

After successful execution of the workflow, a registration report can be found at `results/{PROJECT_NAME}/data/registration_metric/report/{PROJECT_NAME}_registration_evaluation.html`. It contains all the registration steps and should provide a quick overview if registration was succesfull for individual TMA cores.

Additionally the following sections might be helpful.

### Automatic linking of IMS pixels to postIMS

For each sample check output of `results/{PROJECT_NAME}/data/registration_metric/{SAMPLE_NAME}_IMS_to_postIMS_reg_metrics_auto.ome.tiff` (e.g. using QuPath). Blue dots should be in the middle of the IMS pixels. If dots are too far off manual registration is required (See [above](#manually-using-napari-imsmicrolink)).

### Automatic linking of IMC location to postIMC

For each sample check output of `logs/IMC_location_from_postIMC/{PROJECT_NAME}_{CORE_NAME}.ome.tiff`


## Workflow output

The main output of the workflow is for each input imzml file the per IMS pixel and IMC cell aggregated results: `results/{PROJECT_NAME}/data/cell_overlap/{PROJECT_NAME}_{imzml_filename}_peak_cell_overlap.csv`

Additionally the following ouput images are created:
- the directory `results/{PROJECT_NAME}/data/IMS/{PROJECT_NAME}_IMS_on_postIMS` contains one image for each m/z value.
- `results/{PROJECT_NAME}/data/IMC_mask/{PROJECT_NAME}_IMC_transformed_on_postIMC.ome.tiff` contains the transformed IMC masks
- `results/{PROJECT_NAME}/data/IMC/{PROJECT_NAME}_IMC_aggr_transformed.ome.tiff` contains the transformed (on postIMS) IMC image
- `results/{PROJECT_NAME}/data/postIMC/{PROJECT_NAME}_postIMC_transformed_on_postIMS.ome.tiff` contains the transformed postIMC microscopy image

# Downstream analysis

Example downstream analysis (R markdown) scripts can be found in [workflow/scripts/Downstream_Analysis](../workflow/scripts/Downstream_Analysis). Those show how to test for associations between cell types and m/z analytes. The ordering in which the scripts have to be run follow from the naming. In-script adaption are necessary for this to work. 