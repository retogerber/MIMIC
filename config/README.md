
# Software requirements

- conda and snakemake [Install instructions here](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html#installation)
- singularity/apptainer [Install instructions here](http://apptainer.org/docs/admin/1.1/installation.html#installing-apptainer)
- QuPath [Install instructions here](https://github.com/qupath/qupath/wiki/Installing-QuPath)

# Workflow

## raw data / results structure

For each project (slide) create a project result directory with (replace `PROJECT_NAME`)
```
bash workflow/scripts/setup_project_results_directory.sh -p PROJECT_NAME
```
which will create the following results directory:

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
│   │   └── registration_metric
│   └── registrations
│       ├── IMC_to_IMS
│       ├── IMC_to_preIMC
│       ├── postIMC_to_postIMS
│       └── preIMC_to_preIMS
```
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


## Linking IMS pixels to postIMS 

Two options are possible, automatically or manually:

### Automatically

Requirements: 
- More than 2 layers of IMS pixels outside of tissue around the whole tissue.  
- Stepsize higher than pixelsize so that individual pixels are clearly separate.
If not around whole tissue automatic linking might not work.


Adapt file `config/sample_metadata.csv`: (see also section [add metadata](#add-metadata))
- `coords_filename`: empty ()
- `IMS_rotation_angle`: initial rotation angle in degrees of IMS relative to postIMS, should be one of {-270,-180.-90,0,90,180,270}
- `IMS_to_postIMS_n_splits`: number of combinations of weights to detect ablation marks, should be one of {3,5,7,9,11,13,15,17,19}. Higher number means longer runtime but potentially better detection
- `IMS_to_postIMS_init_gridsearch`: number of rounds of inital grid search, should be one of {0,1,2,3}. 0 Means no additiional grid search.
- `within_IMC_fine_registration`: should final fine registration be performed where points in IMC location are upweigthed? Should be one of {True, False}.


### Manually using napari-imsmicrolink

First create a conda environment with (to speed up this process consider using `mamba`):
```
conda env create -f workflow/env/napari_imsmicrolink.yaml
```
then run 
```
conda run -n napari_imsmicrolink napari
```
This should open `napari`. In `napari` go to `Plugins` -> `IMS MicroLink` which should open `napari-imsmicrolink` on the righthand side. 
Interactively register the two modalities by choosing corresponding pixels on the two images.
Two possibilities for registration are possible: IMS to postIMS or postIMS to IMS (preferred) which you can choosing by specifying the target modality in the napari plugin  
- save the output to `results/{PROJECT_NAME}/data/IMS`
- if output generates .ome.tiff (if register postIMS to IMS) move the created `.ome.tiff` file (which should be the postIMS microscopy image registered to IMS) to the directory `results/{PROJECT_NAME}/data/postIMS`, replace the existing file



## Register IMC to postIMC (QuPath)


In QuPath: 
- open postIMC microscopy image
- draw quadrangles around ablation marks from IMC, if automatic registration should be done this can be very approximate and just has to include the whole IMC (see parameter `do_register_IMC_location` in section [add metadata](#add-metadata)). If manual registration is required draw the quadrangles as precicely as possible.
- name each quadrangle by right click -> `Annotations` -> `Set properties` -> `Name`. Keep this name in mind since you will need to fill it into file `config/sample_metadata.csv` (see [below](#add-metadata)). 
- Do this for all IMC ablated areas. 
- Then go to `File` -> `Export objects as GeoJSON`, **Unselect** `Export as FeatureCollection`, then confirm.  Save as `results/{PROJECT_NAME}/data/IMC_location/{PROJECT_NAME}_IMC_mask_on_postIMC.geojson`



## Add metadata

Adapt file `config/sample_metadata.csv`:
- `sample_name`: name of sample (i.e. name of IMC image without file ending)
- `project_name`: name of project (above: `{PROJECT_NAME}`)
- `core_name`: name of GeoJSON object from step above
- (optional) `reg_type`: one of "precise" (default, i.e. manual registration of IMC to postIMC as described above) or "register" (experimental, not robust. Tries to register an aggregated version of the IMC file to the postIMC (set used channels with parameter `IMC_channels_for_aggr` in file config/config.yaml), a GeoJSON as described above is still needed but only to specify the general location of the IMC ablation mask.)
- (optional) `do_register_IMC_location`: bool, default=True, should automatic IMC to postIMC be performed? This setting means that the manually drawn IMC location mask doesn't need to be precise.
- (optional) `register_IMC_location_with_pycpd`: bool, default=True, in case `do_register_IMC_location` is True, should an additional registration step using CPD be performed. Might or might not improve precision.
- `IMS_pixel_size`: size of IMS pixel in micrometer (stepsize)
- `IMS_shrink_factor`: ratio of IMS stepsize to pixelsize (e.g. if stepsize is 30 and pixelsize (ablation area) is 15 then the ratio is 15/30=0.5). Set to 1 if stepsize is equal to pixelsize.
- `IMC_pixel_size`: size of IMC pixel in micrometer
- `microscopy_pixel_size`: size of microscopy pixels in micrometer
- `coords_filename`: name of `-coords.h5` file as obtained from step `Manually using napari-imsmicrolink` or empty. If present the file should be placed in `results/{PROJECT_NAME}/data/IMS`
- `imzml_filename`: name of `.imzML` file (in directory `results/data/{PROJECT_NAME}/IMS` ), `.ibd` file with same base name has to be present in the same directory
- (optional) `IMS_rotation_angle`: int, default=0, one of 0,90,180,270. Initial IMS rotation angle relative to postIMS microscopy image
- (optional) `IMS_to_postIMS_n_splits`: int, default=19, should be a whole number between 3 and 19, number of splits for thresholding to find IMS ablation marks
- (optional) `IMS_to_postIMS_init_gridsearch`: int, default=3, should be a whole number between 0 and 3, number of rounds of gridsearch for IMS ablation mark to IMS pixel initial registration
- (optional) `within_IMC_fine_registration`: bool, default=True, should an additional registration step IMS to postIMS be done only taking into account IMS pixels within the location of IMC.
- (optional) `min_index_length`: int, default=10, minimum number of neighbors to take into account for matching IMS ablation marks with IMS pixels.
- (optional) `max_index_length`: int, default=30, maximum number of neighbors to take into account for matching IMS ablation marks with IMS pixels.

## Register preIMC to preIMS (case 3. only)

For case 3. additionally the regions that should be independently registered have to be selected.  
In QuPath: 
- open preIMC microscopy image
- draw quadrangles around regions to be registered independently. Should contain at least one full core.
- name each quadrangle by right click -> `Annotations` -> `Set properties` -> `Name`. Keep this name in mind since the corresponding core on the preIMS microscopy image has to have the same name. 
- Do this for all cores where the registration didn't work properly. 
- Then go to `File` -> `Export objects as GeoJSON`, **Unselect** `Export as FeatureCollection`, then confirm.  Save as `results/{PROJECT_NAME}/data/preIMC_location_combined/{PROJECT_NAME}_reg_mask_on_preIMC.geojson`

- Do the same for the preIMS microscopy image
- Make sure the names of the corresponding regions are the same
- Save as `results/{PROJECT_NAME}/data/preIMS_location_combined/{PROJECT_NAME}_reg_mask_on_preIMS.geojson`


## Run workflow

To run the workflow:
```
snakemake --use-conda --use-singularity -c 1
```

If there are links to files outside of the project directory adding the flag ` --singularity-args "--bind $HOME"` might be needed:
```
snakemake --use-conda --use-singularity -c 1  --singularity-args "--bind $HOME"
```


## Check output

### Automatic linking of IMS pixels to postIMS

For each sample check output of `results/{PROJECT_NAME}/data/registration_metric/{SAMPLE_NAME}_IMS_to_postIMS_reg_metrics_auto.ome.tiff` (e.g. using QuPath). Blue dots should be in the middle of the IMS pixels. If dots are too far off manual registration is required (See [above](#manually-using-napari-imsmicrolink)).

### Automatic linking of IMC location to postIMC

For each sample check output of `logs/IMC_location_from_postIMC/{PROJECT_NAME}_{CORE_NAME}.ome.tiff`

### postIMC to postIMS registration

The file `results/{PROJECT_NAME}/data/registration_metric/{PROJECT_NAME}_reg_metrics_combined.csv` contains some metrics regarding the precision of the registration. Columns starting with `euclidean_distance_centroids_` show the difference in centroid position between the tissue samples in micrometer.
Additionally the columns starting with `dice_score_` show the dice coefficient between the tissue samples (values should be close to 1). Since those comparisons rely on automatic detection of the tissue the calculated values should be considered to be more of an lower bound of the precision instead of the actual true measure. 
To get a better view of the precision visual inspection needs to be done. For that load images with napari (e.g. run `conda run -n napari_imsmicrolink napari`) or QuPath. The relevant images are: `results/{PROJECT_NAME}/data/postIMS/{PROJECT_NAME}_postIMS.ome.tiff`, `results/{PROJECT_NAME}/data/preIMS/{PROJECT_NAME}_preIMS.ome.tiff`, `results/{PROJECT_NAME}/data/preIMC/{PROJECT_NAME}_preIMC.ome.tiff` and `results/{PROJECT_NAME}/data/postIMC/{PROJECT_NAME}_postIMC.ome.tiff`.


