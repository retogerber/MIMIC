
# Software requirements

- conda and snakemake [Install instructions here](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html#installation)
- singularity/apptainer [Install instructions here](http://apptainer.org/docs/admin/1.1/installation.html#installing-apptainer)
- QuPath [Install instructions here](https://github.com/qupath/qupath/wiki/Installing-QuPath)

# Workflow

## raw data / results structure

For each project (slide):
- create project result directory:
```
bash workflow/scripts/setup_project_results_directory.sh -p PROJECT_NAME
```
this will create the following results directory:

results/
├── PROJECT_NAME
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
│   │   └── preIMS_location_combined
│   └── registrations
│       ├── IMC_to_IMS
│       ├── IMC_to_preIMC
│       ├── postIMC_to_postIMS
│       └── preIMC_to_preIMS

Three example projects are available in the results directory: 
1. test_combined
    normal use case, one imzML file per project, registration of preIMC to preIMS successful in one step (whole slide registration)
2. test_split_ims
    multiple imzML files per project, registration of preIMC to preIMS successful in one step (whole slide registration)
3. test_split_pre
    one imzML file per project,  registration of preIMC to preIMS **no** successful in one step, register subsets of slides to each other

use cases 2. and 3. can also be combined.


## Setup

The following steps to setup your own projects are required: (as guidance you can also use the directory results/test_combined)

- .imzml and .ibd files in results/PROJECT_NAME/data/IMS
- microscopy images in their respective folders (e.g. postIMS into results/PROJECT_NAME/data/postIMS)
    the expected names are: PROJECT_NAME_FILETYPE.ome.tiff , e.g. for project name test_combined: test_combined_preIMS.ome.tiff
- IMC images (as single, multi-channel tiff files) to results/PROJECT_NAME/data/IMC
- IMC segmentation mask image (as single-channel tiff files) to results/PROJECT_NAME/data/IMC_mask
    IMC images and IMC segmentation mask should have the same name
- IMC summary panels (as csv files) to results/PROJECT_NAME/data/IMC_summary_panel, expected name is IMC_NAME_summary.csv (replace IMC_NAME with the corresponding IMC file basename (no extension), e.g. Cirrhosis-TMA-5_New_Detector_001_summary.csv)


## Linking IMS pixels to postIMS (napari-imsmicrolink)

First create environment:
```
conda env create -f workflow/env/napari_imsmicrolink.yaml
```
then run 
```
conda run -n napari_imsmicrolink napari
```
This should open napari. There go to `Plugins` -> `IMS MicroLink` which should open napari-imsmicrolink on the righthand side. Register IMS to postIMS or postIMS to IMS (preferred) by choosing the target modality in the napari plugin
    - save the output to results/PROJECT_NAME/data/IMS
    - if output generates .ome.tiff (if register postIMS to IMS) move the created .ome.tiff file (which should be the postIMS microscopy image registered to IMS) to the directory results/PROJECT_NAME/data/postIMS, replace the existing file


## Register IMC to postIMC (QuPath)


In QuPath open postIMC microscopy image, draw quadrangles around ablation marks from IMC, do this as precicely as possible, name each quadrangle by right click -> `Annotations` -> `Set properties` -> `Name`. Keep this name in mind since you will need to fill it into file `config/sample_metadata.csv` (see below). Do this for all IMC ablation marks. Then go to `File` -> `Export objects as GeoJSON`, **Unselect** `Export as FeatureCollection`, then confirm.  Save as results/PROJECT_NAME/data/IMC_location/PROJECT_NAME_IMC_mask_on_postIMC.geojson .



## Add metadata

Adapt file config/sample_metadata.csv:
- sample_name: name of sample (i.e. name of IMC image without file ending)
- project_name: name of project (above: PROJECT_NAME)
- core_name: name of GeoJSON object from step above
- reg_type: one of "precise" (default, i.e. manual registration of IMC to postIMC as described above) or "register" (experimental, not robust. Tries to register an aggregated version of the IMC file to the postIMC (set used used channels with parameter `IMC_channels_for_aggr` in file config/config.yaml), a GeoJSON as described above is still needed but only to specify the general location of the IMC ablation mask)
- IMS_pixel_size: size of IMS pixel in micrometer (stepsize)
- IMS_shrink_factor: ratio of IMS stepsize to pixelsize (e.g. if stepsize is 30 and pixelsize (ablation area) is 15 then the ratio is 15/30=0.5)
- IMC_pixel_size: size of IMC pixel in micrometer
- microscopy_pixel_size: size of microscopy pixels in micrometer
- coords_filename: name of `-coords.h5` file (as obtained from step `Linking IMS pixels to postIMS`), the file should be placed in `results/PROJECT_NAME/data/IMS`
- imzml_filename: name of `.imzML` file (in directory `results/data/PROJECT_NAME/IMS` ), `.ibd` file with same base name has to be present in the same directory





```
snakemake --use-conda --use-singularity -c 1 --report
```
