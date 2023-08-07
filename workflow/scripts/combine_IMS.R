# logging
stdlog <- file(snakemake@log[["stdout"]], open="wt")
sink(stdlog, type = "output")
stderr <- file(snakemake@log[["stderr"]], open="wt")
sink(stderr, type = "message")

# n_worker <- snakemake@threads
# RhpcBLASctl::blas_set_num_threads(n_worker)
source(file.path("workflow", "scripts", "combine_IMS_utils.R"))

# imscoords_filenames <- here::here("data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/", project_name, "/data/IMS/", paste0(project_name, "-IMSML-coords.h5"))
imscoords_filename <- "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/IMS/postIMS_to_IMS_test_split_ims_2-IMSML-coords.h5"
# imscoords_filename <- c("/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/IMS/postIMS_to_IMS_test_combined-IMSML-coords.h5","/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/IMS/postIMS_to_IMS_test_combined-Cirrhosis-TMA-5_New_Detector_002-IMSML-coords.h5")
imscoords_filename <- snakemake@input[["imsml_coords_fp"]]
# imspeaks_filename <- here::here("data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/", project_name, "/data/IMS/", paste0(project_name, "_IMS_peaks.h5"))
imspeaks_filename <- "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/IMS/IMS_test_split_ims_2_peaks.h5"
# imspeaks_filename <- "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/IMS/IMS_test_combined_peaks.h5"
imspeaks_filename <- snakemake@input[["peaks"]]

imzml_name <- gsub("_peaks.h5","",basename(imspeaks_filename))

# project_name <- "test_combined"
project_name <- basename(dirname(dirname(dirname(imspeaks_filename)[1])))

# maldi_step_size <- 30
maldi_step_size <- as.numeric(snakemake@params["IMS_pixelsize"])
# maldi_pixel_size <- 24
maldi_pixel_size <- maldi_step_size*as.numeric(snakemake@params["IMS_shrink_factor"])

# celloverlap_filename <- "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/cell_overlap/test_split_ims_Cirrhosis-TMA-5_New_Detector_002_cell_overlap_IMS.csv"
# celloverlap_filename <- "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/cell_overlap/test_combined_Cirrhosis-TMA-5_New_Detector_001_cell_overlap_IMS.csv"
celloverlap_filename <- snakemake@input[["cell_overlaps"]]
names(celloverlap_filename) <- basename(celloverlap_filename) |>
  gsub(pattern="_cell_overlap_IMS.csv",replacement="") |>
  gsub(pattern=paste0(project_name,"_"),replacement="")


# cellcentroids_filename <- "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/cell_overlap/test_split_ims_Cirrhosis-TMA-5_New_Detector_002_cell_centroids.csv"
# cellcentroids_filename <- "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/cell_overlap/test_combined_Cirrhosis-TMA-5_New_Detector_001_cell_centroids.csv"
cellcentroids_filename <- snakemake@input[["cell_centroids"]]
names(cellcentroids_filename) <- basename(cellcentroids_filename) |>
  gsub(pattern="_cell_centroids.csv",replacement="") |>
  gsub(pattern=paste0(project_name,"_"),replacement="")


imcims_df <- create_imsc(imspeaks_filename, imscoords_filename, celloverlap_filename, cellcentroids_filename,  maldi_pixelsize = maldi_step_size,
  additional_colData = c("sample_id"), complete_maldi = TRUE
)

imcims_df[["maldi_pixel_size"]] <- maldi_pixel_size
imcims_df[["maldi_step_size"]] <- maldi_step_size
imcims_df[["project_name"]] <- project_name
imcims_df[["imzml_name"]] <- imzml_name


data.table::fwrite(imcims_df, snakemake@output[["combined_data"]])
