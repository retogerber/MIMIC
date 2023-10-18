# logging
stdlog <- file(snakemake@log[["stdout"]], open="wt")
sink(stdlog, type = "output")
sink(stdlog, type = "message")

# n_worker <- snakemake@threads
# RhpcBLASctl::blas_set_num_threads(n_worker)
source(file.path("workflow", "scripts", "combine_IMS_utils.R"))

# imspeaks_filename <- "/home/retger/Downloads/mSpleen_test/mSpleen_pngasesa_05132023_peaks.h5"
# imspeaks_filename <- here::here("data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/", project_name, "/data/IMS/", paste0(project_name, "_IMS_peaks.h5"))
# imspeaks_filename <- "/home/retger/Downloads/test_IMS_coords_comb/NASH_HCC_TMA_IMS_peaks.h5"
# imspeaks_filename <- "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/IMS/IMS_test_split_ims_2_peaks.h5"
# imspeaks_filename <- "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/IMS/IMS_test_combined_peaks.h5"
imspeaks_filename <- snakemake@input[["peaks"]]

imzml_name <- gsub("_peaks.h5","",basename(imspeaks_filename))


# project_name <- "test_split_ims"
# project_name <- "mSpleen"
# project_name <- "NASH_HCC_TMA"
project_name <- basename(dirname(dirname(dirname(imspeaks_filename)[1])))

# imscoords_filename <- "/home/retger/Downloads/mSpleen_test/mSpleen-IMSML-coords.h5"
# imscoords_filename <- here::here("data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/", project_name, "/data/IMS/", paste0(project_name, "_IMSML-coords.h5"))
# imscoords_filename <- "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/IMS/postIMS_to_IMS_test_split_ims-Cirrhosis-TMA-5_New_Detector_002-IMSML-coords.h5"

# imscoords_filename <- sapply(1:45, function(i) sprintf("/home/retger/Downloads/test_IMS_coords_comb/postIMS_to_IMS_NASH_HCC_TMA-NASH_HCC_TMA-2_0%02d-IMSML-coords.h5",i))[-39]
# imscoords_filename[11] <- "/home/retger/Downloads/test_images_ims_to_imc_workflow/NASH_HCC_TMA_011-IMSML-coords.h5"
# imscoords_filename <- sapply(1:4, function(i) sprintf("/home/retger/Downloads/test_IMS_coords_comb/postIMS_to_IMS_NASH_HCC_TMA-NASH_HCC_TMA-2_0%02d-IMSML-coords.h5",i))
imscoords_filename <- unique(snakemake@input[["imsml_coords_fp"]])
names(imscoords_filename) <- basename(imscoords_filename) |>
  gsub(pattern="(-|_)IMSML-coords.h5",replacement="") |>
  gsub(pattern="postIMS_to_IMS_",replacement="") |>
  gsub(pattern=paste0("^",project_name,"(-|_)"),replacement="")
# imscoords_filename <- "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/IMS/postIMS_to_IMS_test_split_ims_2-IMSML-coords.h5"
# imscoords_filename <- c("/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/IMS/postIMS_to_IMS_test_combined-IMSML-coords.h5","/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/IMS/postIMS_to_IMS_test_combined-Cirrhosis-TMA-5_New_Detector_002-IMSML-coords.h5")


# maldi_step_size <- 10
# maldi_step_size <- 30
# maldi_step_size <- 20
maldi_step_size <- as.numeric(snakemake@params["IMS_pixelsize"])

# maldi_pixel_size <- 8
# maldi_pixel_size <- 24
# maldi_pixel_size <- 16 
maldi_pixel_size <- maldi_step_size*as.numeric(snakemake@params["IMS_shrink_factor"])

# celloverlap_filename <- sapply(1:3, function(i) sprintf("/home/retger/Downloads/mSpleen_test/mSpleen_mSpleen_06092023_0%02d_cell_overlap_IMS.csv",i))
# celloverlap_filename <- "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/cell_overlap/test_split_ims_Cirrhosis-TMA-5_New_Detector_002_cell_overlap_IMS.csv"
# celloverlap_filename <- "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/cell_overlap/test_combined_Cirrhosis-TMA-5_New_Detector_001_cell_overlap_IMS.csv"
# celloverlap_filename <- sapply(1:45, function(i) sprintf("/home/retger/Downloads/test_IMS_coords_comb/NASH_HCC_TMA_NASH_HCC_TMA-2_0%02d_cell_overlap_IMS.csv",i))[-39]
celloverlap_filename <- snakemake@input[["cell_overlaps"]]
names(celloverlap_filename) <- basename(celloverlap_filename) |>
  gsub(pattern="_cell_overlap_IMS.csv",replacement="") |>
  gsub(pattern=paste0(project_name,"_"),replacement="")

# if (length(imscoords_filename)<length(celloverlap_filename)){
#   if (length(imscoords_filename)==0) {
#     imscoords_filename <- rep(imscoords_filename,length(celloverlap_filename))
#     names(imscoords_filename) <- names(celloverlap_filename)
#   } else {
#      stop("imscoords files and celloverlap files cannot be matched!")
#   }
# }

# check and replace names of manual imsmicrolink files
if (length(imscoords_filename)>1) {
  is_correct <- names(imscoords_filename) == names(celloverlap_filename)
  sprintf("Number of matching names %s / %s", sum(is_correct), length(names(imscoords_filename)))
  if (sum(is_correct) < length(is_correct)){
    for (i in seq_len(length(is_correct)-sum(is_correct))) {
      print(sprintf("Incorrect name: %s, expected: %s", names(imscoords_filename)[!is_correct][i], names(celloverlap_filename)[!is_correct][i]))
    }
    print("Replacing names")
    names(imscoords_filename)[!is_correct] <- names(celloverlap_filename)[!is_correct]
  }
}

# cellcentroids_filename <- sapply(1:3, function(i) sprintf("/home/retger/Downloads/mSpleen_test/mSpleen_mSpleen_06092023_0%02d_cell_centroids.csv",i))
# cellcentroids_filename <- "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/cell_overlap/test_split_ims_Cirrhosis-TMA-5_New_Detector_002_cell_centroids.csv"
# cellcentroids_filename <- "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/cell_overlap/test_combined_Cirrhosis-TMA-5_New_Detector_001_cell_centroids.csv"
# cellcentroids_filename <- sapply(1:45, function(i) sprintf("/home/retger/Downloads/test_IMS_coords_comb/NASH_HCC_TMA_NASH_HCC_TMA-2_0%02d_cell_centroids.csv",i))[-39]
# cellcentroids_filename <- sapply(1:4, function(i) sprintf("/home/retger/Downloads/test_IMS_coords_comb/NASH_HCC_TMA_NASH_HCC_TMA-2_0%02d_cell_centroids.csv",i))
cellcentroids_filename <- snakemake@input[["cell_centroids"]]
names(cellcentroids_filename) <- basename(cellcentroids_filename) |>
  gsub(pattern="_cell_centroids.csv",replacement="") |>
  gsub(pattern=paste0(project_name,"_"),replacement="")


idx_by_location <- rep(TRUE, length(imscoords_filename))

imcims_df <- create_imsc(
  imspeaks_filename, 
  imscoords_filename, 
  celloverlap_filename, 
  cellcentroids_filename,  
  maldi_pixelsize = maldi_step_size,
  complete_maldi = TRUE,
  idx_by_location = idx_by_location
)

imcims_df[["maldi_pixel_size"]] <- maldi_pixel_size
imcims_df[["maldi_step_size"]] <- maldi_step_size
imcims_df[["project_name"]] <- project_name
imcims_df[["imzml_name"]] <- imzml_name


data.table::fwrite(imcims_df, snakemake@output[["combined_data"]])
