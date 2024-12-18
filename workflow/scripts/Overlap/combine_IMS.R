if (interactive()){
  source(file.path("workflow", "scripts","Overlap", "combine_IMS_utils.R"))
  source(file.path("workflow", "scripts","utils", "logging_utils.R"))
} else{
  source(file.path("workflow", "scripts", "Overlap","combine_IMS_utils.R"))
  source(file.path("workflow", "scripts", "utils", "logging_utils.R"))
}

# prepare
set_threads(snakemake)
n_worker <- as.integer(snakemake@threads)
log_snakemake_variables(snakemake)

if (interactive()){
  # params
  maldi_step_size <- 30
  maldi_pixel_size <- 24

  # input
  imspeaks_filename <- c("results/test_split_pre/data/IMS/IMS_test_split_pre_peaks.h5")
  imscoords_filename <- c("results/test_split_pre/data/IMS/postIMS_to_IMS_test_split_pre-IMSML-coords.h5","results/test_split_pre/data/IMS/postIMS_to_IMS_test_split_pre-Cirrhosis-TMA-5_New_Detector_002-IMSML-coords.h5")
  celloverlap_filename <- c("results/test_split_pre/data/cell_overlap/test_split_pre_Cirrhosis-TMA-5_New_Detector_001_cell_overlap_IMS.csv","results/test_split_pre/data/cell_overlap/test_split_pre_Cirrhosis-TMA-5_New_Detector_002_cell_overlap_IMS.csv")
  cellcentroids_filename <- c("results/test_split_pre/data/cell_overlap/test_split_pre_Cirrhosis-TMA-5_New_Detector_001_cell_centroids.csv","results/test_split_pre/data/cell_overlap/test_split_pre_Cirrhosis-TMA-5_New_Detector_002_cell_centroids.csv")
} else{
  # params
  maldi_step_size <- as.numeric(snakemake@params["IMS_pixelsize"])
  maldi_pixel_size <- maldi_step_size*as.numeric(snakemake@params["IMS_shrink_factor"])

  # input
  imspeaks_filename <- snakemake@input[["peaks"]]
  imscoords_filename <- unique(snakemake@input[["imsml_coords_fp"]])
  celloverlap_filename <- snakemake@input[["cell_overlaps"]]
  cellcentroids_filename <- snakemake@input[["cell_centroids"]]
  IMC_mean_on_IMS_filename <- snakemake@input[["IMC_mean_on_IMS"]]

  #output
  combined_data_out <- snakemake@output[["combined_data"]]
}


imzml_name <- gsub("_peaks.h5","",basename(imspeaks_filename))
log_message(sprintf("IMZML name: %s", imzml_name))
project_name <- basename(dirname(dirname(dirname(imspeaks_filename)[1])))
log_message(sprintf("Project name: %s", project_name))

names(imscoords_filename) <- basename(imscoords_filename) |>
  gsub(pattern="(-|_)IMSML-coords.h5",replacement="") |>
  gsub(pattern="postIMS_to_IMS_",replacement="") |>
  gsub(pattern=paste0("^",project_name,"(-|_)"),replacement="")
log_message(sprintf("IMS coords names: %s", paste(names(imscoords_filename), collapse=", ")))

names(celloverlap_filename) <- basename(celloverlap_filename) |>
  gsub(pattern="_cell_overlap_IMS.csv",replacement="") |>
  gsub(pattern=paste0(project_name,"_"),replacement="")
log_message(sprintf("Cell overlap names: %s", paste(names(celloverlap_filename), collapse=", ")))

names(cellcentroids_filename) <- basename(cellcentroids_filename) |>
  gsub(pattern="_cell_centroids.csv",replacement="") |>
  gsub(pattern=paste0(project_name,"_"),replacement="")
log_message(sprintf("Cell centroids names: %s", paste(names(cellcentroids_filename), collapse=", ")))

# check and replace names of manual imsmicrolink files
if (length(imscoords_filename)>1) {
  is_correct <- names(imscoords_filename) == names(celloverlap_filename)
  log_message(sprintf("Number of matching names %s / %s", sum(is_correct), length(names(imscoords_filename))))
  if (sum(is_correct) < length(is_correct)){
    for (i in seq_len(length(is_correct)-sum(is_correct))) {
      log_message(sprintf("Incorrect name: %s, expected: %s", names(imscoords_filename)[!is_correct][i], names(celloverlap_filename)[!is_correct][i]))
    }
    log_message("Replacing names")
    names(imscoords_filename)[!is_correct] <- names(celloverlap_filename)[!is_correct]
  }
}

n_regions <- sapply(seq_along(imscoords_filename), function(i){
  df <- rhdf5::h5read(imscoords_filename[i], "xy_original") |>
    t() |>
    as.data.frame() |>
    dplyr::rename(ims_x = V1, ims_y = V2) |>
    dplyr::mutate(ims_xy = paste0(ims_x, "_", ims_y))
  sp::coordinates(df) <- c("ims_x", "ims_y")
  sp::gridded(df) <- TRUE
  dst <- 3
  nl2 <- BiocNeighbors::findNeighbors(sp::coordinates(df), dst)
  wm <- nl2$index
  class(wm) <- c("nb", "list")
  dj <- spdep::n.comp.nb(wm)
  df$region_id <- dj$comp.id
  length(unique(df$region_id))
})
log_message(sprintf("Number of regions: %s", paste(n_regions, collapse=", "))) 
is_manual <- sapply(seq_along(imscoords_filename), function(i) !stringr::str_detect(imscoords_filename[i], paste0("postIMS_to_IMS_",project_name,"-",names(imscoords_filename)[i],"-IMSML-coords.h5")))
log_message(sprintf("Is manual: %s", paste(is_manual, collapse=", ")))

# if automatic:
#  idx_by_location = TRUE
# if manual:
#  if number of regions == 1
#    idx_by_location = TRUE
#  else
#    idx_by_location = FALSE
idx_by_location <- !is_manual | n_regions == 1
log_message(sprintf("Index by location: %s", paste(idx_by_location,collapse=", ")))
log_message("Running create_imsc")
plan(multisession, workers = n_worker)
imcims_df <- create_imsc(
  imspeaks_filename, 
  imscoords_filename, 
  celloverlap_filename, 
  cellcentroids_filename,  
  maldi_pixelsize = maldi_step_size,
  complete_maldi = TRUE,
  idx_by_location = idx_by_location
)
log_message("Finished create_imsc")
log_message("Add metadata")
imcims_df[["maldi_pixel_size"]] <- maldi_pixel_size
imcims_df[["maldi_step_size"]] <- maldi_step_size
imcims_df[["project_name"]] <- project_name
imcims_df[["imzml_name"]] <- imzml_name

names(IMC_mean_on_IMS_filename) <- basename(IMC_mean_on_IMS_filename) |>
  gsub(pattern="_mean_intensity_on_IMS.csv",replacement="") |>
  gsub(pattern=paste0(project_name,"_"),replacement="")
dfls <- lapply(IMC_mean_on_IMS_filename, data.table::fread) 
for (i in seq_along(dfls)){
  dfls[[i]]$sample_id <- names(IMC_mean_on_IMS_filename)[i]
}
tempdt <- Reduce(rbind,dfls) 
imcims_df <- dplyr::left_join(imcims_df, tempdt, by = c("ims_idx" = "label", "sample_id" = "sample_id"))

log_message("Writing combined data")
data.table::fwrite(imcims_df, combined_data_out)

log_message("Finished")