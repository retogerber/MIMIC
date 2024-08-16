if (interactive()){
  source(file.path("imc_to_ims_workflow", "workflow", "scripts", "combine_IMS_utils.R"))
  source(file.path("imc_to_ims_workflow", "workflow", "scripts", "logging_utils.R"))
} else{
  source(file.path("workflow", "scripts", "Overlap","combine_IMS_utils.R"))
  source(file.path("workflow", "scripts", "logging_utils.R"))
}

# prepare
set_threads(snakemake)
log_snakemake_variables(snakemake)

if (interactive()){
  # params
  maldi_step_size <- 30
  maldi_pixel_size <- 24

  # input
  imspeaks_filename <- c("")
  imscoords_filename <- c("")
  celloverlap_filename <- c("")
  cellcentroids_filename <- c("")
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
  log_message(sprintf("Number of matching names %s / %s", sum(is_correct), length(names(imscoords_filename))))
  if (sum(is_correct) < length(is_correct)){
    for (i in seq_len(length(is_correct)-sum(is_correct))) {
      log_message(sprintf("Incorrect name: %s, expected: %s", names(imscoords_filename)[!is_correct][i], names(celloverlap_filename)[!is_correct][i]))
    }
    log_message("Replacing names")
    names(imscoords_filename)[!is_correct] <- names(celloverlap_filename)[!is_correct]
  }
}


idx_by_location <- rep(TRUE, length(imscoords_filename))
log_message("Running create_imsc")
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