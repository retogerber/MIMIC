if (interactive()){
  source(file.path("imc_to_ims_workflow", "workflow", "scripts", "combine_IMS_utils.R"))
  source(file.path("imc_to_ims_workflow", "workflow", "scripts", "logging_utils.R"))
} else{
  source(file.path("workflow", "scripts", "combine_IMS_utils.R"))
  source(file.path("workflow", "scripts", "logging_utils.R"))
}

# prepare
set_threads(snakemake)
log_snakemake_variables(snakemake)

log_message("Load packages")
suppressPackageStartupMessages({
  library(Cardinal)
  library(rhdf5)
  })

# input
filename_imzml <- snakemake@input[["imzml"]]
filename_ibd <- snakemake@input[["ibd"]]
filename_peaklist <- snakemake@input[["peaklist"]]
# output
output_hdf5 <- snakemake@output[["peaks"]]

log_message("Check input")
stopifnot(dirname(filename_imzml) == dirname(filename_ibd))
stopifnot(grepl("\\.imzML$",basename(filename_imzml)))
stopifnot(grepl("\\.ibd$",basename(filename_ibd)))

log_message("Setup parrallelization")
setCardinalBPPARAM(BiocParallel::MulticoreParam(workers = snakemake@threads) )
setCardinalNumBlocks(n=ifelse(snakemake@threads>10,as.integer(snakemake@threads)*10,200))
setCardinalVerbose(verbose=TRUE)

log_message("Read peaklist")
ref_mzvals_df <- read.csv(filename_peaklist,header=TRUE)
ref_mzvals_df <- ref_mzvals_df[order(ref_mzvals_df$mz),]
ref_mzvals <- ref_mzvals_df[["mz"]]
internal_standard_mzval <- ref_mzvals_df[["mz"]][ref_mzvals_df[["is_internal_standard"]]==1]

log_message("Read imzml")
msi <- readImzML(
  name=sub("\\.imzML$","",basename(filename_imzml)),
  resolution=0.01, units="mz",
  folder=dirname(filename_imzml),
  mass.range = c(min(ref_mzvals)-1,max(ref_mzvals)+1))

log2_na <- function(x, xmin=0){
  xnew <- log2(round(x,2))
  xnew <- replace(xnew, is.infinite(xnew), xmin)
  xnew
}

log_message("Setup process imzml")
msi_processed <- msi |>
  normalize("tic") |>
  mzAlign(ref = ref_mzvals, tolerance = 0.02, units = "mz") |>
  peakBin(ref = ref_mzvals, type = "height", tolerance = 0.02, units = "mz") |>
  process(fun=log2_na, xmin=NA, label="transform", delay=TRUE)

log_message("Run process imzml")
msi_processed <- process(msi_processed)
log_message("Done")

log_message("Write output")
h5createFile(output_hdf5)
tmpmat <- as.matrix(iData(msi_processed))
h5write(tmpmat, output_hdf5,"peaks")
h5write(as.matrix(fData(msi_processed)), output_hdf5,"mzs")
h5write(as.matrix(coord(msi_processed)), output_hdf5,"coord")

log_message("Finished")