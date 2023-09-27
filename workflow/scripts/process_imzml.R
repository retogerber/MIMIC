# logging
stdlog <- file(snakemake@log[["stdout"]], open="wt")
sink(stdlog, type = "output")
stderr <- file(snakemake@log[["stderr"]], open="wt")
sink(stderr, type = "message")

suppressPackageStartupMessages({
  library(Cardinal)
  library(rhdf5)
  })


# filename_imzml <- here::here("data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/cirrhosis_TMA/data/IMS/cirrhosis-tma-4_aaxl_01022022-total ion count.imzML")
# filename_ibd <- here::here("data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/cirrhosis_TMA/data/IMS/cirrhosis-tma-4_aaxl_01022022-total ion count.ibd")
#filename_peaklist <- "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/cirrhosis_TMA/data/IMS/cirrhosis_TMA_peaklist.csv"
filename_imzml <- snakemake@input[["imzml"]]
filename_ibd <- snakemake@input[["ibd"]]
filename_peaklist <- snakemake@input[["peaklist"]]

dirname(filename_imzml)
basename(filename_imzml)

stopifnot(dirname(filename_imzml) == dirname(filename_ibd))
stopifnot(grepl("\\.imzML$",basename(filename_imzml)))
stopifnot(grepl("\\.ibd$",basename(filename_ibd)))

setCardinalBPPARAM(BiocParallel::MulticoreParam(workers = snakemake@threads) )
setCardinalNumBlocks(n=ifelse(snakemake@threads>10,as.integer(snakemake@threads)*10,200))
setCardinalVerbose(verbose=TRUE)

# read peaklist
ref_mzvals_df <- read.csv(filename_peaklist,header=TRUE)
ref_mzvals_df <- ref_mzvals_df[order(ref_mzvals_df$mz),]
ref_mzvals <- ref_mzvals_df[["mz"]]
internal_standard_mzval <- ref_mzvals_df[["mz"]][ref_mzvals_df[["is_internal_standard"]]==1]

# read imzml
msi <- readImzML(
  name=sub("\\.imzML$","",basename(filename_imzml)),
  resolution=0.01, units="mz",
  folder=dirname(filename_imzml),
  mass.range = c(min(ref_mzvals)-1,max(ref_mzvals)+1))


obs_mzvals <- as.data.frame(fData(msi))[["mz"]]
ref_feature <- features(msi)[which.min(abs(obs_mzvals-internal_standard_mzval))]


msi_processed <- msi |>
  normalize(method="tic") |>
  #mzAlign(ref=ref_mzvals,tolerance=0.02, units="mz") |>
  peakBin(ref=ref_mzvals, type="height",tolerance=0.02, units="mz")


# msi_processed <- msi |> 
#   smoothSignal(method="sgolay") |>
#   normalize(method="reference", feature=ref_feature) |>
#   reduceBaseline() |> 
#   peakBin(ref=ref_mzvals, type="height") 




msi_processed <- process(msi_processed)
  

# write to hdf5
# output_hdf5 <- here::here("data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/cirrhosis_TMA/data/IMS/cirrhosis_TMA_peaks.h5")
output_hdf5 <- snakemake@output[["peaks"]]

h5createFile(output_hdf5)
tmpmat <- as.matrix(iData(msi_processed))
h5write(tmpmat, output_hdf5,"peaks")
h5write(as.matrix(fData(msi_processed)), output_hdf5,"mzs")
h5write(as.matrix(coord(msi_processed)), output_hdf5,"coord")
