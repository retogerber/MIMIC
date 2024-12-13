
set_threads <- function(snakemake){
    n_worker <- snakemake@threads
    RhpcBLASctl::blas_set_num_threads(n_worker)
}

log_message <- function(x, type = "INFO"){
    message(paste0(format(Sys.time(), format="%y-%m-%d %H:%M:%S"), " [",type,"] ", paste(x, collapse = " ")))
}

log_snakemake_variables <- function(snakemake){
    # logging
    stdlog <- file(snakemake@log[["stdout"]], open="wt")
    sink(stdlog, type = "output")
    sink(stdlog, type = "message")

    log_message("Snakemake variables:")
    log_message(paste0("Threads: ", snakemake@threads))
    log_message("Params:")
    params <- snakemake@params
    for (nam in names(params)){
      if (nam != ""){
        log_message(paste0("\t",nam,": ", paste(params[[nam]], collapse = " ")))
      }
    }
    log_message("Inputs")
    input <- snakemake@input
    for (nam in names(input)){
      if (nam != ""){
        log_message(paste0("\t",nam,": ", paste(input[[nam]], collapse = " ")))
      }
    }
    log_message("Output")
    output <- snakemake@output
    for (nam in names(output)){
      if (nam != ""){
        log_message(paste0("\t",nam,": ", paste(output[[nam]], collapse = " ")))
      }
    }
}