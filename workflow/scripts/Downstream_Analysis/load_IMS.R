#' Create IMS peak data frame
#'
#' @param imspeaks_filename filename of h5 file containing peaks of IMS
#' @param imscoords_filename filename of h5 file containing coordinates of IMS
#' @param malid_pixelsize size of IMS pixels, to get physical size
#'
#' @return data.frame
#' @export
#'
#' @examples
#' imspeaks_filename <- file.path("inst", "extdata", "IMS_test_combined_peaks.h5")
#' imscoords_filename <- file.path("inst", "extdata", "postIMS_to_IMS_test_combined-IMSML-coords.h5")
#' pd <- get_peak_data(imspeaks_filename, imscoords_filename, 30)
get_peak_data <- function(imspeaks_filename, imscoords_filename, maldi_pixelsize) {
  if (!file.exists(imspeaks_filename)) {
    stop(call. = FALSE, paste0("file '", imspeaks_filename, "' does not exist"))
  }
  if (!file.exists(imscoords_filename)) {
    stop(call. = FALSE, paste0("file '", imscoords_filename, "' does not exist"))
  }
  pd <- as.data.frame(t(rhdf5::h5read(imspeaks_filename, "peaks"))) # or peak_data
  mz <- rhdf5::h5read(imspeaks_filename, "mzs")
  colnames(pd) <- as.character(signif(mz, 7))
  pd$ims_idx <- seq_len(dim(pd)[1]) - 1
  coords <- rhdf5::h5read(imspeaks_filename, "coord")
  pd$ims_x <- coords[, 1]
  pd$ims_y <- coords[, 2]

  # check if IMS was registered to microscopy
  if ("xy_micro_physical" %in% rhdf5::h5ls(imscoords_filename)$name) {
    tmpimsxy <- rhdf5::h5read(imscoords_filename, "xy_original") |>
      t() |>
      as.data.frame() |>
      dplyr::rename(ims_x = V2, ims_y = V1) |>
      dplyr::mutate(ims_xy = paste0(ims_x, "_", ims_y))

    pdout <- dplyr::filter(pd, paste0(ims_x, "_", ims_y) %in% tmpimsxy$ims_xy)

    imsphy <- rhdf5::h5read(imscoords_filename, "xy_micro_physical") |>
      t() |>
      as.data.frame() |>
      dplyr::rename(ims_x_phy = V2, ims_y_phy = V1)

    imsxy <- cbind(tmpimsxy, imsphy)
    if (dim(pdout)[1] != dim(imsxy)[1]) {
      pdout <- dplyr::filter(pd, paste0(ims_y, "_", ims_x) %in% imsxy$ims_xy)
      pdout$ims_x_phy <- imsxy$ims_y_phy
      pdout$ims_y_phy <- imsxy$ims_x_phy
    } else {
      pdout$ims_x_phy <- imsxy$ims_x_phy
      pdout$ims_y_phy <- imsxy$ims_y_phy
    }
    stopifnot(dim(pdout)[1] == dim(imsxy)[1])

    # if microscopy was registered to IMS
  } else {
    pdout <- pd
    padimsxy <- rhdf5::h5read(imscoords_filename, "xy_padded") |>
      t() |>
      as.data.frame() |>
      dplyr::rename(ims_x = V1, ims_y = V2)

    stopifnot(dim(pdout)[1] == dim(padimsxy)[1])
    if (cor(padimsxy$ims_x, pdout$ims_x) < -0.99 &
      cor(padimsxy$ims_y, pdout$ims_y) < -0.99) {
      padimsxy <- dplyr::rename(padimsxy, ims_y = ims_x, ims_x = ims_y)
    }
    pdout$ims_x_phy <- padimsxy$ims_x * maldi_pixelsize
    pdout$ims_y_phy <- padimsxy$ims_y * maldi_pixelsize
  }
  pdout$ims_idx <- seq_len(dim(pdout)[1]) - 1
  pdout
}

#' Combine IMC and IMS data
#'
#' @param spe SpatialExperiment containing IMC data. Or filename of SpatialExperiment object.
#' @param imspeaks_filename filename of h5 file containing peaks of IMS
#' @param imscoords_filename filename of h5 file containing coordinates of IMS
#' @param celloverlap_filename filename of csv file containing IMC cells overlap with IMS pixels
#' @param cellcentroids_filenames filename of csv file containing cell centroids
#' @param malid_pixelsize size of IMS pixels
#' @param spe_cellcolname column name of cell identifier in spe
#' @param sample_id_colname column name of sample identifier in spe
#' @param additional_colData character vector, column names of colData(spe) to include
#' @param complete_maldi logical, should all IMS pixels be returned? default is to only return pixels containing cells
#'
#' @return data.frame
#' @export
#'
#' @examples
#' spe_filename <- file.path("inst", "extdata", "SPE_raw_Cirrhosis-TMA-5_New_Detector_001.rds")
#' imspeaks_filename <- file.path("inst", "extdata", "IMS_test_combined_peaks.h5")
#' imscoords_filename <- file.path("inst", "extdata", "postIMS_to_IMS_test_combined-IMSML-coords.h5")
#' celloverlap_filename <- here::here("inst", "extdata", "test_combined_Cirrhosis-TMA-5_New_Detector_001_cell_overlap_IMS.csv")
#' cellcentroids_filename <- here::here("inst", "extdata", "test_combined_Cirrhosis-TMA-5_New_Detector_001_cell_centroids.csv")
#' imcims_df <- create_imsc(spe_filename, imspeaks_filename, imscoords_filename, celloverlap_filename, cellcentroids_filename, 30)
create_imsc <- function(spe, imspeaks_filename, imscoords_filename,
                        celloverlap_filename, cellcentroids_filename,
                        maldi_pixelsize,
                        spe_cellcolname = "ObjectNumber",
                        sample_id_colname = "sample_id",
                        additional_colData = "sample_id",
                        complete_maldi = FALSE) {
  force(maldi_pixelsize)
  pd <- get_peak_data(imspeaks_filename, imscoords_filename, maldi_pixelsize)
  if (!is(spe, "SpatialExperiment")) {
    if (!file.exists(spe)) {
      stop(call. = FALSE, paste0("file '", spe, "' does not exist"))
    }
    spe <- readRDS(spe)
  }
  if (any(additional_colData %in% c("cell_idx", "ims_idx"))) {
    stop(call. = FALSE, "values of 'additional_colData' are not
                         allowed to include 'cell_idx' or 'ims_idx'")
  }

  if (length(spe[[spe_cellcolname]]) != length(unique(paste0(spe[[spe_cellcolname]], spe[[sample_id_colname]])))) {
    stop(call. = FALSE, paste0(
      "cell ids specified with 'spe_cellcolname=",
      spe_cellcolname, "' are not unique"
    ))
  }

  if (!all(file.exists(celloverlap_filename))) {
    stop(call. = FALSE, paste0("file '", celloverlap_filename, "' does not exist"))
  }
  if (!all(file.exists(cellcentroids_filename))) {
    stop(call. = FALSE, paste0("file '", cellcentroids_filename, "' does not exist"))
  }
  stopifnot(length(celloverlap_filename) == length(cellcentroids_filename))
  if (length(celloverlap_filename) == 1) {
    celldf <- as.data.frame(data.table::fread(celloverlap_filename))
    spe_sample_id <- unique(spe[[sample_id_colname]])
    if (length(spe_sample_id) > 1) {
      stop("More than one sample detected in 'spe', unambiguous matching not guaranteed!")
    }
    celldf[[sample_id_colname]] <- spe_sample_id
  } else {
    stopifnot(!is.null(names(celloverlap_filename)))
    stopifnot(length(unique(names(celloverlap_filename))) == length(names(celloverlap_filename)))
    stopifnot(length(unique(celloverlap_filename)) == length(celloverlap_filename))

    stopifnot(!is.null(names(cellcentroids_filename)))
    stopifnot(length(unique(names(cellcentroids_filename))) == length(names(cellcentroids_filename)))
    stopifnot(length(unique(cellcentroids_filename)) == length(cellcentroids_filename))


    celldf_ls <- lapply(seq_along(celloverlap_filename), function(i) {
      tmpcelldf <- data.table::fread(celloverlap_filename[i])
      tmpcelldf$sample_id <- names(celloverlap_filename)[i]

      tmpcentdf <- data.table::fread(cellcentroids_filename[i])
      data.table::setnames(tmpcentdf, "x", "Pos_X_on_IMS")
      data.table::setnames(tmpcentdf, "y", "Pos_Y_on_IMS")

      tmpcelldf <- tmpcelldf[tmpcentdf, on = .(cell_idx), nomatch = NULL]
      as.data.frame(tmpcelldf)
    })
    celldf <- do.call(rbind, celldf_ls)
  }

  cold <- as.data.frame(colData(spe)[, additional_colData, drop = FALSE])
  # combine data
  speco <- SpatialExperiment::spatialCoords(spe) |>
    as.data.frame() |>
    dplyr::mutate(cell_idx = spe[[spe_cellcolname]]) |>
    cbind(cold) |>
    dplyr::inner_join(celldf, by = c("cell_idx", sample_id_colname), multiple = "all")

  if (complete_maldi) {
    subdf <- pd
    pd_sub <- subdf
    sp::coordinates(subdf) <- c("ims_x", "ims_y")
    sp::gridded(subdf) <- TRUE
    dst <- 3
    nl2 <- BiocNeighbors::findNeighbors(sp::coordinates(subdf), dst)
    wm <- nl2$index
    class(wm) <- c("nb", "list")
    # wm <- spdep::dnearneigh(sp::coordinates(subdf), 0, dst, row.names = paste0(subdf$ims_idx),k=50)
    dj <- spdep::n.comp.nb(wm)
    pd_sub$region_id <- dj$comp.id

    reg <- unique(pd_sub$region_id[pd_sub$ims_idx %in% celldf$ims_idx])
    if (length(celloverlap_filename) == 1 & length(reg) > 1) {
      warning("Multiple regions of IMS associated with single IMC found!")
    }
    combined_df <- speco |>
      dplyr::right_join(pd_sub[pd_sub$region_id %in% reg, ], by = "ims_idx")
    combined_df_split <- base::split(combined_df, combined_df[["region_id"]])
  } else {
    combined_df <- speco |>
      dplyr::inner_join(pd, by = "ims_idx")
    combined_df_split <- base::split(combined_df, combined_df[[sample_id_colname]])
  }

  # test and correct for switched axis
  has_flipped_axis <- sapply(combined_df_split, has_flipped_axis_ims)
  if (any(has_flipped_axis)) {
    message("Axis of IMS are flipped relative to IMC. Reordering IMS axis.")
    combined_df_split <- lapply(combined_df_split, switch_ims_axis, message = FALSE)
    combined_df <- do.call(rbind, combined_df_split)
  }

  # check if cell positions and corresponding overlaping IMS pixels make sense
  large_difference_of_positions <- combined_df |>
    dplyr::filter(!is.na(Pos_X)) |>
    dplyr::group_by(ims_idx, !!dplyr::sym(sample_id_colname)) |>
    dplyr::summarise(
      .groups = "keep",
      Pos_X_max = max(Pos_X),
      Pos_X_min = min(Pos_X),
      Pos_Y_max = max(Pos_Y),
      Pos_Y_min = min(Pos_Y)
    ) |>
    dplyr::mutate(
      range_x = Pos_X_max - Pos_X_min,
      range_y = Pos_Y_max - Pos_Y_min,
      large_difference_of_positions = range_x > 4 * maldi_pixelsize |
        range_y > 4 * maldi_pixelsize
    ) |>
    dplyr::pull(large_difference_of_positions)

  if (any(large_difference_of_positions)) {
    warning(call. = FALSE, "The difference in cell positions per pixel is higher than expected. Are you using the correct cell mask?")
  }

  combined_df
}

#' Test for association between spatial association of IMC and IMS data
#'
#' @param x output of `create_imsc`
#' @param type string, one of 'cor' (correlation) and 'dist'
#' @param threshold float, threshold for correlation (if type='cor')
#' @param IMS_pixel_size integer, size of IMS pixels for distance thresholds (if type='dist')
#'
#' @return logical
#'
#' @examples
#' spe_filename <- file.path("inst", "extdata", "SPE_raw_Cirrhosis-TMA-5_New_Detector_001.rds")
#' imspeaks_filename <- file.path("inst", "extdata", "IMS_test_combined_peaks.h5")
#' imscoords_filename <- file.path("inst", "extdata", "postIMS_to_IMS_test_combined-IMSML-coords.h5")
#' celloverlap_filename <- here::here("inst", "extdata", "test_combined_Cirrhosis-TMA-5_New_Detector_001_cell_overlap_IMS.csv")
#' cellcentroids_filename <- here::here("inst", "extdata", "test_combined_Cirrhosis-TMA-5_New_Detector_001_cell_centroids.csv")
#' imcims_df <- create_imsc(spe_filename, imspeaks_filename, imscoords_filename, celloverlap_filename, cellcentroids_filename, 30)
#' is_associated_ims_cell_positions(imcims_df)
#' is_associated_ims_cell_positions(imcims_df, type = "dist", IMS_pixel_size = 30)
is_associated_ims_cell_positions <- function(x, type = c("cor", "dist"), threshold = 0.9,
                                             IMS_pixel_size = NULL, ims_cols = c("ims_x", "ims_y"), cell_cols = c("Pos_X", "Pos_Y")) {
  type <- match.arg(type)
  if (type == "dist") {
    stopifnot(!is.null(IMS_pixel_size))
    stopifnot(IMS_pixel_size > 0)
  }
  stopifnot(length(ims_cols) == 2)
  stopifnot(length(cell_cols) == 2)
  x <- dplyr::filter(x, !sapply(x[[cell_cols[1]]], is.na))
  if (type == "cor") {
    cors <- sapply(seq_along(ims_cols), function(i) {
      cor(x[[ims_cols[i]]], x[[cell_cols[i]]])
    })
    if (any(cors < 0.5)) {
      cors <- sapply(seq_along(ims_cols), function(i) {
        cor(-x[[ims_cols[i]]] + max(x[[ims_cols[i]]]), x[[cell_cols[i]]])
      })
    }
    mean_cor <- mean(cors)
    return(mean_cor >= threshold)
  } else {
    large_difference_of_positions <- x |>
      dplyr::filter(!is.na(!!dplyr::sym(cell_cols[1]))) |>
      dplyr::group_by(ims_idx) |>
      dplyr::summarise(
        .groups = "keep",
        Pos_X_max = max(!!dplyr::sym(cell_cols[1])),
        Pos_X_min = min(!!dplyr::sym(cell_cols[1])),
        Pos_Y_max = max(!!dplyr::sym(cell_cols[2])),
        Pos_Y_min = min(!!dplyr::sym(cell_cols[2]))
      ) |>
      dplyr::mutate(
        d = sqrt((Pos_X_max - Pos_X_min)**2 + (Pos_Y_max - Pos_Y_min)**2),
        large_difference_of_positions = d > 4 * IMS_pixel_size
      ) |>
      dplyr::pull(large_difference_of_positions)
    return(!any(large_difference_of_positions))
  }
}

#' Check if axis of IMS are switched relative to IMC
#'
#' @param x output of `create_imsc`
#' @param type string, one of 'cor' (correlation) and 'dist'
#' @param threshold float, threshold for correlation (if type='cor')
#' @param IMS_pixel_size integer, size of IMS pixels for distance thresholds (if type='dist')
#'
#' @return logical
#'
#' @examples
#' spe_filename <- file.path("inst", "extdata", "SPE_raw_Cirrhosis-TMA-5_New_Detector_001.rds")
#' imspeaks_filename <- file.path("inst", "extdata", "IMS_test_combined_peaks.h5")
#' imscoords_filename <- file.path("inst", "extdata", "postIMS_to_IMS_test_combined-IMSML-coords.h5")
#' celloverlap_filename <- here::here("inst", "extdata", "test_combined_Cirrhosis-TMA-5_New_Detector_001_cell_overlap_IMS.csv")
#' cellcentroids_filename <- here::here("inst", "extdata", "test_combined_Cirrhosis-TMA-5_New_Detector_001_cell_centroids.csv")
#' imcims_df <- create_imsc(spe_filename, imspeaks_filename, imscoords_filename, celloverlap_filename, cellcentroids_filename, 30)
#' has_flipped_axis_ims(imcims_df)
#' has_flipped_axis_ims(imcims_df, type = "dist", IMS_pixel_size = 30)
has_flipped_axis_ims <- function(x, type = c("cor", "dist"), threshold = 0.9, IMS_pixel_size = NULL, ims_cols = c("ims_x", "ims_y"), cell_cols = c("Pos_X", "Pos_Y")) {
  normal_case <- is_associated_ims_cell_positions(x, type, threshold, IMS_pixel_size, ims_cols, cell_cols)
  flipped_case <- is_associated_ims_cell_positions(x, type, threshold, IMS_pixel_size, rev(ims_cols), cell_cols)
  if (normal_case & !flipped_case) {
    return(FALSE)
  } else if (!normal_case & flipped_case) {
    return(TRUE)
  } else if (normal_case & flipped_case) {
    stop("Both are associated!")
  } else {
    stop("Neither are associated!")
  }
}


#' Switch axis of IMS if axis are switched relative to IMC
#'
#' @param x output of `create_imsc`
#' @param type string, one of 'cor' (correlation) and 'dist'
#' @param threshold float, threshold for correlation (if type='cor')
#' @param IMS_pixel_size integer, size of IMS pixels for distance thresholds (if type='dist')
#'
#' @return data.frame, x
#'
#' @examples
#' spe_filename <- file.path("inst", "extdata", "SPE_raw_Cirrhosis-TMA-5_New_Detector_001.rds")
#' imspeaks_filename <- file.path("inst", "extdata", "IMS_test_combined_peaks.h5")
#' imscoords_filename <- file.path("inst", "extdata", "postIMS_to_IMS_test_combined-IMSML-coords.h5")
#' celloverlap_filename <- here::here("inst", "extdata", "test_combined_Cirrhosis-TMA-5_New_Detector_001_cell_overlap_IMS.csv")
#' cellcentroids_filename <- here::here("inst", "extdata", "test_combined_Cirrhosis-TMA-5_New_Detector_001_cell_centroids.csv")
#' imcims_df <- create_imsc(spe_filename, imspeaks_filename, imscoords_filename, celloverlap_filename, cellcentroids_filename, 30)
#' imcims_df <- switch_ims_axis(imcims_df)
switch_ims_axis <- function(x, type = c("cor", "dist"), threshold = 0.9, IMS_pixel_size = NULL,
                            ims_cols = c("ims_x", "ims_y"), cell_cols = c("Pos_X", "Pos_Y"),
                            message = TRUE) {
  is_flipped <- has_flipped_axis_ims(x, type, threshold, IMS_pixel_size, ims_cols, cell_cols)
  if (is_flipped) {
    if (message) message("Axis of IMS are flipped relative to IMC. Reordering IMS axis.")
    tmpx <- x
    tmpx[[ims_cols[1]]] <- x[[ims_cols[2]]]
    tmpx[[ims_cols[2]]] <- x[[ims_cols[1]]]
    return(tmpx)
  } else {
    return(x)
  }
}
