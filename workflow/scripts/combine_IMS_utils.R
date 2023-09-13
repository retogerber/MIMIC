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
get_peak_data <- function(imspeaks_filename, imscoords_filename, maldi_pixelsize, idx_by_location=FALSE) {
  if (length(imspeaks_filename)>1){
    stop(call. = FALSE, "Multiple imspeaks files given!")
  }
  if (length(imscoords_filename)>1){
    stop(call. = FALSE, "Multiple imscoords files given!")
  }
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
  pd$ims_x <- coords[, 2]
  pd$ims_y <- coords[, 1]

  # check if IMS was registered to microscopy
  if ("xy_micro_physical" %in% rhdf5::h5ls(imscoords_filename)$name) {
    tmpimsxy <- rhdf5::h5read(imscoords_filename, "xy_original") |>
      t() |>
      as.data.frame() |>
      dplyr::rename(ims_x = V1, ims_y = V2) |>
      dplyr::mutate(ims_xy = paste0(ims_x, "_", ims_y))

    pdf <- paste0(pd$ims_x, "_", pd$ims_y) %in% tmpimsxy$ims_xy
    pdout <- pd[pdf,]

    imsphy <- rhdf5::h5read(imscoords_filename, "xy_micro_physical") |>
      t() |>
      as.data.frame() |>
      dplyr::rename(ims_x_phy = V1, ims_y_phy = V2)

    imsxy <- cbind(tmpimsxy, imsphy)
    if (dim(pdout)[1] != dim(imsxy)[1]) {
      do_flip <- FALSE
      pdf <- paste0(pd$ims_y, "_", pd$ims_x) %in% tmpimsxy$ims_xy
      pdout <- pd[pdf,]
      pdout$ims_x_phy <- imsxy$ims_y_phy
      pdout$ims_y_phy <- imsxy$ims_x_phy
    } else {
      do_flip <- TRUE 
      pdout$ims_x_phy <- imsxy$ims_x_phy
      pdout$ims_y_phy <- imsxy$ims_y_phy
    }
    stopifnot(dim(pdout)[1] == dim(imsxy)[1])
    if (idx_by_location){
      subdf <- pd
      pd_sub <- subdf
      sp::coordinates(subdf) <- c("ims_x", "ims_y")
      sp::gridded(subdf) <- TRUE
      dst <- 3
      nl2 <- BiocNeighbors::findNeighbors(sp::coordinates(subdf), dst)
      wm <- nl2$index
      class(wm) <- c("nb", "list")
      dj <- spdep::n.comp.nb(wm)
      pd$region_id <- dj$comp.id
      reg <- unique(pd$region_id[pdf])
      stopifnot(length(reg) == 1)
      pdout2 <- pd[pd$region_id == reg,]
      pdout2$ims_idx <- seq_len(dim(pdout2)[1]) - 1
      pdout2 <- pdout2[,c("ims_idx","ims_x","ims_y")]
      if(do_flip){
        tmp <- pdout2$ims_x
        pdout2$ims_x <- pdout2$ims_y
        pdout2$ims_y <- tmp 
      }
      pdout$ims_idx <- NULL
      pdout <- dplyr::left_join(pdout,pdout2, by = dplyr::join_by(ims_x, ims_y))
    }

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
  # pdout$ims_idx <- seq_len(dim(pdout)[1]) - 1
  pdout
}

#' Combine IMC and IMS data
#'
#' @param imspeaks_filename filename of h5 file containing peaks of IMS
#' @param imscoords_filename filename of h5 file containing coordinates of IMS
#' @param celloverlap_filename filename of csv file containing IMC cells overlap with IMS pixels
#' @param cellcentroids_filenames filename of csv file containing cell centroids
#' @param malid_pixelsize size of IMS pixels
#' @param sample_id_colname column name of sample identifier in spe
#' @param complete_maldi logical, should all IMS pixels be returned? default is to only return pixels containing cells
#'
#' @return data.frame
#' @export
#'
#' @examples
#' imspeaks_filename <- file.path("inst", "extdata", "IMS_test_combined_peaks.h5")
#' imscoords_filename <- file.path("inst", "extdata", "postIMS_to_IMS_test_combined-IMSML-coords.h5")
#' celloverlap_filename <- here::here("inst", "extdata", "test_combined_Cirrhosis-TMA-5_New_Detector_001_cell_overlap_IMS.csv")
#' cellcentroids_filename <- here::here("inst", "extdata", "test_combined_Cirrhosis-TMA-5_New_Detector_001_cell_centroids.csv")
#' imcims_df <- create_imsc(imspeaks_filename, imscoords_filename, celloverlap_filename, cellcentroids_filename, 30)
create_imsc <- function(imspeaks_filename, imscoords_filename,
                        celloverlap_filename, cellcentroids_filename,
                        maldi_pixelsize,
                        sample_id_colname = "sample_id",
                        complete_maldi = FALSE,
                        idx_by_location = FALSE) {
  force(maldi_pixelsize)
  if (!all(file.exists(imscoords_filename))) {
    stop(call. = FALSE, paste0("file '", imscoords_filename[!file.exists(imscoords_filename)], "' does not exist"))
  }
  if (length(idx_by_location) == 1 & length(imscoords_filename) > 1){
    idx_by_location <- rep(idx_by_location,length(imscoords_filename))
  } else{
    stopifnot(length(idx_by_location) == length(imscoords_filename))
  }

  if (length(imspeaks_filename)>1){
    stop(call. = FALSE, "Multiple imspeaks files are given!")
  }
  if(length(imscoords_filename) == 1){
    pd <- get_peak_data(imspeaks_filename, imscoords_filename, maldi_pixelsize)
    pd$sample_id <- names(imscoords_filename)
  } else{
    pdls <- lapply(seq_along(imscoords_filename), function(i) get_peak_data(imspeaks_filename, imscoords_filename[i], maldi_pixelsize,idx_by_location=idx_by_location[i]))
    for(i in seq_along(pdls)){
      pdls[[i]]$sample_id <- names(imscoords_filename)[i]
    }
    pd <- do.call(rbind,pdls)
  }

  if (!all(file.exists(celloverlap_filename))) {
    stop(call. = FALSE, paste0("file '", celloverlap_filename, "' does not exist"))
  }
  if (!all(file.exists(cellcentroids_filename))) {
    stop(call. = FALSE, paste0("file '", cellcentroids_filename, "' does not exist"))
  }
  stopifnot(length(celloverlap_filename) == length(cellcentroids_filename))
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
  # combine data
  celldf <- do.call(rbind, celldf_ls)

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
    combined_df <- celldf |>
      dplyr::right_join(pd_sub[pd_sub$region_id %in% reg, ], by = dplyr::join_by("ims_idx","sample_id"))
    combined_df_split <- base::split(combined_df, combined_df[["region_id"]])
  } else {
    combined_df <- celldf |>
      dplyr::inner_join(pd, by = dplyr::join_by("ims_idx","sample_id")) |>
      dplyr::mutate(region_id = !!dplyr::sym(sample_id_colname))
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
    dplyr::filter(!is.na(Pos_X_on_IMS)) |>
    dplyr::group_by(ims_idx, !!dplyr::sym(sample_id_colname)) |>
    dplyr::summarise(
      .groups = "keep",
      Pos_X_max = max(Pos_X_on_IMS),
      Pos_X_min = min(Pos_X_on_IMS),
      Pos_Y_max = max(Pos_Y_on_IMS),
      Pos_Y_min = min(Pos_Y_on_IMS)
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

  used_samples <- unique(combined_df[["sample_id"]])
  used_samples <- used_samples[!is.na(used_samples)]
  dfls <- lapply(used_samples, function(sam) {
    # get locations
    dff <- combined_df[combined_df$sample_id == sam & !is.na(combined_df$cell_idx), ]
    dfa <- combined_df[combined_df$region_id == unique(dff$region_id), ]

    # find convex hull of pixels with cells
    sfmf <- sf::st_multipoint(as.matrix(dff[, c("ims_x", "ims_y")]))
    chf <- sf::st_convex_hull(sfmf)

    # find pixels in convex hull without cells
    sfma <- sf::st_multipoint(as.matrix(dfa[, c("ims_x", "ims_y")]))
    sfmaf <- sf::st_intersection(sf::st_multipoint(sfma), sf::st_polygon(chf))

    matp <- as.matrix(sfmaf)
    inds <- paste0(dfa$ims_x, "_", dfa$ims_y) %in% paste0(matp[, 1], "_", matp[, 2])

    if (complete_maldi){
      dfa$in_IMC <- inds
      dffn <- dfa
    } else{
      dffn <- dfa[inds, ]
      dffn$in_IMC <- TRUE
    }

    dffn
  })
  combined_df <- do.call(rbind, dfls)

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
                                             IMS_pixel_size = NULL, ims_cols = c("ims_x", "ims_y"), cell_cols = c("Pos_X_on_IMS", "Pos_Y_on_IMS")) {
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
has_flipped_axis_ims <- function(x, type = c("cor", "dist"), threshold = 0.9, IMS_pixel_size = NULL, ims_cols = c("ims_x", "ims_y"), cell_cols = c("Pos_X_on_IMS", "Pos_Y_on_IMS")) {
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
                            ims_cols = c("ims_x", "ims_y"), cell_cols = c("Pos_X_on_IMS", "Pos_Y_on_IMS"),
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
