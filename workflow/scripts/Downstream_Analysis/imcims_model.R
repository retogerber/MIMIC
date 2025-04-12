suppressPackageStartupMessages(suppressWarnings({
  library(SpatialExperiment)
  library(future.apply)
  library(spdep)
  library(spatialreg)
}))


#' Create Interaction Terms
#'
#' This function creates interaction terms between cell types within a specified maximum distance.
#' It processes the input data frames, splits them by sample names, and calculates interaction values
#' for each combination of cell types.
#'
#' @param imc_full A data frame containing cell information including sample name, cell type, interaction value, cell ID, and coordinates.
#' @param imcims_full A data frame containing IMS information including sample name, cell ID, and IMS ID.
#' @param max_distance The maximum distance for defining interactions between cells. Default is 20.
#' @param column_names A nested list specifying column names in the input data frames with the following structure:
#'                     - imc_full: A list specifying column names in the imc_full data frame.
#'                     - imcims_full: A list specifying column names in the imcims_full data frame.
#' @param n_workers The number of workers for parallel processing. Default is 1.
#' @return A list of data frames containing interaction terms for each sample.
#' @export
create_interaction_terms <- function(
    imc_full,
    imcims_full,
    max_distance = 20,
    column_names = list(
      imc_full = list(
        sample_name = "sample_name",
        celltype = "celltype",
        interaction = "interaction",
        cellid = "cellid",
        x = "x",
        y = "y"
      ),
      imcims_full = list(
        sample_name = "sample_name",
        cellid = "cellid",
        imsid = "imsid"
      )
    ),
    n_workers = 1) {
  # require(data.table)
  stopifnot(all(names(column_names) %in% c("imc_full", "imcims_full")))
  stopifnot(all(unlist(names(column_names[["imc_full"]])) %in% c("sample_name", "celltype", "interaction", "cellid", "x", "y")))
  stopifnot(all(unlist(column_names[["imc_full"]]) %in% colnames(imc_full)))
  stopifnot(all(unlist(names(column_names[["imcims_full"]])) %in% c("sample_name", "cellid", "imsid")))
  stopifnot(all(unlist(column_names[["imcims_full"]]) %in% colnames(imcims_full)))

  cn_imc <- column_names[["imc_full"]]
  cn_imcims <- column_names[["imcims_full"]]

  tva <- sort(unlist(column_names[["imc_full"]]))
  imc_full <- data.table::as.data.table(imc_full)[, ..tva]
  data.table::setnames(imc_full, tva, names(tva))
  data.table::setkey(imc_full, sample_name, celltype, cellid)

  tva <- sort(unlist(column_names[["imcims_full"]]))
  imcims_full <- data.table::as.data.table(imcims_full)[, ..tva]
  data.table::setnames(imcims_full, tva, names(tva))
  data.table::setkey(imcims_full, cellid, sample_name, imsid)

  celltypes_used <- unique(as.character(imc_full$celltype))
  # create all combinations of cell types for interactions
  all_ct_mat <- t(combn(celltypes_used, 2))
  colnames(all_ct_mat) <- c("x", "y")
  all_ct_combs <- as.data.frame(all_ct_mat)


  sample_names <- unique(imc_full$sample_name)
  sample_names <- sample_names[sample_names %in% unique(imcims_full$sample_name)]
  imc_full <- imc_full[sample_name %in% sample_names, ]
  imcims_full <- imcims_full[sample_name %in% sample_names, ]

  imc_ls <- split(imc_full, imc_full$sample_name)
  imcims_ls <- split(imcims_full, imcims_full$sample_name)
  inp_ls <- lapply(sample_names, \(sam) list(imc = imc_ls[[sam]], imcims = imcims_ls[[sam]], sample = sam))


  plan(multisession, workers = min(n_workers, length(sample_names)))
  dflc_comb_ls <- future_lapply(
    inp_ls,
    future.seed = 123,
    future.globals =
      list(
        all_ct_combs = all_ct_combs,
        max_distance = max_distance
      ),
    \(inp){
      imc_sub <- inp$imc
      imcims_sub <- inp$imcims

      all_imsid <- data.frame(imsid = unique(imcims_sub$imsid))

      cord <- as.matrix(imc_sub[, c("x", "y")])
      cogr <- BiocNeighbors::findNeighbors(cord, threshold = max_distance, get.distance = FALSE)
      cp <- igraph::graph_from_adj_list(cogr$index) |>
        igraph::simplify() |>
        igraph::as_edgelist()


      # loop through all cell type combinations
      dflc_ls <- lapply(seq_along(all_ct_combs[, 1]), \(i){
        # select all nodes
        fi <- seq_along(imc_sub$celltype)[imc_sub$celltype == all_ct_combs[i, 1]]
        ti <- seq_along(imc_sub$celltype)[imc_sub$celltype == all_ct_combs[i, 2]]
        shi <- cp[, 1] %in% fi & cp[, 2] %in% ti
        cp_sub <- cp[shi, , drop = FALSE]

        # interaction value, product of cell interactions
        interactions <- imc_sub$interaction[cp_sub[, 1]] * imc_sub$interaction[cp_sub[, 2]]

        match_imsid <- function(type_edge) {
          a <- data.table::data.table(cellid = imc_sub$cellid[cp_sub[, type_edge]])
          a[imcims_sub, on = "cellid", imsid := i.imsid]
          a[["imsid"]]
        }

        # combined data frame of ims ids and values
        df <- data.frame(
          from = imc_sub$cellid[cp_sub[, 1]],
          to = imc_sub$cellid[cp_sub[, 2]],
          value = interactions,
          from_imsid = match_imsid(1),
          to_imsid = match_imsid(2)
        )
        if (dim(df)[1] > 0) {
          # in long format
          dfl <- tidyr::pivot_longer(df, dplyr::ends_with("_imsid"), names_to = "edge_type", values_to = "imsid")
          # aggregate by ims pixel
          dfl <- dfl |>
            dplyr::filter(!is.na(imsid)) |>
            dplyr::group_by(imsid) |>
            dplyr::summarise(value = sum(value))

          # add interaction values, set missing values to 0
          dflc <- dplyr::left_join(all_imsid, dfl, by = "imsid") |>
            dplyr::mutate(value = ifelse(is.na(value), 0, value))

          # rename
          dflc[[paste0("`", all_ct_combs[i, 1], ":", all_ct_combs[i, 2], "`")]] <- dflc$value
          dflc$value <- NULL
        } else {
          # full length ims pixel ids
          dflc <- all_imsid
          dflc[[paste0("`", all_ct_combs[i, 1], ":", all_ct_combs[i, 2], "`")]] <- 0
        }
        return(dflc)
      })

      # combine all interaction terms
      dflc_lssu <- lapply(dflc_ls, \(l) l[, 2])
      dflc <- do.call(Reduce, list(f = cbind, x = dflc_lssu))
      colnames(dflc) <- sapply(seq_along(all_ct_combs[, 1]), \(i)paste0("`", all_ct_combs[i, 1], ":", all_ct_combs[i, 2], "`"))
      return(dflc)
    }
  )
  names(dflc_comb_ls) <- sample_names

  return(dflc_comb_ls)
}
#
# imcims_df_full <- readRDS(here::here("output","imcims_combined_full.rds"))
# sce <- readRDS(here::here("output","combined","combined_subsce_v2.rds"))
#
# imc_full <- cbind(as.data.frame(colData(sce)[,c("sample_name","Cell_Type","area","ObjectNumber")]),
#                   as.data.frame(spatialCoords(sce)))
# # imc_full <- imc_full[imc_full$sample_name == unique(imc_full$sample_name)[1],]
# imcims_full <- imcims_df_full[,c("sample_name","cell_idx","ims_idx")]
# # imcims_full <- imcims_full[imcims_full$sample_name == unique(imcims_full$sample_name)[1],]
# imc_full$interaction <- 1
#
# dflc_comb_ls <- create_interaction_terms(
#   imc_full,
#   imcims_full,
#   column_names=list(
#     imc_full=list(
#       sample_name="sample_name",
#       celltype="Cell_Type",
#       interaction="interaction",
#       cellid="ObjectNumber",
#       x="Pos_X",
#       y="Pos_Y"
#
#     ),
#     imcims_full=list(
#       sample_name="sample_name",
#       cellid="cell_idx",
#       imsid="ims_idx"
#     )
#   ),
#   n_workers = 10
#   )




#' Convert imcims data from long to wide format
#'
#' This function takes an imcims data frame in long format and converts it to wide format.
#' It reshapes the data frame by pivoting the celltype column into multiple columns,
#' with each unique celltype value becoming a separate column. The values in the new columns
#' are calculated using the provided function.
#'
#' @param imcims_full The imcims data frame.
#' @param column_names A nested list specifying column names in the imcims data frame with the following structure:
#'                     - sample_name: The column name for the sample name.
#'                     - celltype: The column name for the celltype.
#'                     - value: The column name for the value.
#'                     - imsid: The column name for the imsid.
#'                     - coords: A character vector specifying the column names for the x and y coordinates.
#'                     - responses: A character vector specifying the column names for the response variables.
#'                     - covariates: A character vector specifying the column names for the covariate variables.
#' @return The imcims data frame in wide format.
#' @export
to_wide <- function(
    imcims_full,
    column_names = list(
      sample_name = "sample_name",
      celltype = "celltype",
      value = "value",
      imsid = "imsid",
      coords = c("x", "y"),
      responses = c("response1", "response2"),
      covariates = c("covariate1", "covariate2"),
      sample_covariates = c("covariate1", "covariate2")
    )) {
  fn <- list(
    value = function(x) {
      sum(na.omit(x))
    }
  )
  names(fn) <- column_names$value
  imcims_wide <- imcims_full |>
    dplyr::select(c(
      !!column_names$imsid,
      !!column_names$sample_name,
      !!column_names$value,
      !!column_names$celltype,
      dplyr::all_of(!!column_names$coords),
      dplyr::all_of(!!column_names$responses),
      dplyr::all_of(!!column_names$covariates),
      dplyr::all_of(!!column_names$sample_covariates)
    )) |>
    tidyr::pivot_wider(
      id_cols = c(
        !!column_names$imsid,
        !!column_names$sample_name,
        dplyr::all_of(column_names$coords),
        dplyr::all_of(column_names$responses),
        dplyr::all_of(column_names$covariates),
      dplyr::all_of(!!column_names$sample_covariates)
      ),
      names_from = !!column_names$celltype,
      values_from = !!column_names$value,
      values_fn = fn,
      values_fill = 0
    )
  imcims_wide
}
# mzcol_names_used <- attr(imcims_df_full,"metadata")[["mzcol_names_used"]]
# imcims_wide <- to_wide(
#   imcims_df_full,
#   column_names=list(
#     sample_name="sample_name",
#     celltype="Cell_Type",
#     value="maldi_area_filled",
#     imsid="ims_idx",
#     coords=c("ims_x","ims_y"),
#     responses=mzcol_names_used,
#     covariates=c("TMA_type","type","Sex")
#   )
#   )
#
# imcims_wide <- imcims_wide |>
#   dplyr::filter(!(sample_name %in% c("NASH_HCC_TMA-2_E8", "NASH_HCC_TMA-2_E9")))



#' neighborhood function
#'
#' This function generates a neighborhood object based on the spatial coordinates of points.
#'
#' @param x A numeric vector representing the x-coordinates of the points.
#' @param y A numeric vector representing the y-coordinates of the points.
#' @param rownam A character vector of row names for the points.
#' @param max_dist The maximum distance for defining neighborhood relationships. Default is sqrt(2).
#' @param style The style of neighborhood weights. Default is "W".
#' @param weights Logical value indicating whether to calculate weights based on distances. Default is TRUE.
#'
#' @return A listw object representing the neighborhood weights.
#'
#' @examples
#' x <- c(1, 2, 3)
#' y <- c(4, 5, 6)
#' rownam <- c("A", "B", "C")
#' neighborhood(x, y, rownam)
#'
#' @export
neighborhood <- function(x, y, rownam, max_dist = sqrt(2)+2*.Machine$double.eps, style = "B", weights = FALSE) {
  spatialreg::set.ZeroPolicyOption(TRUE)
  subdf <- data.frame(x = x, y = y)
  sp::coordinates(subdf) <- c("x", "y")
  sp::gridded(subdf) <- TRUE
  wm <- spdep::dnearneigh(sp::coordinates(subdf), 0, max_dist, row.names = rownam)
  if (weights) {
    dists <- spdep::nbdists(wm, subdf)
    weights <- lapply(dists, \(d) 1 / d)
    listw <- spdep::nb2listw(wm, glist = weights, style = style, zero.policy = TRUE)
  } else {
    listw <- spdep::nb2listw(wm, style = style, zero.policy = TRUE)
  }
  listw
}

#' Calculate neighborhoods for multiple datasets
#'
#' This function generates neighborhoods for multiple datasets using the neighborhood function.
#'
#' @param dat_ls A list of datasets.
#' @param x The name of the x-coordinate column in each dataset.
#' @param y The name of the y-coordinate column in each dataset.
#' @param rownam The name of the row name column in each dataset.
#' @param max_dist The maximum distance for defining a neighborhood.
#' @param style The style of neighborhood calculation.
#' @param weights Whether to use weights in the neighborhood calculation.
#'
#' @return A list of neighborhoods for each dataset.
#'
#' @examples
#' dat1 <- data.frame(x = c(1, 2, 3), y = c(4, 5, 6), rownam = c("A", "B", "C"))
#' dat2 <- data.frame(x = c(7, 8, 9), y = c(10, 11, 12), rownam = c("D", "E", "F"))
#' dat_ls <- list(dat1, dat2)
#' all_neighborhoods(dat_ls, x = "x", y = "y", rownam = "rownam", max_dist = 2)
#'
#' @export
all_neighborhoods <- function(dat_ls, x = "x", y = "y", rownam = "rownam", max_dist = sqrt(2)+2*.Machine$double.eps, style = "B", weights = FALSE) {
  future_lapply(dat_ls, future.seed = 123, future.globals = list(x = x, y = y, rownam = rownam, max_dist = max_dist, style = style, weights = weights, neighborhood = neighborhood), \(df){
    tryCatch(
      {
        neighborhood(df[[x]], df[[y]], df[[rownam]], max_dist, style = style, weights = weights)
      },
      error = function(e) {
        NA
      }
    )
  })
}

#' Add Interactions to Nested Data Frame
#'
#' This function adds interaction data from a list of data frames to an existing nested data frame.
#'
#' @param dflc_comb_ls A named list of data frames, each corresponding to a sample.
#' @param imcims_wide_nest A nested data frame containing the original data.
#' @param sample_names A character vector of sample names to be processed.
#'
#' @return A nested data frame with the added interaction data.
#' 
#' @export
add_interactions <- function(dflc_comb_ls, imcims_wide_nest, sample_names) {
  if (any(!(names(dflc_comb_ls) %in% sample_names))) {
    missing_samples <- names(dflc_comb_ls)[!(names(dflc_comb_ls) %in% sample_names)]
    stop(paste0("Samples ", paste0(missing_samples, collapse = ", "), "are missing in `imcims_wide`!"))
  }

  if (any(!(sample_names %in% names(dflc_comb_ls)))) {
    missing_samples <- sample_names[!(sample_names %in% names(dflc_comb_ls))]
    stop(paste0("Samples ", paste0(missing_samples, collapse = ", "), " are missing in `dflc_comb_ls`!"))
  }

  for (sam in sample_names) {
    # add interactions to existing data frame
    imcims_wide_nest[["data"]][imcims_wide_nest[[column_names$sample_name]] == sam][[1]] <-
      cbind(imcims_wide_nest[["data"]][imcims_wide_nest[[column_names$sample_name]] == sam][[1]], dflc_comb_ls[[sam]])
  }
  imcims_wide_nest
}

#' Get Interaction Names
#'
#' This function generates interaction names by creating all possible combinations of the provided cell types.
#'
#' @param celltypes A character vector of cell types.
#'
#' @return A character vector of interaction names, where each name is a combination of two cell types separated by a colon.
#'
#' @export
get_interaction_names <- function(celltypes) {
  # create all combinations of cell types for interactions
  all_ct_mat <- t(combn(sort(as.character(celltypes)), 2))
  colnames(all_ct_mat) <- c("x", "y")
  all_ct_combs <- as.data.frame(all_ct_mat)

  interaction_names <- apply(all_ct_combs, 1, paste0, collapse = ":")
  interaction_names
}

create_formula <- function(response, celltypes, interaction=NULL, celltype_interaction = FALSE, intercept=TRUE) {
  if (celltype_interaction) {
    celltype_interaction_names <- get_interaction_names(celltypes)
  } else {
    celltype_interaction_names <- NULL
  }
  if (!is.null(interaction)) {
    interactions <- paste0("+",paste(paste0(celltypes,":", interaction),collapse = "+"))
    covariates <- c(celltypes, interaction)
  } else {
    interactions <- NULL
    covariates <- celltypes
  }
  if (intercept){
    intercept_str <- "1 +"
  } else{
    intercept_str <- "0 +"
  }
  fl <- formula(paste0(
    "`", response, "`~",intercept_str,
    paste(paste0("`", sort(as.character(covariates)), "`"), collapse = "+"),
    paste(c("", sort(celltype_interaction_names)), collapse = "+"),
    interactions
  ))
  fl
}


get_wmls_ls <- function(imcims_wide, column_names, wmls_ls = NULL) {
  xl <- split(imcims_wide[[column_names$coords[1]]], imcims_wide[[column_names$sample_name]])
  yl <- split(imcims_wide[[column_names$coords[2]]], imcims_wide[[column_names$sample_name]])
  rnl <- split(paste0(imcims_wide[[column_names$sample_name]], ".", imcims_wide$ims_idx), imcims_wide[[column_names$sample_name]])
  dat_ls <- lapply(names(xl), \(k){
    list(
      x = xl[[k]],
      y = yl[[k]],
      rownam = rnl[[k]]
    )
  })
  names(dat_ls) <- names(xl)
  if (!is.null(wmls_ls)) {
    style <- wmls_ls[[1]]$style
    weights <- any(unlist(lapply(wmls_ls[[1]]$weights[1:100], function(x) x >= 1)))
  } else {
    style <- "W"
    weights <- TRUE
  }
  wmls_ls <- all_neighborhoods(dat_ls, max_dist = 1, style = style, weights = weights)
  names(wmls_ls) <- names(xl)
  wmls_ls
}

fit_SARs <- function(imcims_wide_nest,
                     fl,
                     column_names = list(
                       sample_name = "sample_name",
                       celltypes = c("celltype_1", "celltype_2"),
                       value = "value",
                       imsid = "imsid",
                       coords = c("x", "x"),
                       response = "response1",
                       covariate = "covariate1",
                       sample_covariates = c("covariate1", "covariate2")
                     ),
                     interaction_names_quo = NULL,
                     n_workers = 1,
                     wmls_ls = NULL,
                     trs_ls = NULL,
                     debug=FALSE, 
                     compress_fit = FALSE) {
  sample_names <- unique(imcims_wide_nest[[column_names$sample_name]])
  inp_ls <- lapply(sample_names, \(sam) {
    tmpd <- imcims_wide_nest[["data"]][imcims_wide_nest[[column_names$sample_name]] == sam][[1]]
    celltype_is_present <- column_names$celltypes %in% colnames(tmpd)
    tmpd <- as.data.frame(tmpd[, c(column_names$response, column_names$imsid, column_names$celltypes[celltype_is_present],column_names$covariate, interaction_names_quo)])
    for (ct in column_names$celltypes[!celltype_is_present]){
      tmpd[[ct]] <- 0
    }
    rownames(tmpd) <- paste0(sam, ".", tmpd[[column_names$imsid]])
    list(
      data = tmpd,
      wmls = wmls_ls[[sam]],
      trs = trs_ls[[sam]],
      sample = sam
    )
  })
  fit_ls <- future_lapply(
    inp_ls,
    future.seed = 123,
    future.globals =
      list(
        fl = fl,
        sample_names = sample_names,
        n_workers = n_workers,
        debug=debug,
        compress_fit = compress_fit,
        compress_sarlm = compress_sarlm
      ),
    \(inp){
      suppressPackageStartupMessages(library(spdep))
      suppressPackageStartupMessages(library(spatialreg))
      suppressPackageStartupMessages(library(Matrix))
      RhpcBLASctl::blas_set_num_threads(floor(n_workers / max(1, min(floor(n_workers / 4), length(sample_names)))))
      tryCatch(
        {
          # spatialreg::errorsarlm(fl, data = inp$data, listw = inp$wmls, method = "Matrix", trs = inp$trs, zero.policy = TRUE, control = list(pWOrder = 1000))
          # spatialreg::lagsarlm(fl, data = inp$data, listw = inp$wmls, method = "Matrix", trs = inp$trs, zero.policy = TRUE)
          fit <- do.call(
            spatialreg::lagsarlm,
            list(
              formula = fl,
              data = inp$data,
              listw = inp$wmls,
              method = ifelse(inp$wmls$style == "B","Matrix_J", "Matrix"),
              trs = inp$trs,
              zero.policy = TRUE
            )
          )
          if (compress_fit){
            fit <- compress_sarlm(fit)
          }
          fit
        },
        error = function(e) {
          if(debug){
            warning(as.character(e))
          }
          NULL
        }
      )
    }
  )
  names(fit_ls) <- sample_names
  is_null <- sapply(fit_ls, is.null)
  fit_ls[is_null] <- NULL
  fit_ls
}

fit_rlm <- function(imcims_wide_nest, fl, column_names) {
  sample_names <- unique(imcims_wide_nest[[column_names$sample_name]])
  ctrl <- robustbase::lmrob.control("KS2014", fast.s.large.n = Inf)
  fit_ls <- lapply(sample_names, \(sam){
    tmpd <- as.data.frame(imcims_wide_nest[["data"]][imcims_wide_nest[[column_names$sample_name]] == sam][[1]])
    rownames(tmpd) <- paste0(sam, ".", tmpd[[column_names$imsid]])
    # print(sam)
    do.call(
      robustbase::lmrob,
      list(
        formula = fl,
        data = tmpd,
        control = ctrl
      )
    )
  })
  names(fit_ls) <- sample_names
  is_null <- rep(FALSE, length(sample_names))
  fit_ls[is_null] <- NULL
  fit_ls
}

fit_lm <- function(imcims_wide_nest, fl, column_names, family) {
  sample_names <- unique(imcims_wide_nest[[column_names$sample_name]])
  fit_ls <- lapply(sample_names, \(sam){
    tmpd <- as.data.frame(imcims_wide_nest[["data"]][imcims_wide_nest[[column_names$sample_name]] == sam][[1]])
    rownames(tmpd) <- paste0(sam, ".", tmpd[[column_names$imsid]])
    do.call(
      glm,
      list(
        formula = fl,
        data = tmpd,
        family = family
      )
    )
  })
  names(fit_ls) <- sample_names
  is_null <- rep(FALSE, length(sample_names))
  fit_ls[is_null] <- NULL
  fit_ls
}
      
compress_sarlm <- function(fit){
  stopifnot(is(fit, "Sarlm"))
  # if tarX == X
  if (all(fit$tarX == fit$X)){
    for(at in names(attributes(fit$tarX))){
      if (!(at %in% names(attributes(fit$X))) | !identical(attr(fit$tarX,at),attr(fit$X,at))){
        if (at == "dimnames"){
          attr(fit$X, paste0("tarX",at)) <- list()
          for (i in seq(2)){
            if (!identical(attr(fit$tarX,at)[[i]],attr(fit$X,at)[[i]])){
              attr(fit$X, paste0("tarX",at))[[i]] <- attr(fit$tarX,at)[[i]]
            }
          }
        } else{
          attr(fit$X, paste0("tarX",at)) <- attr(fit$tarX,at)
        }
      }
    }
    fit$tarX <- NULL
  }
  # if tary == y
  if (all(fit$tary == fit$y)){
    fit$tary <- NULL
  }

  # convert to sparse matrix
  Xsp <- as(fit$X, "CsparseMatrix")
  if (object.size(fit$X)>object.size(Xsp)){
    Xattrs <- attributes(fit$X)
    fit$X <- Xsp
    matchings <- list("Dim"="dim", "Dimnames"="dimnames")
    for (tmpat in names(Xattrs)){
      if (!(tmpat %in% matchings)){
        matchings[[tmpat]] <- tmpat
      }
    }
    for (at in matchings){
      matattrs <- names(attributes(fit$X))
      matattrs <- sapply(matattrs, function(tmpat) ifelse(tmpat %in% names(matchings), matchings[[tmpat]], tmpat))
      if (!(at %in% matattrs) | !identical(Xattrs[[at]],attributes(fit$X)[[names(matchings)[matchings==at]]])){
        attr(fit$X, paste0("X",at)) <- Xattrs[[at]]
      }
    }
  }
  if (!is.null(fit$tarX)){
    # convert to sparse matrix
    tarXattr <- attributes(fit$tarX)
    tarXsp <- as(fit$tarX, "CsparseMatrix")
    if (object.size(fit$tarX)>object.size(tarXsp)){
      fit$tarX <- tarXsp
      attr(fit$tarX, "assign") <- tarXattr$assign
    }
  }

  # rename covariate names
  attr(fit$coefficients, "names")
  if(all(attr(fit$rest.se, "names")==attr(fit$coefficients, "names"))){
    attr(fit$rest.se, "names") <-  NULL
  }

  # rename sample names
  if (all(attr(fit$y, "names") == attr(fit$fitted.values, "names"))){
    attr(fit$fitted.values, "names") <-  NULL
  }
  if (all(attr(fit$y, "names") == attr(fit$residuals, "names"))){
    attr(fit$residuals, "names") <-  NULL
  }
  if (all(attr(fit$y, "names") == attr(fit$tary, "names"))){
    attr(fit$tary, "names") <-  NULL
  }


  # rename covariate names + sample names
  if (all(dimnames(fit$X)[[1]] == attr(fit$y, "names"))){
    dimnames(fit$X) <- list(NULL, dimnames(fit$X)[[2]])
  }
  if (!is.null(fit$tarX)){
    if (all(dimnames(fit$tarX)[[1]] == attr(fit$y, "names"))){
      dimnames(fit$tarX) <- list(NULL, dimnames(fit$tarX)[[2]])
    }
  }

  # compress sample names
  nams <- attr(fit$y, "names")
  nams_split <- stringr::str_split(nams, "\\.", simplify = TRUE)
  nams_compressed <- paste0(".",nams_split[,2])
  sam_nams <- unique(nams_split[,1])
  for (i in seq_along(sam_nams)){
    nams_compressed[nams_split[,1] == sam_nams[i]] <- paste0(i, nams_compressed[nams_split[,1] == sam_nams[i]])
    nams_compressed[nams_split[,1] == sam_nams[i]][1] <- paste0(nams_split[nams_split[,1] == sam_nams[i],1][1],".",nams_split[nams_split[,1] == sam_nams[i],2][1])
  }
  attr(fit$y, "names") <-  nams_compressed

  fit
}

decompress_sarlm <- function(fit){
  stopifnot(is(fit, "Sarlm"))
  nams <- attr(fit$y, "names")
  nams_split <- stringr::str_split(nams, "\\.", simplify = TRUE)
  nams_compressed <- paste0(".",nams_split[,2])
  sam_nams <- unique(nams_split[,1])

  i<-1
  while(length(sam_nams)>1){
    cur_sam <- sam_nams[1]
    sam_nams <- sam_nams[-1]
    if(any(sam_nams == i)){
      nams_split[nams_split[,1] == i,1] <- cur_sam
      sam_nams <- sam_nams[sam_nams != i]
    }
    i<-i+1
  }
  nams_uncompressed <- paste0(nams_split[,1],".",nams_split[,2])
  attr(fit$y, "names") <-  nams_uncompressed


  if(is.null(fit$tary)){
    fit$tary <- fit$y
  }

  if (is.null(attr(fit$rest.se, "names"))){
    attr(fit$rest.se, "names") <-  attr(fit$coefficients, "names")
  }

  if (is.null(attr(fit$fitted.values, "names"))){
    attr(fit$fitted.values, "names") <-  attr(fit$y, "names")
  }
  if (is.null(attr(fit$residuals, "names"))){
    attr(fit$residuals, "names") <-  attr(fit$y, "names")
  }
  if (is.null(attr(fit$tary, "names"))){
    attr(fit$tary, "names") <-  attr(fit$y, "names")
  }

  if ("dgCMatrix" %in% class(fit$X)){
    Xattrs <- attributes(fit$X)
    Xtarattrs <- names(Xattrs)[stringr::str_starts(names(Xattrs),"Xtar")]
    fit$X <- as.matrix(fit$X)
    for (xtarat in Xtarattrs){
      if (!(xtarat %in% names(attributes(fit$X)))){
        attr(fit$X, xtarat) <- Xattrs[[xtarat]]
      }
    }
  }
  if (is.null(dimnames(fit$X)[[1]])){
    dimnames(fit$X) <- list(attr(fit$y, "names"), dimnames(fit$X)[[2]])
  }

  if(is.null(fit$tarX)){
    fit$tarX <- fit$X
    Xattrs <- attributes(fit$X)
    Xtarattrs <- names(Xattrs)[stringr::str_starts(names(Xattrs),"Xtar")]
    for (xtarat in Xtarattrs){
        at <- stringr::str_replace(xtarat,"XtarX","")
        if (at == "dimnames"){
          for (i in seq(2)){
            if (!is.null(attr(fit$X,xtarat)[[i]]) && !identical(attr(fit$tarX,at)[[i]],attr(fit$X,xtarat)[[i]])){
              attr(fit$tarX, at)[[i]] <- Xattrs[[xtarat]][[i]]
            }
          }
        } else{
          # attr(fit$X, paste0("tarX",at)) <- attr(fit$tarX,at)
          attr(fit$tarX, at) <- Xattrs[[xtarat]]
        }
        attr(fit$tarX, xtarat) <- NULL
    }
  }
  if ("dgCMatrix" %in% class(fit$tarX)){
    tarXattrs <- attributes(fit$tarX)
    fit$tarX <- as.matrix(fit$tarX)
    if ("assign" %in% names(tarXattrs)){
      attr(fit$tarX, "assign") <- tarXattrs$assign
    } else{
      attr(fit$tarX, "assign") <- attr(fit$X, "XtarXassign")
    }
  }
  if (is.null(dimnames(fit$tarX)[[1]])){
    dimnames(fit$tarX) <- list(attr(fit$y, "names"), dimnames(fit$tarX)[[2]])
  }

  Xattrs <- attributes(fit$X)
  Xtarattrs <- names(Xattrs)[stringr::str_starts(names(Xattrs),"Xtar")]
  for (xtarat in Xtarattrs){
    attr(fit$X, xtarat) <- NULL
    attr(fit$tarX, xtarat) <- NULL
  }

  fit
}

# fit_init <- fit_ls[[1]]
# fit_compr <- compress_sarlm(fit_init)
# fit_uncompr <- decompress_sarlm(fit_compr)
# format(object.size(fit_init),"Mb")
# format(object.size(fit_compr),"Mb")
# format(object.size(fit_uncompr),"Mb")

# all.equal(fit_init$X, fit_init$tarX)
# for (n in names(fit_init)){
#   print(n)
#   print(all.equal(fit_init[[n]], fit_uncompr[[n]]))
# }

# imcims_wide <- tmp_args$imcims_wide
# dflc_comb_ls <- tmp_args$dflc_comb_ls
# column_names <- tmp_args$column_names
# wmls_ls <- tmp_args$wmls_ls
# method <- tmp_args$method
# n_workers <- tmp_args$n_workers
# remove_na <- tmp_args$remove_na
# trs_ls <- tmp_args$trs_ls


fit_first_layer <- function(imcims_wide,
                            dflc_comb_ls = NULL,
                            column_names = list(
                              sample_name = "sample_name",
                              celltypes = c("celltype_1", "celltype_2"),
                              value = "value",
                              imsid = "imsid",
                              coords = c("x", "x"),
                              response = "response1",
                              covariate = "covariate1",
                              sample_covariates = c("sample_covariate1", "sample_covariate2")
                            ),
                            method = c("SAR", "lm", "rlm"),
                            n_workers = 1,
                            wmls_ls = NULL,
                            trs_ls = NULL,
                            family = Gamma(link = "log"),
                            remove_na = FALSE,
                            fl=NULL,
                            debug=FALSE,
                            compress_fit = FALSE,
                            intercept = TRUE) {
  method <- match.arg(method)
  stopifnot(n_workers > 0)
  if (remove_na) {
    if ("data.table" %in% class(imcims_wide)){
      tmpcols <- c(column_names$celltypes, column_names$response)
      imcims_wide <- na.omit(imcims_wide, cols=tmpcols)
    } else {
      imcims_wide <- imcims_wide[complete.cases(imcims_wide[,c(column_names$celltypes, column_names$response)]),]
    }
  }
  # split data by sample names
  imcims_wide_nest <- imcims_wide |>
    dplyr::group_by(dplyr::across(c(!!column_names$sample_name, !!dplyr::all_of(column_names$sample_covariates)))) |>
    tidyr::nest() |>
    dplyr::mutate(n = sapply(data, nrow))

  sample_names <- unique(imcims_wide_nest[[column_names$sample_name]])
  # create workers
  if ("sequential" %in% class(plan()) & n_workers > 1){
    # plan(multisession, workers = n_workers){
    plan(multisession, workers = max(1, min(floor(n_workers / 4), length(sample_names))))
  }

  # add interaction terms
  if (!is.null(dflc_comb_ls)) {
    imcims_wide_nest <- add_interactions(dflc_comb_ls, imcims_wide_nest, sample_names)
    interaction_names_quo <- paste0("`", get_interaction_names(column_name$celltypes), "`")
  } else {
    interaction_names_quo <- NULL
  }

  if (is.null(fl)){
    # create formula
    fl <- create_formula(column_names$response, column_names$celltypes, interaction=column_names$covariate, celltype_interaction = !is.null(dflc_comb_ls), intercept=intercept)
  }

  if (method %in% c("SAR")) {
    # get neighborhood weights
    if (is.null(wmls_ls) | remove_na) {
      wmls_ls <- get_wmls_ls(imcims_wide, column_names, wmls_ls)
    }
    # get spatial weights
    if (is.null(trs_ls) | remove_na) {
      # trs_ls <- lapply(wmls_ls, \(wmls){
      trs_ls <- future_lapply(wmls_ls, future.seed = 123, future.globals = FALSE, \(wmls){
        W <- as(wmls, "CsparseMatrix")
        spatialreg::trW(W)
      })
      names(trs_ls) <- names(wmls_ls)
    }
    # fit SAR
    fit_ls <- fit_SARs(imcims_wide_nest = imcims_wide_nest, fl = fl, column_names = column_names, interaction_names_quo = interaction_names_quo, n_workers = n_workers, wmls_ls = wmls_ls, trs_ls = trs_ls, debug=debug, compress_fit = compress_fit)
  } else if (method == "rlm") {
    # fit robust linear model
    fit_ls <- fit_rlm(imcims_wide_nest = imcims_wide_nest, fl = fl, column_names = column_names)
  } else {
    # fit linear model
    fit_ls <- fit_lm(imcims_wide_nest = imcims_wide_nest, fl = fl, column_names = column_names, family = family)
  }
  fit_ls_tidy <- lapply(fit_ls, broom::tidy)
  for (sam in names(fit_ls)) {
    fit_ls_tidy[[sam]]$sample_name <- sam
  }
  fitdf <- do.call(Reduce, list(f = rbind, fit_ls_tidy))

  if (!is.null(fitdf)){
    coefdf <- tidyr::pivot_wider(fitdf, id_cols = "sample_name", values_from = "estimate", names_from = "term")
    coefdf <- cbind(coefdf, imcims_wide_nest[imcims_wide_nest[[column_names$sample_name]] %in% names(fit_ls), c(column_names$sample_name, column_names$sample_covariates, "n")])
  } else{
    coefdf <- NULL
  }

  list(fitdf = fitdf, coefdf = coefdf, fitls = fit_ls)
}


get_predictions <- function(fit1ls, imcims_wide = NULL, wmls_ls = NULL, compress_fit = FALSE) {
  met <- class(fit1ls[[1]]$fitls[[1]])
  mzcol_names <- names(fit1ls)
  dfls <- list()
  for (mzval in mzcol_names) {
    samples <- names(fit1ls[[mzval]]$fitls)
    n_obs <- sum(sapply(samples, function(sam){
      length(predict(fit1ls[[mzval]]$fitls[[sam]]))
    }))
    if (!is.null(imcims_wide) && n_obs < nrow(imcims_wide)){
      if (!data.table::is.data.table(imcims_wide)){
        imcims_wide <- data.table::as.data.table(imcims_wide)
      } 
      do_out_of_sample <- TRUE
    } else{
      do_out_of_sample <- FALSE
    }
    if ("Sarlm" %in% met & !do_out_of_sample){
      tmpdf <- tibble::tibble("sample_id"=character(0), "ims_idx"=character(0), !!paste0("pred_trend_", mzval) := numeric(0), !!paste0("pred_signal_", mzval) := numeric(0), !!paste0("resid_", mzval) := numeric(0))
    } else{
      tmpdf <- tibble::tibble("sample_id"=character(0), "ims_idx"=character(0), !!paste0("pred_", mzval) := numeric(0), !!paste0("resid_", mzval) := numeric(0))
    }

    inp_ls <- lapply(samples, \(sam) {
      if (!is.null(imcims_wide)){
        predout <- predict(fit1ls[[mzval]]$fitls[[sam]])
        if (!is.null(names(predout))){
          predimsidx <- stringr::str_split(names(predout), "\\.", simplify = TRUE)[,2]
        } else {
          predimsidx <- stringr::str_split(attr(predout,"region.id"), "\\.", simplify = TRUE)[,2]
        }
        newdata <-  imcims_wide[sample_id == sam,][!(ims_idx %in% predimsidx),]
      } else{
        newdata <- NULL
      }
      list(
        fit=fit1ls[[mzval]]$fitls[[sam]], 
        newdata=newdata,
        wmls=wmls_ls[[sam]],
        sam=sam
      )
    })

    # dflsls <- future_lapply(
    #   inp_ls, 
    #   future.seed = 123,
    #   future.globals = list(
    #     compress_fit = compress_fit,
    #     mzval = mzval,
    #     decompress_sarlm = decompress_sarlm
    #   ),
    #   \(inp){
    dflsls <- lapply(
      inp_ls, 
      \(inp){
      sam <- inp$sam
      if (compress_fit){
        inp$fit <- decompress_sarlm(inp$fit)
      }
      predout <- predict(inp$fit)
      if (!is.null(inp$newdata)){
        if(dim(inp$newdata)[1]>0 & !is.null(inp$wmls)){
          rownames(inp$newdata) <- paste0(inp$newdata$sample_id, ".", inp$newdata$ims_idx)
          if ("Sarlm" %in% met){
            predout <- predict(inp$fit,newdata=inp$newdata, listw = inp$wmls, pred.type="TC", all.data=TRUE)
          }else{
            predout <- predict(inp$fit,newdata=inp$newdata)
          }
        } else{
          if ("Sarlm" %in% met & do_out_of_sample){
            # to get consisted output
            predout <- predict(inp$fit,pred.type="trend")
          }
        }
      }

      residout <- resid(inp$fit)

      if ("Sarlm" %in% met){
        nn_ims_idx <- as.character(stringr::str_split(attr(predout, "region.id"),"\\.", simplify=TRUE)[,2])
        if (attr(predout, "pred.type") == "TC" | attr(predout, "pred.type") == "trend"){
          preddf <- tibble::tibble("sample_id"=sam, "ims_idx"=nn_ims_idx, !!paste0("pred_", mzval):=as.numeric(predout))
        } else{
          preddf <- tibble::tibble("sample_id"=sam, "ims_idx"=nn_ims_idx, !!paste0("pred_trend_", mzval):=attr(predout, "trend"), !!paste0("pred_signal_", mzval):=attr(predout, "signal"))
        }
        nn_ims_idx <- as.character(stringr::str_split(names(residout),"\\.", simplify=TRUE)[,2])
        residdf <- tibble::tibble("sample_id"=sam, "ims_idx"=nn_ims_idx, !!paste0("resid_", mzval):=residout)
        tmp2df <- dplyr::left_join(preddf, residdf, by=c("sample_id","ims_idx"))
      }else{
        nn_ims_idx <- as.character(stringr::str_split(names(predout),"\\.", simplify=TRUE)[,2])
        tmp2df <- tibble::tibble("sample_id"=sam, "ims_idx"=nn_ims_idx, !!paste0("pred_", mzval):=predout, !!paste0("resid_", mzval):=residout)
      }
      tmpdf <- dplyr::bind_rows(tmpdf, tmp2df)
      tmpdf
    })
    dfls[[mzval]] <- dflsls
  }
  dfls
}


get_impacts <- function(fitls) {
  lmimp_ls <- lapply(
    names(fitls),
    \(k){
      fitsum <- summary(fitls[[k]])
      tmpimp <- spatialreg::impacts(fitls[[k]], tr = fitls[[k]]$trs, R = 200, zero.policy = TRUE)
      tmpimpsum <- summary(tmpimp, short = TRUE, zstats = TRUE)
      data.frame(
        direct = tmpimpsum$res$direct[1],
        direct_se = tmpimpsum$semat[1, "Direct"],
        direct_pval = tmpimpsum$pzmat[1, "Direct"],
        indirect = tmpimpsum$res$indirect[1],
        indirect_se = tmpimpsum$semat[1, "Indirect"],
        indirect_pval = tmpimpsum$pzmat[1, "Indirect"],
        total = tmpimpsum$res$total[1],
        total_se = tmpimpsum$semat[1, "Total"],
        total_pval = tmpimpsum$pzmat[1, "Total"],
        sample_name = k,
        rho = fitsum$rho,
        rho_pval = fitsum$Wald1$p.value,
        LL = fitsum$LL,
        s2 = fitsum$s2,
        SSE = fitsum$SSE
      )
    }
  )
  lmimp_df <- do.call(Reduce, list(f = rbind, x = lmimp_ls))
  rownames(lmimp_df) <- lmimp_df$sample_name
  lmimp_df
}

# column_names=list(
#   sample_name="sample_name",
#   celltypes=unique(imcims_df_full$Cell_Type),
#   value="maldi_area_filled",
#   imsid="ims_idx",
#   coords=c("ims_x","ims_y"),
#   response=hepacol,
#   covariates=c("TMA_type","type","Sex")
# )
#
# fitls <- fit_first_layer(imcims_wide,
#                          dflc_comb_ls = NULL,
#                          column_names,
#                          wmls_ls=wmls_ls,
#                          method = "lagsarlm",
#                          n_workers = n_workers)
#



fit_second_layer <- function(
    coefdf,
    fitdf = NULL,
    column_names = list(
      celltypes = sort(unique(imcims_df_full$Cell_Type)),
      sample_covariates = c("type", "Sex"),
      random_effects = NULL
    ),
    type = c("lm", "rlm"),
    return_fit = FALSE) {
  type <- match.arg(type)

  present_celltypes <- sort(as.character(column_names$celltypes)[as.character(column_names$celltypes) %in% colnames(coefdf)])
  has_coefs <- !apply(coefdf[, present_celltypes], 2, function(x) all(is.na(x)))
  has_stderr <- fitdf |>
    dplyr::group_by(term) |>
    dplyr::filter(term %in% present_celltypes) |>
    dplyr::summarise(not_all_nan = sum(is.na(std.error))!=dplyr::n()) |> 
    dplyr::pull(not_all_nan) 
  actual_celltypes <- present_celltypes[has_coefs & has_stderr]
  ft <- lapply(actual_celltypes, \(ct){
    filt_coefdf <- coefdf[!is.na(coefdf[[ct]]), ]
    filt_fitdf <- fitdf[fitdf$sample_name %in% filt_coefdf$sample_name, ]
    is_interaction <- stringr::str_detect(ct,":")
    if (is_interaction){
      ct_new <- paste0("`",ct,"`")
      filt_coefdf[[ct_new]] <- filt_coefdf[[ct]]
      filt_fitdf$term[filt_fitdf$term == ct] <- ct_new
      ct <- ct_new
    }

    if ("n" %in% colnames(coefdf) & !is.null(fitdf)) {
      stopifnot("std.error" %in% colnames(fitdf))
      vars <- dplyr::filter(filt_fitdf, term == ct) |>
        dplyr::inner_join(filt_coefdf[, c("sample_name", "n")], by = "sample_name") |>
        dplyr::mutate(vars = (std.error * sqrt(n))^2) |>
        dplyr::pull(vars)
      wgs <- 1 / vars
    } else {
      vars <- NULL
      wgs <- NULL
    }
    used_covariates <- column_names$sample_covariates[sapply(column_names$sample_covariates, function(co) length(unique(filt_coefdf[[co]])) > 1)]
    if (length(used_covariates) == 0) {
      used_covariates <- NULL
    }
    if (is.null(column_names$random_effects)) {
      fl <- as.formula(paste0(ct, "~ 1 ", ifelse(is.null(used_covariates), "", "+"), paste0(used_covariates, collapse = "+")))
      filt_coefdf <- filt_coefdf[!is.nan(wgs),]
      wgs <- wgs[!is.nan(wgs)]
      if (type == "lm"){
        ft_single <- lm(fl, filt_coefdf, weights = wgs)
        if (return_fit){
          ft_single
        } else {
          cbind(data.frame(response = ct), broom::tidy(ft_single))
        }
      } else if (type == "rlm"){
        robfit <- tryCatch({
          robustbase::lmrob(fl, coefdf, weights=wgs, control=robustbase::lmrob.control("KS2014", maxit=2000, maxit.scale = 1000))
        }, error = function(e) {
          NULL
        })
        if (!is.null(robfit)){
          if (return_fit){
            robfit
          } else {
            cbind(data.frame(response = ct), broom::tidy(robfit))
          }
        } else {
          NULL
        }
      }
    } else {
      fl_fixed <- as.formula(paste0(
        ct, "~ 1 ", ifelse(is.null(used_covariates), "", "+"),
        paste0(used_covariates, collapse = "+")
      ))
      fl_random <- as.formula(paste0(
        "~",
        paste0(paste0("1|", column_names$random_effects, "", collapse = ""), collapse = "+")
      ))

      filt_coefdf$weights <- vars
      tryCatch(
        {
          ft_single <- nlme::lme(
            fixed = fl_fixed, random = fl_random,
            data = filt_coefdf, weights = ~weights,
            control = c(
              maxIter = 100, pnlsMaxIter = 10,
              msMaxIter = 1000, msMaxEval = 1000, niterEM = 100
            )
          )
          # ft_single <- lme4::lmer(fl, filt_coefdf,weights = wgs)
          if (return_fit){
            ft_single
          } else {
            cbind(data.frame(response = ct), suppressWarnings(broom.mixed::tidy(ft_single)))
          }
        },
        error = function(e) {
          data.frame(
            response = ct,
            effect = NA,
            group = NA,
            term = NA,
            estimate = NA,
            std.error = NA,
            df = NA,
            statistic = NA,
            p.value = NA
          )
        }
      )
    }
  })
  is_null <- sapply(ft, is.null)
  ft[is_null] <- NULL
  if(return_fit){
    ft
  } else{
    ft_tidy <- do.call(rbind, ft)
    list(fitdf = ft_tidy, fitls = ft)
  }
}


if (FALSE) {
  fl <- as.formula(paste0(
    "cbind(", paste0(column_names$celltypes, collapse = ","), ")~ 1 + ",
    paste0(column_names$sample_covariates, collapse = "+")
  ))
  serr <- dplyr::filter(fit1ls$fitdf, term == "B_Cell") |> dplyr::pull(std.error)
  ns <- imcims_wide |>
    dplyr::group_by(sample_name) |>
    dplyr::tally() |>
    dplyr::pull(n)
  sds <- serr * sqrt(ns)
  vars <- sds^2
  ftw <- lm(fl, coefdf, weights = 1 / vars)
  ft <- lm(fl, coefdf)
  summary(ftw)
  summary(ft)
  ft_tidy <- broom::tidy(ft)
  ftw_tidy <- broom::tidy(ftw)
  ft_tidy
  ftw_tidy
  simex::simex(ft, SIMEXvariable = "B_Cell", measurement.error = serr, asymptotic = FALSE)
}

# out <- fit_second_layer(
#   coefdf,
#   column_names=list(
#     celltypes=sort(unique(imcims_df_full$Cell_Type)),
#     covariates=c("type","Sex")
#   ),
#   type=c("lm","rlm")
# )
# print(out$fitdf,n=100)
