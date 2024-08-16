# logging
stdlog <- file(snakemake@log[["stdout"]], open="wt")
sink(stdlog, type = "output")
sink(stdlog, type = "message")

print("Load libraries")
# Load the required libraries
library(plotly)


print("Load data")
# input_csv_files <- "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/registration_metric/test_split_ims_reg_metrics_combined.csv"
# input_csv_files <- c("/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/registration_metric/test_combined_reg_metrics_combined.csv","/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/test_split_pre_reg_metrics_combined.csv","/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/registration_metric/test_split_ims_reg_metrics_combined.csv")
input_csv_files <- snakemake@input[["registration_metrics_combined"]]

print(paste0("input_csv_files: ",input_csv_files))

# output_html_file <- "/home/retger/Downloads/test.html"
output_html_file <- snakemake@output[["vis_plot"]]

# vis_type <- "IMS_to_postIMS"
vis_type <- snakemake@params[["vis_type"]]

stopifnot(vis_type %in% c("IMS_to_postIMS", "postIMC_to_preIMC", "preIMC_to_preIMS", "preIMS_to_postIMS"))

# colname_reg <- paste0(vis_type, "_part_inmask_")
# colname_reg <- paste0(vis_type, "_inmask_")
colname_reg <- paste0(vis_type, "_")


dfls <- lapply(input_csv_files, function(x) {
    df <- read.csv(x)
    df$project <- stringr::str_replace(basename(x),"_reg_metrics_combined.csv","")
    df <- df[,grepl(colname_reg, colnames(df)) | colnames(df) %in% c("sample", "project")]

    colnames(df) <- stringr::str_replace(colnames(df), colname_reg, "")
    colnames(df) <- stringr::str_replace(colnames(df),"quantile","Q" )
    df
})
df <- Reduce(rbind, dfls)

print("Create plot")
x_axis_var_names <- colnames(df)[grepl("error", colnames(df))]
create_buttons <- function(df, x_axis_var_names) {
  lapply(
    x_axis_var_names,
    FUN = function(var_name, df) {
      button <- list(
        method = 'restyle',
        args = list('x', list(df[, var_name])),
        label = stringr::str_to_title(stringr::str_replace(var_name,"_"," "))
      )
    },
    df
  )
}

global_x_range <- range(unlist(lapply(x_axis_var_names, function(x) df[[x]])))
global_x_range[1] <- 0 
global_x_range[2] <- global_x_range[2]+2

if (vis_type == "IMS_to_postIMS") {
  df$completeness <- df$n_points/df$n_points_total
  y_axis_colname <- "completeness"
  y_axis_name <- "Completeness"
} else {
  y_axis_colname <- "n_points_total"
  y_axis_name <- "Number of Landmarks"
}

dfls <- split(df, df$project)
figls <- lapply(dfls, function(df){
    plotly::plot_ly(
        data = df,
        x = ~mean_error,
        y = dplyr::quo(!!dplyr::sym(y_axis_colname)),
        text = ~sample,
        # hoverinfo = "text",
        hovertemplate = paste0('<b>%{text}</b><br>',y_axis_name,': %{y:.2f}<br>Error: %{x:.2f}'),
        type = "scatter",
        mode = "markers",
        color = ~project,
        colors = "Set1",
        marker = list(size = 10)
    ) |>
         layout(
             title = paste0(stringr::str_replace_all(vis_type,"_"," "),":   Error vs ",y_axis_name),
             xaxis = list(domain = c(0.1, 1), range=global_x_range, title = ""),
             yaxis = list(title = y_axis_name),
             updatemenus = list(
                 list(
                     y = -0.05,
                     x = 0.5,
                     buttons = create_buttons(df, x_axis_var_names)
                 )
             ))
})
fig <- subplot(figls, nrows = 1, shareX = TRUE, shareY = TRUE) |>
    layout(margin = list(t=50, b=150, l=50, r=50))

htmlwidgets::saveWidget(partial_bundle(fig),output_html_file)
