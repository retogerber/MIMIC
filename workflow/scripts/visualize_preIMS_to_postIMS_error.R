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

vis_type <- "preIMS_to_postIMS"
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

df$global_shift <- sqrt(colSums(rbind(df$global_x_shift**2, df$global_y_shift**2)))
df$sitk_global_shift <- sqrt(colSums(rbind(df$sitk_global_x_shift**2, df$sitk_global_y_shift**2)))


global_range <- range(unlist(lapply(colnames(df)[grepl("_shift",colnames(df))], function(x) df[[x]])))
global_range[1] <- global_range[1]-1 
global_range[2] <- global_range[2]+1

d <- highlight_key(df, ~sample)
base <- plot_ly(
    d, 
    text = ~sample,
    # hoverinfo = "text",
    type = "scatter",
    mode = "markers",
    color = ~project,
    colors = "Set1",
    marker = list(size = 10),
    showlegend = FALSE
)
fig <- subplot(
  add_markers(base, x = ~global_x_shift, y = ~sitk_global_x_shift,
    hovertemplate = paste0('<b>%{text}</b><br>X error Landmarks: %{y:.2f}<br>X error sitk: %{x:.2f}')) |>
    layout(
    xaxis = list(range=global_range, title = "X translation"),
    yaxis = list(range=global_range, title = "X translation sitk")
  ),
  add_markers(base, x = ~global_y_shift, y = ~sitk_global_y_shift,
    hovertemplate = paste0('<b>%{text}</b><br>Y error Landmarks: %{y:.2f}<br>Y error sitk: %{x:.2f}')) |>
    layout(
    xaxis = list(range=global_range, title = "Y translation"),
    yaxis = list(range=global_range, title = "Y translation sitk")
  ),
  add_markers(base, x = ~global_shift, y = ~sitk_global_shift,
    hovertemplate = paste0('<b>%{text}</b><br>Error Landmarks: %{y:.2f}<br>Error sitk: %{x:.2f}')) |>
    layout(
    xaxis = list(title = "translation"),
    yaxis = list(title = "translation sitk")
  )
) |> 
  add_annotations(
    text = "X translation",
    x = 0.15,
    y = 1,
    yref = "paper",
    xref = "paper",
    yanchor = "bottom",
    showarrow = FALSE,
    font = list(size = 15)
  ) |> 
  add_annotations(
    text = "Y translation",
    x = 0.5,
    y = 1,
    yref = "paper",
    xref = "paper",
    yanchor = "bottom",
    showarrow = FALSE,
    font = list(size = 15)
  ) |> 
  add_annotations(
    text = "Error",
    x = 0.85,
    y = 1,
    yref = "paper",
    xref = "paper",
    yanchor = "bottom",
    showarrow = FALSE,
    font = list(size = 15)
  ) |>
    layout(
    xaxis = list(title = "Landmarks"),
    yaxis = list(title = "sitk"),
    margin = list(t=50, b=150, l=50, r=50)
  ) |>
  highlight(on = "plotly_hover", off = "plotly_doubleclick")

htmlwidgets::saveWidget(partial_bundle(fig),output_html_file)
