# Original script from Heath Patterson with adaptions from Reto Gerber
from typing import List, Optional, Union
import pickle
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from shapely import geometry, affinity
from scipy.spatial.distance import cdist
import warnings
from wsireg.reg_shapes import RegShapes
import sys,os
import logging, traceback
logging.basicConfig(filename=snakemake.log["stdout"],
                    level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    )
from logging_utils import handle_exception, StreamToLogger
sys.excepthook = handle_exception
sys.stdout = StreamToLogger(logging.getLogger(),logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger(),logging.ERROR)

logging.info("Start")


def px_to_box(x: Union[int, float], y: Union[int, float], px_width: Union[int, float]):
    """IMS pixel to shapely geometry representation."""
    box = geometry.box(x, y, x + px_width, y + px_width)
    return box


def compute_intersections(
    cells_fp: Union[Path, str],
    imsml_coords_fp: Union[Path, str],
    ims_spacing: Union[int, float],
    micro_spacing: Union[int, float],
    cell_indices: List[int],
    ims_shrink_factor: Optional[float] = None,
):
    """Compute intersections in multi-scale spatial domain using polygons"""

    logging.info("Read cells to shapes")
    rs = RegShapes(cells_fp)
    # rs.shape_data.pop(0)
    # rs.shape_data_gj.pop(0)

    # resolution_factor = ims_spacing / micro_spacing

    logging.info("Read h5 coords file")
    with h5py.File(imsml_coords_fp, "r") as f:
        # if in imsmicrolink IMS was the target
        if "xy_micro_physical" in [key for key, val in f.items()]:
            xy_micro_physical = f["xy_micro_physical"][:]

            micro_x = xy_micro_physical[:,0]
            micro_y = xy_micro_physical[:,1]
        # if the microscopy image was the target
        else:
            padded = f["xy_padded"][:]
            
            micro_x = (padded[:, 0] * ims_spacing)
            micro_y = (padded[:, 1] * ims_spacing)

    logging.info("Create cell polygons")
    cell_polygons = [geometry.Polygon(r["array"]) for r in rs.shape_data]
    assert len(cell_polygons) == len(cell_indices)
    logging.info("Create IMS pixel polygons")
    pixel_boxes = [px_to_box(x, y, ims_spacing*ims_shrink_factor) for x, y in zip(micro_x, micro_y)]
    if ims_shrink_factor:
        pixel_boxes = [
            affinity.scale(px, xfact=ims_shrink_factor, yfact=ims_shrink_factor)
            for px in pixel_boxes
        ]

    logging.info("Calculate centroids")
    centroids = np.stack([np.asarray(p.centroid.xy).transpose() for p in pixel_boxes])
    centroids = np.squeeze(centroids)

    logging.info("Calculate overlaps")
    cell_overlaps = []
    for idx, cell in enumerate(cell_polygons):
        cell_centroid = np.asarray(cell.centroid.xy).transpose()
        dist_from_cell = cdist(cell_centroid, centroids)
        close_pixels = dist_from_cell < 100
        cell_idx = cell_indices[idx]
        if np.any(close_pixels):
            close_pixel_indices = np.where(close_pixels)[1]
            for close in close_pixel_indices:
                if cell.is_valid:
                    inter_area = cell.intersection(pixel_boxes[close]).area / cell.area
                    if inter_area > 0:
                        cell_overlaps.append((cell_idx, close, inter_area, cell.area))

    cell_overlap_df = pd.DataFrame(
        cell_overlaps, columns=["cell_idx", "ims_idx", "overlap", "cell_area"]
    )

    cell_overlap_df.sort_values("cell_idx", inplace=True)

    return cell_overlap_df


# inputs

imsml_coords_fp = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/IMS/NASH_HCC_TMA_011-IMSML-coords.h5"
# imsml_coords_fp = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMS/postIMS_to_IMS_test_split_pre-IMSML-coords.h5"
# imsml_coords_fp = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMS/postIMS_to_IMS_test_split_pre-Cirrhosis-TMA-5_New_Detector_002-IMSML-coords.h5"
#imsml_coords_fp = "/home/retger/imc_to_ims_workflow/results/cirrhosis_TMA/data/IMS/cirrhosis_TMA-IMSML-coords.h5"
imsml_coords_fp = snakemake.input["imsml_coords_fp"]
cell_indices_fp = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/IMC_mask/NASH_HCC_TMA-2_011_transformed_on_postIMS_cell_indices.pkl"
# cell_indices_fp = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMC_mask/Cirrhosis-TMA-5_New_Detector_001_transformed_on_postIMS_cell_indices.pkl"
#cell_indices_fp =  "/home/retger/imc_to_ims_workflow/results/cirrhosis_TMA/data/IMC_mask/Cirrhosis-TMA-5_New_Detector_006_transformed_cell_indices.pkl"
cell_indices_fp = snakemake.input["cell_indices"]
cell_shapes_fp = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/IMC_mask/NASH_HCC_TMA-2_011_transformed_on_postIMS_cell_masks.geojson"
# cell_shapes_fp = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMC_mask/Cirrhosis-TMA-5_New_Detector_001_transformed_on_postIMS_cell_masks.geojson"
#cell_shapes_fp = "/home/retger/imc_to_ims_workflow/results/cirrhosis_TMA/data/IMC_mask/Cirrhosis-TMA-5_New_Detector_006_transformed_cell_masks.geojson"
cell_shapes_fp = snakemake.input["IMCmask_transformed"]

output_csv = snakemake.output["cell_overlaps"]


# data specific settings
# ims_spacing=30
ims_spacing = snakemake.params["IMS_pixelsize"]
# micro_spacing = 0.22537
micro_spacing = snakemake.params["IMC_pixelsize"]
# ims_shrink_factor = 0.8
ims_shrink_factor = snakemake.params["IMS_shrink_factor"]




logging.info("Read pickle")
cell_indices = pickle.load(open(cell_indices_fp, "rb"))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    logging.info("Compute intersections")
    cell_overlap_df = compute_intersections(
        cell_shapes_fp,
        imsml_coords_fp,
        ims_spacing,
        micro_spacing,
        cell_indices,
        ims_shrink_factor=ims_shrink_factor,
    )

assert(len(cell_overlap_df['cell_idx'].to_list())>0)

logging.info("Save to csv")
cell_overlap_df.to_csv(
    output_csv,
    index=False,
)

logging.info("Finished")