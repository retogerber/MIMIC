# Original script from Heath Patterson with adaptions from Reto Gerber
from typing import List, Optional, Union
import pickle
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from shapely import geometry, affinity
from tqdm import tqdm
from scipy.spatial.distance import cdist
import warnings
from wsireg.reg_shapes import RegShapes




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

    rs = RegShapes(cells_fp)
    # rs.shape_data.pop(0)
    # rs.shape_data_gj.pop(0)

    resolution_factor = ims_spacing / micro_spacing

    with h5py.File(imsml_coords_fp, "r") as f:
        padded = f["xy_padded"][:]

    # half pixel correction factor is because napari canvas is "center-pixel" oriented
    # rather than top-left corner
    micro_x = (padded[:, 0] * resolution_factor) - 0.5 * resolution_factor
    micro_y = (padded[:, 1] * resolution_factor) - 0.5 * resolution_factor

    cell_polygons = [geometry.Polygon(r["array"]) for r in rs.shape_data]
    assert len(cell_polygons) == len(cell_indices)
    pixel_boxes = [px_to_box(x, y, resolution_factor) for x, y in zip(micro_x, micro_y)]
    if ims_shrink_factor:
        pixel_boxes = [
            affinity.scale(px, xfact=ims_shrink_factor, yfact=ims_shrink_factor)
            for px in pixel_boxes
        ]

    centroids = np.stack([np.asarray(p.centroid.xy).transpose() for p in pixel_boxes])
    centroids = np.squeeze(centroids)

    cell_overlaps = []
    for idx, cell in enumerate(tqdm(cell_polygons)):
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
                        cell_overlaps.append((cell_idx, close, inter_area))

    cell_overlap_df = pd.DataFrame(
        cell_overlaps, columns=["cell_idx", "ims_idx", "overlap"]
    )

    cell_overlap_df.sort_values("cell_idx", inplace=True)

    return cell_overlap_df


# inputs
#imsml_coords_fp = "/home/retger/imc_to_ims_workflow/results/cirrhosis_TMA/data/IMS/cirrhosis_TMA-IMSML-coords.h5"
imsml_coords_fp = snakemake.input["imsml_coords_fp"]
#cell_indices_fp =  "/home/retger/imc_to_ims_workflow/results/cirrhosis_TMA/data/IMC_mask/Cirrhosis-TMA-5_New_Detector_006_transformed_cell_indices.pkl"
cell_indices_fp = snakemake.input["cell_indices"]

#cell_shapes_fp = "/home/retger/imc_to_ims_workflow/results/cirrhosis_TMA/data/IMC_mask/Cirrhosis-TMA-5_New_Detector_006_transformed_cell_masks.geojson"
cell_shapes_fp = snakemake.input["IMCmask_transformed"]

output_csv = snakemake.output["cell_overlaps"]


# data specific settings
ims_spacing = snakemake.params["IMS_pixel_size"]
micro_spacing = snakemake.params["IMC_pixel_size"]
ims_shrink_factor = snakemake.params["ims_shrink_factor"]




cell_indices = pickle.load(open(cell_indices_fp, "rb"))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cell_overlap_df = compute_intersections(
        cell_shapes_fp,
        imsml_coords_fp,
        ims_spacing,
        micro_spacing,
        cell_indices,
        ims_shrink_factor=ims_shrink_factor,
    )


cell_overlap_df.to_csv(
    output_csv,
    index=False,
)
