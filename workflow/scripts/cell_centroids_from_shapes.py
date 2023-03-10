import pickle
from typing import List, Optional, Union
from pathlib import Path
from wsireg.reg_shapes import RegShapes
from shapely import geometry
import numpy as np
import pandas as pd

def compute_cell_centroids(
    cells_fp: Union[Path, str],
    cell_indices: List[int],
):
    """Compute cell centroids from shapes"""

    rs = RegShapes(cell_shapes_fp)

    cell_polygons = [geometry.Polygon(r["array"]) for r in rs.shape_data]
    assert len(cell_polygons) == len(cell_indices)

    #print([np.asarray(p.centroid.xy) for p in cell_polygons])
    centroids = np.stack([np.asarray(p.centroid.xy).transpose() for p in cell_polygons])
    centroids = np.squeeze(centroids)

    centroids_df = pd.DataFrame(
        centroids, columns=["x", "y"]
    )
    centroids_df['cell_idx'] = cell_indices
    centroids_df.sort_values("cell_idx", inplace=True)
    return centroids_df



cell_shapes_fp = snakemake.input["IMCmask_transformed"]
cell_indices_fp = snakemake.input["cell_indices"]
#cell_indices_fp =  "/home/retger/imc_to_ims_workflow/results/cirrhosis_TMA/data/IMC_mask/Cirrhosis-TMA-5_New_Detector_006_transformed_cell_indices.pkl"
#cell_shapes_fp = "/home/retger/imc_to_ims_workflow/results/cirrhosis_TMA/data/IMC_mask/Cirrhosis-TMA-5_New_Detector_006_transformed_cell_masks.geojson"

output_csv = snakemake.output["cell_centroids"]

cell_indices = pickle.load(open(cell_indices_fp, "rb"))
cell_centroids_df = compute_cell_centroids(
    cell_shapes_fp,
    cell_indices
)
cell_centroids_df.to_csv(
    output_csv,
    index=False,
)
