import pickle
from typing import List, Optional, Union
from pathlib import Path
from wsireg.reg_shapes import RegShapes
from shapely import geometry
import numpy as np
import pandas as pd
from utils import setNThreads, snakeMakeMock
import sys,os
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.input["IMCmask_transformed"] = ""
    snakemake.input["cell_indices"] = ""
    snakemake.output["cell_centroids"] = ""
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

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

# inputs
cell_shapes_fp = snakemake.input["IMCmask_transformed"]
cell_indices_fp = snakemake.input["cell_indices"]

# outputs
output_csv = snakemake.output["cell_centroids"]

logging.info("Read pickle")
cell_indices = pickle.load(open(cell_indices_fp, "rb"))
logging.info("Compute centroids")
cell_centroids_df = compute_cell_centroids(
    cell_shapes_fp,
    cell_indices
)
logging.info("Save centroids")
cell_centroids_df.to_csv(
    output_csv,
    index=False,
)

logging.info("Finished")