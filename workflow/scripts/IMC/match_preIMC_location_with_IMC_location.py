import sys,os
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "..","..","workflow","scripts","utils")))
import json
from shapely.geometry import shape
import numpy as np
import pandas as pd
import re
from utils import setNThreads, snakeMakeMock
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.input['preIMC_location'] = ""
    snakemake.input['IMC_location'] = ""
    snakemake.output['matching'] = ""
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# params
generic_input = snakemake.params["generic_input"]
# inputs
preIMC_geojson_file=snakemake.input['preIMC_location']
IMC_geojson_files=snakemake.input['IMC_location']

if preIMC_geojson_file == generic_input:
    header = "core,preIMC_location"
    with open(snakemake.output['matching'], "w") as f:
        f.write(header)
    sys.exit(0)

logging.info("Read preIMC geojson")
preIMC_geojson = json.load(open(preIMC_geojson_file, "r"))
if isinstance(preIMC_geojson, dict):
    preIMC_geojson = [preIMC_geojson]

logging.info("Read IMC geojson")
IMC_geojson = [ json.load(open(e, "r")) for e in IMC_geojson_files ]

# extract core names from file names
cores = [ re.search('_IMC_mask_on_preIMC_(.+?)\.geojson', s).group(1) for s in IMC_geojson_files]

#cores=snakemake.wildcards['core']

logging.info("preIMC geojson to shapes")
# polygons from preIMC_location
preIMC_geojson_polygons = [ shape(e['geometry']) for e in preIMC_geojson ]

logging.info("IMC geojson to shapes")
# polygons from IMC_location
IMC_geojson_polygons = [ shape(e['geometry']) for e in IMC_geojson]

logging.info("Detect overlaps")
# find overlaps
matchings=[]
names = [ gj['properties']['name'] for gj in preIMC_geojson]
for IMC_poly in IMC_geojson_polygons:
    inters = IMC_poly.intersection(preIMC_geojson_polygons)
    inters_areas = [ inter.area for inter in inters ]
    has_large_intersection = np.array(inters_areas)>=0.99*IMC_poly.area
    matching_name = np.array(names)[has_large_intersection]
    if len(matching_name)==1:
        matchings.append(matching_name[0])
    else:
        exit()

logging.info("Save csv")
df = pd.DataFrame({'core':cores,'preIMC_location':matchings})
df.to_csv(snakemake.output['matching'], index=False)

logging.info("Finished")