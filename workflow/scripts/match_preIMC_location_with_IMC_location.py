import json
from shapely.geometry import shape
import numpy as np
import pandas as pd
import re
import sys,os
import logging, traceback
logging.basicConfig(filename=snakemake.log["stdout"],
                    level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    )
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

#preIMC_geojson_file="/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/preIMC_location/NASH_HCC_TMA_reg_mask_on_preIMC.geojson"
preIMC_geojson_file=snakemake.input['preIMC_location']
logging.info("Read preIMC geojson")
preIMC_geojson = json.load(open(preIMC_geojson_file, "r"))

#IMC_geojson_filebase="/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/IMC_location/"
#cores = ["A1","A2","A3"]
#IMC_geojson_files = [IMC_geojson_filebase+f'NASH_HCC_TMA_IMC_mask_on_preIMC_{core}.geojson' for core in cores]
IMC_geojson_files=snakemake.input['IMC_location']
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
IMC_geojson_polygons = [ shape(e[0]['geometry']) for e in IMC_geojson]

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