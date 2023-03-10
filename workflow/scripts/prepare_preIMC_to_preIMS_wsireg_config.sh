#!/bin/bash

template_file=config/template_postIMC_to_postIMS-wsireg-config.yaml
while getopts p:a:b:c:d:s:t:o: flag
do
    case "${flag}" in
        p) project_name=${OPTARG};;
        a) preIMS_file=${OPTARG};;
        b) preIMS_mask_file=${OPTARG};;
        c) preIMC_file=${OPTARG};;
        d) preIMC_mask_file=${OPTARG};;
        s) microscopy_pixelsize=${OPTARG};;
        t) template_file=${OPTARG};;
        o) outdir=${OPTARG};;
    esac
done
shift "$(( OPTIND - 1 ))"

if [ -z "$project_name" ]; then
    echo "Variable project_name (flag -p) is empty!" >&2
    exit 1
fi
# check if file exists
if [ ! -f "$preIMS_file" ]; then
    echo "File '$preIMS_file' does not exist" >&2
    exit 1
fi
# check if file exists
if [ ! -f "$preIMS_mask_file" ]; then
    echo "File '$preIMS_mask_file' does not exist" >&2
    exit 1
fi
# check if file exists
if [ ! -f "$preIMC_file" ]; then
    echo "File '$preIMC_file' does not exist" >&2
    exit 1
fi
# check if file exists
if [ ! -f "$preIMC_mask_file" ]; then
    echo "File '$preIMC_mask_file' does not exist" >&2
    exit 1
fi
#file_basename="${file%.*}"
#file_extension="${file##*.}"

if [ -z "$microscopy_pixelsize" ]; then
    echo "Variable microscopy_pixelsize (flag -s) is empty!" >&2
    exit 1
fi
if [ ! -f "$template_file" ]; then
    echo "File '$template_file' does not exist" >&2
    exit 1
fi

if [ ! -d "$outdir" ]; then
    echo "Directory ${outdir} does not exist!" >&2
    mkdir $outdir
    #exit 1
fi

out_config_file="${outdir}/${project_name}-wsireg-config.yaml"

cp ${template_file} ${out_config_file}
sed -i "s,SEDVAR_PROJECT_NAME,${project_name},g" "${out_config_file}"
sed -i "s,SEDVAR_PREIMS_FILE,${preIMS_file},g" "${out_config_file}"
sed -i "s,SEDVAR_PREIMS_MASK_FILE,${preIMS_mask_file},g" "${out_config_file}"
sed -i "s,SEDVAR_PREIMC_FILE,${preIMC_file},g" "${out_config_file}"
sed -i "s,SEDVAR_PREIMC_MASK_FILE,${preIMC_mask_file},g" "${out_config_file}"
sed -i "s,SEDVAR_OUTPUT_DIR,${outdir},g" "${out_config_file}"
sed -i "s,SEDVAR_MICROSCOPY_PIXELSIZE,${microscopy_pixelsize},g" "${out_config_file}"




