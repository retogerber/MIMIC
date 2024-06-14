#!/bin/bash

template_file=config/template_postIMC_to_postIMS-wsireg-config.yaml
while getopts p:a:b:c:d:m:s:t:o: flag
do
    case "${flag}" in
        p) project_name=${OPTARG};;
        a) postIMS_file=${OPTARG};;
        b) preIMS_file=${OPTARG};;
        c) preIMC_file=${OPTARG};;
        d) postIMC_file=${OPTARG};;
        m) postIMSpreIMSmask=${OPTARG};;
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
if [ ! -f "$postIMS_file" ]; then
    echo "File '$postIMS_file' does not exist" >&2
    exit 1
fi
# check if file exists
if [ ! -f "$preIMS_file" ]; then
    echo "File '$preIMS_file' does not exist" >&2
    exit 1
fi
# check if file exists
if [ ! -f "$preIMC_file" ]; then
    echo "File '$preIMC_file' does not exist" >&2
    exit 1
fi
# check if file exists
if [ ! -f "$postIMC_file" ]; then
    echo "File '$postIMC_file' does not exist" >&2
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
    exit 1
fi

out_config_file="${outdir}/${project_name}-wsireg-config.yaml"

cp ${template_file} ${out_config_file}
sed -i "s,SEDVAR_PROJECT_NAME,${project_name},g" "${out_config_file}"
sed -i "s,SEDVAR_POSTIMS_FILE,${postIMS_file},g" "${out_config_file}"
sed -i "s,SEDVAR_PREIMS_FILE,${preIMS_file},g" "${out_config_file}"
sed -i "s,SEDVAR_PREIMC_FILE,${preIMC_file},g" "${out_config_file}"
sed -i "s,SEDVAR_POSTIMC_FILE,${postIMC_file},g" "${out_config_file}"
sed -i "s,SEDVAR_OUTPUT_DIR,${outdir},g" "${out_config_file}"
sed -i "s,SEDVAR_MICROSCOPY_PIXELSIZE,${microscopy_pixelsize},g" "${out_config_file}"

# if preIMC and preIMS are the same, do linear registration only 
md5preIMC=($(md5sum ${preIMC_file} ))
md5preIMS=($(md5sum ${preIMS_file} ))

echo $postIMSpreIMSmask
if [ "$postIMSpreIMSmask" = "True" ]; then
    echo "add mask"
    echo "change mask path in config file"
    out_mask_file="${postIMS_file/.ome.tiff/_mask_for_reg.ome.tiff}"
    echo $out_mask_file
    out_mask_file=$out_mask_file results/Misc/yq -i '.modalities.postIMS.mask = strenv(out_mask_file)' "${out_config_file}"

    echo "change mask path in config file"
    out_mask_file="${preIMS_file/.ome.tiff/_mask_for_reg.ome.tiff}"
    echo $out_mask_file
    out_mask_file=$out_mask_file results/Misc/yq -i '.modalities.preIMS.mask = strenv(out_mask_file)' "${out_config_file}"
fi


if [ "${md5preIMC[0]}" == "${md5preIMS[0]}" ]; then
    sed -i "/ - nl/d" "${out_config_file}"
fi


