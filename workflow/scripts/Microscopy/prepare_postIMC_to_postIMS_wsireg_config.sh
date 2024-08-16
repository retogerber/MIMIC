#!/bin/bash

template_file=config/template_postIMC_to_postIMS-wsireg-config.yaml
while getopts p:a:b:c:d:e:f:s:t:o: flag
do
    case "${flag}" in
        p) project_name=${OPTARG};;
        a) postIMS_file=${OPTARG};;
        b) preIMS_file=${OPTARG};;
        c) preIMC_file=${OPTARG};;
        d) postIMC_file=${OPTARG};;
        e) postIMSmask_file=${OPTARG};;
        f) preIMSmask_file=${OPTARG};;
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
# check if file exists
if [ ! -f "$postIMSmask_file" ]; then
    echo "File '$postIMSmask_file' does not exist" >&2
    exit 1
fi
# check if file exists
if [ ! -f "$preIMSmask_file" ]; then
    echo "File '$preIMSmask_file' does not exist" >&2
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

echo "copy file"
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

if [ "$postIMSmask_file" != "$postIMS_file" ]; then
    echo "add postIMS mask"
    echo "change mask path in config file"
    postIMSmask=$postIMSmask_file results/Misc/yq -i '.modalities.postIMS.mask = strenv(postIMSmask)' "${out_config_file}"
    results/Misc/yq -i '.modalities.postIMS.preprocessing.use_mask = true' "${out_config_file}"
fi

if [ "$preIMSmask_file" != "$preIMS_file" ]; then
    echo "add preIMS mask"
    echo "change mask path in config file"
    preIMSmask=$preIMSmask_file results/Misc/yq -i '.modalities.preIMS.mask = strenv(preIMSmask)' "${out_config_file}"
    results/Misc/yq -i '.modalities.preIMS.preprocessing.use_mask = true' "${out_config_file}"
fi


# set nonlinear registration between preIMC and preIMS if the two files are the same
if [ "${md5preIMC[0]}" == "${md5preIMS[0]}" ]; then
    sed -i "/ - nl/d" "${out_config_file}"
fi


