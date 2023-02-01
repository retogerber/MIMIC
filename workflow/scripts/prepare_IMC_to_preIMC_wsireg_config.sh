#!/bin/bash

while getopts p:f:m:a:s:i:t:o: flag
do
    case "${flag}" in
        p) project_name=${OPTARG};;
        f) preIMC_file=${OPTARG};;
        m) preIMC_mask=${OPTARG};;
        a) IMC_aggr=${OPTARG};;
        s) microscopy_pixelsize=${OPTARG};;
        i) IMC_pixelsize=${OPTARG};;
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
if [ ! -f "$preIMC_file" ]; then
    echo "File '$preIMC_file' does not exist" >&2
    exit 1
fi
#file_basename="${file%.*}"
#file_extension="${file##*.}"

if [ ! -f "$preIMC_mask" ]; then
    echo "File '$preIMC_mask' does not exist" >&2
    exit 1
fi
if [ ! -f "$IMC_aggr" ]; then
    echo "File '$IMC_aggr' does not exist" >&2
    exit 1
fi
if [ -z "$microscopy_pixelsize" ]; then
    echo "Microscopy pixelsize (flag -s) not given!" >&2
    exit 1
fi
if [ -z "$IMC_pixelsize" ]; then
    echo "Microscopy pixelsize (flag -s) not given!" >&2
    exit 1
fi
if [ ! -f "$template_file" ]; then
    echo "File '$template_file' does not exist" >&2
    exit 1
fi


if [ -z "$outdir" ]; then
    outdir="${template_file%/*}"
    echo "Setting output directory to ${outdir}" >&2
    exit 1
fi

out_config_file="${outdir}/${project_name}-wsireg-config.yaml"
echo $out_config_file

#SEDVAR_PROJECT_NAME
#SEDVAR_PREIMC_FILE
#SEDVAR_PREIMC_MASK
#SEDVAR_IMC_AGGR_FILE
#SEDVAR_PREIMC_RESOLUTION

cp ${template_file} ${out_config_file}
sed -i "s,SEDVAR_PROJECT_NAME,${project_name},g" "${out_config_file}"
sed -i "s,SEDVAR_OUTDIR,${outdir},g" "${out_config_file}"
sed -i "s,SEDVAR_PREIMC_FILE,${preIMC_file},g" "${out_config_file}"
sed -i "s,SEDVAR_PREIMC_MASK,${preIMC_mask},g" "${out_config_file}"
sed -i "s,SEDVAR_IMC_AGGR_FILE,${IMC_aggr},g" "${out_config_file}"
sed -i "s,SEDVAR_PREIMC_RESOLUTION,${microscopy_pixelsize},g" "${out_config_file}"
sed -i "s,SEDVAR_IMC_RESOLUTION,${IMC_pixelsize},g" "${out_config_file}"
