#!/bin/bash

while getopts c:s:o: flag
do
    case "${flag}" in
        c) preIMC_location_dir=${OPTARG};;
        s) preIMS_location_dir=${OPTARG};;
        o) output_file=${OPTARG};;
    esac
done
shift "$(( OPTIND - 1 ))"

if [ -z "$preIMC_location_dir" ]; then
    echo "Variable preIMC_location_dir (flag -c) is empty!" >&2
    exit 1
fi

# check if file exists
if [ ! -d "$preIMC_location_dir" ]; then
    echo "File '$preIMC_location_dir' does not exist" >&2
    exit 1
fi

if [ -z "$preIMS_location_dir" ]; then
    echo "Variable preIMS_location_dir (flag -s) is empty!" >&2
    exit 1
fi

# check if file exists
if [ ! -d "$preIMS_location_dir" ]; then
    echo "File '$preIMS_location_dir' does not exist" >&2
    exit 1
fi

if [ -z "$output_file" ]; then
    echo "Variable output_file (flag -o) is empty!" >&2
    exit 1
fi


preIMC_out=( )
preIMC_files=( ${preIMC_location_dir}/*_reg_mask_on_preIMC.geojson )
for file in ${preIMC_files[@]}
do
    # if file does not exist, i.e. no files found create empty output and exit
    if [ ! -f "$file" ]; then
        #echo "File '$file' does not exist" >&2
        touch $output_file
        exit 0
    fi
    array=($(jq '.[].properties.name'  $file))
    preIMC_out+=(${array[@]})
done

preIMS_out=( )
preIMS_files=( ${preIMS_location_dir}/*_reg_mask_on_preIMS.geojson )
echo preIMS_files
echo $preIMS_files
for file in ${preIMS_files[@]}
do
    # if file does not exist, i.e. no files found create empty output and exit
    if [ ! -f "$file" ]; then
        #echo "File '$file' does not exist" >&2
        touch $output_file
        exit 0
    fi
    array=($(jq '.[].properties.name'  $file))
    preIMS_out+=(${array[@]})
done

# https://stackoverflow.com/questions/2312762/compare-difference-of-two-arrays-in-bash
diff_elems=(`echo ${preIMS_out[@]} ${preIMC_out[@]} | tr ' ' '\n' | sort | uniq -u `)
echo $diff_elems
if [ ! -z "$diff_elems" ]; then
    echo "Not all elements exist! Differences: ${diff_elems[@]}" >&2
    exit 1
fi


# save to file
echo ${preIMS_out[@]//\"/} | tr ' ' '\n' | sort > $output_file 
