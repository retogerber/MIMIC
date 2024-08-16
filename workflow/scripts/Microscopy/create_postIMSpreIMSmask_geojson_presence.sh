#!/bin/bash

while getopts i:o: flag
do
    case "${flag}" in
        i) input_dir=${OPTARG};;
        o) output_file=${OPTARG};;
    esac
done
shift "$(( OPTIND - 1 ))"

if [ -z "$input_dir" ]; then
    echo "Variable input_idr (flag -c) is empty!" >&2
    exit 1
fi

# check if file exists
if [ ! -d "$input_dir" ]; then
    echo "File '$input_dir' does not exist" >&2
    exit 1
fi

if [ -z "$output_file" ]; then
    echo "Variable output_file (flag -o) is empty!" >&2
    exit 1
fi
echo $output_file
echo $input_dir

echo "type,exists" > $output_file

# check if file exists
postIMS_location=$( find $input_dir -type f -name *TMA_mask_on_postIMS.geojson )
if [ ! -z $postIMS_location ]; then
    echo "postIMS,1" >> $output_file
else
    echo "postIMS,0" >> $output_file
fi

# check if file exists
preIMS_location=$( find $input_dir -type f -name *TMA_mask_on_preIMS.geojson )
if [ ! -z $preIMS_location ]; then
    echo "preIMS,1" >> $output_file
else
    echo "preIMS,0" >> $output_file
fi
