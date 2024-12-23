#!/bin/bash

while getopts a:b:g:o: flag
do
    case "${flag}" in
        a) postIMS_location=${OPTARG};;
        b) preIMS_location=${OPTARG};;
        g) generic_file=${OPTARG};;
        o) output_file=${OPTARG};;
    esac
done
shift "$(( OPTIND - 1 ))"

if [ -z "$postIMS_location" ]; then
    echo "Variable postIMS_location (flag -a) is empty!" >&2
    exit 1
fi

if [ -z "$preIMS_location" ]; then
    echo "Variable preIMS_location (flag -b) is empty!" >&2
    exit 1
fi

if [ -z "$generic_file" ]; then
    echo "Variable generic_file (flag -g) is empty!" >&2
    exit 1
fi

if [ -z "$output_file" ]; then
    echo "Variable output_file (flag -o) is empty!" >&2
    exit 1
fi
echo "postIMS_location: ${postIMS_location}"
echo "preIMS_location: ${preIMS_location}"
echo "generic_file: ${generic_file}"
echo "output_file: ${output_file}"

echo "type,exists" > $output_file

# check if file exists
#postIMS_location=$( find $input_dir -type f -name *TMA_mask_on_postIMS.geojson )
if [ -f $postIMS_location ] && [ $postIMS_location != $generic_file ]; then
    echo "postIMS,1" >> $output_file
else
    echo "postIMS,0" >> $output_file
fi

# check if file exists
#preIMS_location=$( find $input_dir -type f -name *TMA_mask_on_preIMS.geojson )
if [ -f $preIMS_location ] && [ $preIMS_location != $generic_file ]; then
    echo "preIMS,1" >> $output_file
else
    echo "preIMS,0" >> $output_file
fi
