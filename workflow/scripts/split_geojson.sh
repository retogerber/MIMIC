#!/bin/bash


while getopts f:s:c: flag
do
    case "${flag}" in
        f) file=${OPTARG};;
        s) scale=${OPTARG};;
        c) core_name=${OPTARG};;
    esac
done
shift "$(( OPTIND - 1 ))"

# check if file exists
if [ ! -f "$file" ]; then
        echo "File '$file' does not exist" >&2
        exit 1
fi
file_basename="${file%.*}"
file_extension="${file##*.}"

# check if file is geojson
if [ "$file_extension" != "geojson" ]; then
    echo "Extension of file must be geojson!"
    exit 1
fi
# check if file is geojson
if [ -z "$scale" ]; then
    echo "Microscopy pixel size (flag -s) must be given!"
    exit 1
fi

if [ -z "$core_name" ]; then
    echo "Core name (flag -c) must be given!"
    exit 1
fi


# read in objects as array
array=($(jq '.[].properties.name'  $file))
# loop through objects
#for ele in ${array[@]}
for ele in ${core_name}
do
    
    ele="${ele%\"}"
    ele="${ele#\"}"
    # extract element and multiply with scale
    jq --arg ele "$ele" --argjson scale "$scale" '.[] | select(.properties.name==$ele) | .geometry.coordinates[][][]*=$scale'  $file > "${file_basename}_${ele}.geojson"
done
