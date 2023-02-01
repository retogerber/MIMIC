#!/bin/bash

workdir="."

while getopts p: flag
do
    case "${flag}" in
        p) project_name=${OPTARG};;
    esac
done
shift "$(( OPTIND - 1 ))"


if [ -z "$project_name" ]; then
    echo "Project name (-p) has to be given" >&2
    exit 1
fi

mkdir -p results/${project_name}/{data,registrations}
mkdir -p results/${project_name}/data/{IMS,postIMS,preIMS,postIMC,preIMC,IMC,IMC_mask,IMC_location,cell_overlap}
mkdir -p results/${project_name}/registrations/{postIMC_to_postIMS,IMC_to_preIMC,IMC_to_IMS}



