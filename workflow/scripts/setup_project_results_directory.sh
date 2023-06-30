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

mkdir -p results/Misc
mkdir -p results/${project_name}/{data,registrations}
mkdir -p results/${project_name}/data/{IMS,postIMS,preIMS,postIMC,preIMC,IMC,IMC_mask,IMC_summary_panel,IMC_location,cell_overlap,preIMS_location,preIMC_location,preIMC_location_combined,preIMS_location_combined, registration_metric}
mkdir -p results/${project_name}/registrations/{postIMC_to_postIMS,IMC_to_preIMC,IMC_to_IMS,preIMC_to_preIMS}



