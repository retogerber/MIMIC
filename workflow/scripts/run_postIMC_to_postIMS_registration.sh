#!/bin/bash

while getopts a:b:c: flag
do
    case "${flag}" in
        a) wsireg_config=${OPTARG};;
        b) preIMC_to_postIMS_transform=${OPTARG};;
        c) postIMC_to_postIMS_transform=${OPTARG};;
    esac
done
shift "$(( OPTIND - 1 ))"

if [ -z "$wsireg_config" ]; then
    echo "Variable wsireg_config (flag -a) is empty!" >&2
    exit 1
fi
# check if file exists
if [ ! -f "$wsireg_config" ]; then
    echo "File '$wsireg_config' does not exist" >&2
    exit 1
fi

if [ -z "$preIMC_to_postIMS_transform" ]; then
    echo "Variable preIMC_to_postIMS_transform (flag -b) is empty!" >&2
    exit 1
fi

if [ -z "$postIMC_to_postIMS_transform" ]; then
    echo "Variable postIMC_to_postIMS_transform (flag -c) is empty!" >&2
    exit 1
fi

wsireg2d $wsireg_config
mv ${preIMC_to_postIMS_transform/_tmp.json/.json} ${preIMC_to_postIMS_transform}
mv ${postIMC_to_postIMS_transform/_tmp.json/.json} ${postIMC_to_postIMS_transform}


