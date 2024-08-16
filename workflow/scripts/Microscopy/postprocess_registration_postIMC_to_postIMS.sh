#!/bin/bash

while getopts a:b:c:d:e:f:g:h:i:j:k:l: flag
do
    case "${flag}" in
        a) preIMS_to_postIMS_image=${OPTARG};;
        b) preIMC_to_postIMS_image=${OPTARG};;
        c) postIMC_to_postIMS_image=${OPTARG};;
        d) preIMC_to_postIMS_transform=${OPTARG};;
        e) postIMC_to_postIMS_transform=${OPTARG};;
        f) preIMS=${OPTARG};;
        g) preIMC=${OPTARG};;
        h) preIMS_image=${OPTARG};;
        i) preIMC_image=${OPTARG};;
        j) postIMC_image=${OPTARG};;
        k) preIMC_to_postIMS_transform_out=${OPTARG};;
        l) postIMC_to_postIMS_transform_out=${OPTARG};;
    esac
done
shift "$(( OPTIND - 1 ))"

if [ -z "$preIMS_to_postIMS_image" ]; then
    echo "Variable preIMS_to_postIMS_image (flag -a) is empty!" >&2
    exit 1
fi
# check if file exists
if [ ! -f "$preIMS_to_postIMS_image" ]; then
    echo "File '$preIMS_to_postIMS_image' does not exist" >&2
    exit 1
fi

if [ -z "$preIMC_to_postIMS_image" ]; then
    echo "Variable preIMC_to_postIMS_image (flag -a) is empty!" >&2
    exit 1
fi
# check if file exists
if [ ! -f "$preIMC_to_postIMS_image" ]; then
    echo "File '$preIMC_to_postIMS_image' does not exist" >&2
    exit 1
fi

if [ -z "$postIMC_to_postIMS_image" ]; then
    echo "Variable postIMC_to_postIMS_image (flag -a) is empty!" >&2
    exit 1
fi
# check if file exists
if [ ! -f "$postIMC_to_postIMS_image" ]; then
    echo "File '$postIMC_to_postIMS_image' does not exist" >&2
    exit 1
fi

if [ -z "$preIMC_to_postIMS_transform" ]; then
    echo "Variable preIMC_to_postIMS_transform (flag -a) is empty!" >&2
    exit 1
fi
# check if file exists
if [ ! -f "$preIMC_to_postIMS_transform" ]; then
    echo "File '$preIMC_to_postIMS_transform' does not exist" >&2
    exit 1
fi

if [ -z "$postIMC_to_postIMS_transform" ]; then
    echo "Variable postIMC_to_postIMS_transform (flag -a) is empty!" >&2
    exit 1
fi
# check if file exists
if [ ! -f "$postIMC_to_postIMS_transform" ]; then
    echo "File '$postIMC_to_postIMS_transform' does not exist" >&2
    exit 1
fi

if [ -z "$preIMC" ]; then
    echo "Variable preIMC (flag -a) is empty!" >&2
    exit 1
fi
# check if file exists
if [ ! -f "$preIMC" ]; then
    echo "File '$preIMC' does not exist" >&2
    exit 1
fi

if [ -z "$preIMS" ]; then
    echo "Variable preIMS (flag -a) is empty!" >&2
    exit 1
fi
# check if file exists
if [ ! -f "$preIMS" ]; then
    echo "File '$preIMS' does not exist" >&2
    exit 1
fi

if [ -z "$preIMS_image" ]; then
    echo "Variable preIMS_image (flag -a) is empty!" >&2
    exit 1
fi

if [ -z "$preIMC_image" ]; then
    echo "Variable preIMC_image (flag -a) is empty!" >&2
    exit 1
fi

if [ -z "$postIMC_image" ]; then
    echo "Variable postIMC_image (flag -a) is empty!" >&2
    exit 1
fi

if [ -z "$preIMC_to_postIMS_transform_out" ]; then
    echo "Variable preIMC_to_postIMS_transform_out (flag -a) is empty!" >&2
    exit 1
fi

if [ -z "$postIMC_to_postIMS_transform_out" ]; then
    echo "Variable postIMC_to_postIMS_transform_out (flag -a) is empty!" >&2
    exit 1
fi


ln -sr -T $preIMS_to_postIMS_image $preIMS_image
ln -sr -T $postIMC_to_postIMS_image $postIMC_image


# if preIMC and preIMS are the same, do linear registration only 
md5preIMC=($(md5sum ${preIMC} ))
md5preIMS=($(md5sum ${preIMS} ))

# set nonlinear registration between preIMC and preIMS if the two files are the same
if [ "${md5preIMC[0]}" == "${md5preIMS[0]}" ]; then
    ln -sr -T $preIMS_to_postIMS_image $preIMC_image
else 
    ln -sr -T $preIMC_to_postIMS_image $preIMC_image
fi

# set nonlinear registration between preIMC and preIMS if the two files are the same
if [ "${md5preIMC[0]}" == "${md5preIMS[0]}" ]; then
    results/Misc/yq '.000-to-preIMS[0].TransformParameters = ["0","0","0"]' "$preIMC_to_postIMS_transform" > "$preIMC_to_postIMS_transform_out" 
    results/Misc/yq '.001-to-preIMS[0].TransformParameters = ["0","0","0"]' "$postIMC_to_postIMS_transform" > "$postIMC_to_postIMS_transform_out" 
else 
    ln -sr -T $preIMC_to_postIMS_transform $preIMC_to_postIMS_transform_out
    ln -sr -T $postIMC_to_postIMS_transform $postIMC_to_postIMS_transform_out
fi