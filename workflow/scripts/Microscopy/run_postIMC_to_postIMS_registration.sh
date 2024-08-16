#!/bin/bash

while getopts a:b:c: flag
do
    case "${flag}" in
        a) wsireg_config=${OPTARG};;
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


# run registration
wsireg2d $wsireg_config

