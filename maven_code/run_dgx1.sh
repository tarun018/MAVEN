#!/bin/bash
HASH=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 4 | head -n 1)
GPU=$1
GPU_N=$(echo $1 | tr ',' '_')
name=${USER}_maven_GPU_${GPU_N}_${HASH}

echo "Launching container named '${name}' on GPU '${GPU}'"
# Launches a docker container using our image, and runs the provided command

if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi

NV_GPU="$GPU" ${cmd} run \
    --name $name \
    --user $(id -u) \
    -v `pwd`:/pymarl \
    -v $(readlink -f /raid/data/):/data/dgx1/ \
    -e STORAGE_HOSTNAME=`hostname -s` \
    -t tg/pymarl:3.0 \
    ${@:2}