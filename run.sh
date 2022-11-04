#!/bin/bash

echo IMAGE=${IMAGE:=ljocha/raims}
#echo IMAGE=${IMAGE:=xtracko/raims_jupyter}

echo PORT=${PORT:=8888}
echo DATADIR=${DATADIR:=$PWD}

echo GPUS=${GPUS:='"device=1"'}
echo NVIDIA_FLAGS=${NVIDIA_FLAGS:=--gpus $GPUS --ipc=host --ulimit memlock=-1 --ulimit stack=67108864}

docker run $NVIDIA_FLAGS \
	-u $(id -u) \
	-e HOME=/home/jovyan -v $HOME:/home/jovyan \
	-p $PORT:$PORT \
	-v $DATADIR/notebooks:/workspace -v $DATADIR/src/raims:/workspace/raims \
	-v $DATADIR/data:/workspace/data -v $DATADIR/model:/workspace/model -v $DATADIR/wandb:/workspace/wandb\
	-ti --rm \
	$IMAGE jupyter-lab

