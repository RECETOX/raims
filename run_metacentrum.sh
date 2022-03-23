#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb:scratch_ssd=200gb:gpu_cap=cuda70
#PBS -l walltime=8:00:00
#PBS -m ae

echo "${PBS_O_LOGNAME:?This script must be run under PBS scheduling system, execute: qsub $0}"

HOSTNAME=$(hostname -f)
PORT=8888
TOKEN="d08d8f7784636a70572c44c4d4e4a7131d0de87fc54f2eac"
SINGULARITY_IMAGE="$PBS_O_WORKDIR/singularity/raims_jupyter.sif"

while netstat -taln | grep -q $PORT
do
	PORT=$((PORT + 1))
done

mail -s "Jupyter session started" "$PBS_O_LOGNAME" << EOFmail
Jupyter session started http://$HOSTNAME:$PORT?token=$TOKEN
EOFmail

export SINGULARITY_CACHEDIR=$PBS_O_HOME
export SINGULARITY_LOCALCACHEDIR=$SCRATCHDIR
export SINGULARITY_TMPDIR=$SCRATCHDIR

cd "$PBS_O_HOME" || exit $?

singularity exec --nv \
	--home "$PBS_O_HOME" \
	--bind "$SCRATCHDIR" \
	--bind /storage \
	"$SINGULARITY_IMAGE" \
	jupyter-lab \
	  --port=$PORT \
	  --NotebookApp.token=$TOKEN \
	  --notebook-dir="$PBS_O_WORKDIR"

clean_scratch
