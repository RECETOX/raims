#!/bin/bash


base=/storage/brno2/home/ljocha/raims/
# gpus=1
cluster=XXX

for heads in 6 12 ; do
	for layers in 6 12 ; do
		for embed in 120 300 900; do
			for batch in 64 128 256; do
				for half in "" --half; do
					for gpus in 1 2 4; do
			
 

#heads=6
#layers=12
#embed=300
#batch=128
#half=--half

fullname=raims_${cluster}_H${heads}_L${layers}_E${embed}_G${gpus}_B${batch}_${half}

# bulharske konstanty
mem=$(($batch / 4 + $heads / 2 + $layers / 6 + $embed / 60))

cat - >$fullname.sh <<EOF
#!/bin/bash

#PBS -N $fullname
#PBS -q gpu
#PBS -l select=1:ncpus=2:mem=${mem}gb:scratch_local=10gb:ngpus=$gpus:cluster=$cluster
#PBS -l walltime=4:00:00

trap 'rm -r \$SCRATCHDIR' TERM EXIT

cd $base
tar cf - raims_job.py data model wandb/key | (cd \$SCRATCHDIR && tar xpf - )

cd \$SCRATCHDIR

singularity run --nv -B \$SCRATCHDIR:/work --pwd /work $base/ljocha-raims.sif \
	python3 raims_job.py --heads $heads --layers $layers --embed $embed $half --gpus $gpus \
		--batch $batch --name $cluster 
	
EOF

echo qsub $fullname.sh

done;	done;	done;	done;	done;	done

