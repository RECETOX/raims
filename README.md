# AI-based Methods for Mass Spectrometry

## Quickstart

Depending on your platform you may choose from these several options:

#### Docker
```bash
cd docker
docker compose up jupyter
```
Then open your browser and connect to the Jupyter service (watch the output of docker compose for the correct jupyter token).

#### Metacentrum (Singularity)

Modify the script `run_metacentrum.sh` to your liking, ie. change the number of computation nodes or the size of memory, and execute:
```bash
qsub run_metacentrum.sh
```
After the submission of the job wait for an e-mail with an url where you can connect to the jupyter service.
