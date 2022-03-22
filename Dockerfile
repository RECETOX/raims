FROM nvcr.io/nvidia/pytorch:22.02-py3

RUN conda config \
      --add channels bioconda \
      --add channels nlesc \
    && conda install \
      matchms==0.14.0 \
      spec2vec==0.5.0 \
      wandb==0.12.11 \
    && conda clean --all \
    && pip install \
      gensim==4.1.2 \
      pytorch-lightning==1.5.10 \
      transformers==4.16.2

RUN rm -rf /workspace/* \
    && mkdir /workspace/data \
    && mkdir /workspace/model \
    && mkdir /workspace/wandb

COPY notebooks/ /workspace/
COPY src/raims/ /workspace/raims/
