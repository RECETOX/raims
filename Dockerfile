FROM nvcr.io/nvidia/pytorch:21.12-py3

RUN conda config \
      --add channels bioconda \
      --add channels nlesc \
    && conda install \
      matchms==0.14.0 \
      spec2vec==0.5.0 \
    && conda clean --all \
    && pip install \
      gensim==4.1.2 \
      pytorch-lightning==1.5.10 \
      transformers==4.16.2

WORKDIR /code
COPY pyproject.toml setup.cfg /code/
COPY src /code/src/

RUN pip install .

WORKDIR /workspace

COPY data/src/ /workspace/data/src/
COPY notebooks/ /workspace/notebooks/