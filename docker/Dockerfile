FROM nvcr.io/nvidia/pytorch:21.12-py3

RUN pip install --upgrade jupyterlab
RUN pip install ipywidgets
RUN pip install seaborn

RUN pip install pytorch-lightning==1.5.8
RUN pip install transformers==4.16.0
RUN pip install gensim==4.1.2

RUN conda install -c bioconda matchms==0.12.0
RUN conda install -c bioconda -c nlesc spec2vec==0.5.0
