# FROM mcr.microsoft.com/devcontainers/miniconda:1-3

# # Copy environment.yml (if found) to a temp location so we update the environment. Also
# # copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
# COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
# RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
#     && rm -rf /tmp/conda-tmp

# # [Optional] Uncomment to install a different version of Python than the default
# # RUN conda install -y python=3.6 \
# #     && pip install --no-cache-dir pipx \
# #     && pipx reinstall-all

# # [Optional] Uncomment this section to install additional OS packages.
# # RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

# <--->

FROM mcr.microsoft.com/devcontainers/miniconda:1-3

# Imposta directory principale
WORKDIR /cutest

# Installa dipendenze di sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    gfortran \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Clona i repository CUTEst
RUN git clone https://github.com/ralna/ARCHDefs ./archdefs
RUN git clone https://github.com/ralna/SIFDecode ./sifdecode
RUN git clone https://github.com/ralna/CUTEst ./cutest
RUN git clone https://bitbucket.org/optrove/sif ./mastsif

# Imposta variabili d’ambiente necessarie per CUTEst
ENV ARCHDEFS=/cutest/archdefs
ENV SIFDECODE=/cutest/sifdecode
ENV CUTEST=/cutest/cutest
ENV MASTSIF=/cutest/mastsif
ENV MYARCH=pc64.lnx.gfo

# Scarica ed esegue lo script di installazione di CUTEst
RUN wget https://raw.githubusercontent.com/jfowkes/pycutest/master/.install_cutest.sh && \
    chmod +x .install_cutest.sh && \
    ./.install_cutest.sh

# Installa librerie Python necessarie
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    pandas \
    seaborn \
    pycutest

# Crea la cartella di lavoro
RUN mkdir -p /workspace/code
WORKDIR /workspace/code

# Imposta variabile per la cache PyCUTEst
ENV PYCUTEST_CACHE=/workspace/pycutest_cache
RUN mkdir -p /workspace/pycutest_cache

# Inizializza Conda in bash per evitare l’errore "Run 'conda init' before 'conda activate'"
RUN /opt/conda/bin/conda init bash
