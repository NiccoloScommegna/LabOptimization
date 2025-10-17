FROM mcr.microsoft.com/devcontainers/miniconda:1-3

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

# [Optional] Uncomment to install a different version of Python than the default
# RUN conda install -y python=3.6 \
#     && pip install --no-cache-dir pipx \
#     && pipx reinstall-all

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

# # Usa la base Miniconda di Microsoft Dev Containers
# FROM mcr.microsoft.com/devcontainers/miniconda:1-3

# # Imposta la directory di lavoro nel container
# WORKDIR /workspace

# # Installa pacchetti di sistema aggiuntivi (opzionale, utile per pycutest)
# RUN apt-get update && \
#     apt-get install -y build-essential gfortran git wget && \
#     rm -rf /var/lib/apt/lists/*

# # Installa Python packages direttamente via pip
# # Qui metti TUTTE le tue librerie necessarie
# RUN pip install --no-cache-dir \
#     numpy \
#     scipy \
#     matplotlib \
#     pandas \
#     seaborn \
#     pycutest

# # Imposta variabile d’ambiente per PyCUTEst
# ENV PYCUTEST_CACHE=/workspace/pycutest_cache

# # Crea la cartella cache (non serve se montata da host, ma non fa male)
# RUN mkdir -p /workspace/pycutest_cache

# # Imposta la cartella di lavoro finale
# WORKDIR /workspace/code
