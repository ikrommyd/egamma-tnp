ARG FROM_IMAGE=gitlab-registry.cern.ch/batch-team/dask-lxplus/lxdask-al9:latest
FROM ${FROM_IMAGE}

ARG CLUSTER=lxplus-el9

ADD . .

RUN echo "=======================================" && \
    echo "Installing egamma-tnp" && \
    echo "on cluster environment: $CLUSTER" && \
    echo "Current time:" $(date) && \
    echo "=======================================" && \
    if [[ ${CLUSTER} == "lxplus-cc7" ]]; then \
        echo "Fixing dependencies in the image" && \
        conda install -y numba>=0.57.0 llvmlite==0.40.0 numpy>=1.22.0 && \
        python -m pip install -U dask-lxplus==0.3.2 dask-jobqueue==0.8.2; \
    elif [[ ${CLUSTER} == "lxplus-el9" ]]; then \
        echo "Installing on alma9"; \
    fi && \
    echo "Installing egamma-tnp" && \
    python -m pip install . --verbose
