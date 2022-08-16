FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        build-essential \
        cmake \
        python3.10 \
        python3-dev \
        python3-mpi4py \
        python3-numpy \
        python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
COPY ./ /opt/bingo/
WORKDIR /opt/bingo/
RUN python -m pip install -r requirements.txt
RUN mkdir -p /opt/bingo/bingocpp/build/
WORKDIR bingocpp/build/
RUN cmake -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE:FILEPATH=$(which python) .. && \
    make VERBOSE=1 -j
ENV PYTHONPATH "${PYTHONPATH}:/opt/bingo/"
CMD ["/bin/bash"]
