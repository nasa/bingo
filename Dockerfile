FROM ubuntu:22.04
RUN git clone --recurse-submodules --depth 1 https://github.com/nasa/bingo.git /opt/bingo/
WORKDIR /opt/bingo/
RUN python -m pip install -r requirements.txt
RUN mkdir -p bingocpp/build/
WORKDIR /opt/bingo/bingocpp/build/
RUN cmake -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE:FILEPATH=$(which python) .. && \
    make VERBOSE=1 -j
ENV PYTHONPATH "${PYTHONPATH}:/opt/bingo/"
CMD ["/bin/bash"]
