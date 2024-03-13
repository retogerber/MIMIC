FROM rocker/r-ver:4.2.3

# system libraries of general use
RUN apt-get update \
    && apt-get install -y \
        libudunits2-dev \
        libgdal-dev \
        libgeos-dev \
        libproj-dev \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
        libfftw3-3 \
        python3.10 \
        python3-pip \
        jq \
        git \
    && apt-get clean \
    && rm -rf /tmp/* /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

#RUN CLFAGS='-g0 -Wl, --strip-all -I/usr/include:/usr/local/include -L/usr/lib:/usr/local/lib' python3 -m pip install --prefer-binary --no-cache-dir --compile  -I --global-option=build_ext \
#        "numpy==1.24.2" 

RUN python -m pip install --prefer-binary --no-cache-dir --compile  -I --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.0.1" \
        "torchvision==0.15.2" \
    && python -m pip install --prefer-binary --no-cache-dir --compile  -I  \
        "av==9.0.2" \
        "cffi==1.15.1" \
        "click==8.1.3" \
        "git+https://github.com/facebookresearch/segment-anything.git@6fdee8f2727f4506cfbbe553e23b895e27956588" \
        "git+https://github.com/siavashk/pycpd.git@e5ca02d2501fb4b633c1664de939c336ffa2349e" \
        "h5py==3.8.0" \
        "imageio==2.33.1" \
        "napari-imsmicrolink==0.1.9" \
        "napari==0.4.17" \
        "numpy==1.24.2" \
        "ome-types==0.3.3" \
        "opencv-contrib-python-headless==4.7.0.68" \
        "pandas==2.0.1" \
        "pathlib==1.0.1" \
        "PyQt5==5.14.2" \
        "pyyaml==6.0" \
        "pydantic==1.*" \
        "rembg==2.0.49" \
        "scikit-image==0.21.0" \
        "scikit-learn==1.3.0" \
        "scipy==1.10.0" \
        "shapely==2.0.1" \
        "SimpleITK==2.3.1" \
        "tifffile==2023.4.12" \
        "wsireg==0.3.8" \
        "zarr==2.15.0"

# to napari-imsmicrolink use headless
ENV QT_QPA_PLATFORM offscreen

# Install R packages
RUN R -e 'install.packages("pak", repos = sprintf("https://r-lib.github.io/p/pak/stable/%s/%s/%s",.Platform$pkgType, R.Version()$os, R.Version()$arch)); pak::pkg_install(c("Cardinal@3.0.1","rhdf5@2.42.1","RhpcBLASctl@0.23-42","data.table@1.14.8","dplyr@1.1.2","sp@1.6-0","BiocNeighbors@1.16.0","spdep@1.2-8","plotly@4.10.1","ggnewscale@0.4.8"), dependencies=NA, ask=FALSE)'


