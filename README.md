# SLIC-DBSCAN-CUDA

a CUDA implementation of DBSCAN clustering algorithm

Demo video : https://www.youtube.com/watch?v=dB5C3oTeX3o&feature=youtu.be

## Description

This project contains C++ and CUDA implementation of a paper "G-DBSCAN: A GPU Accelerated Algorithm for Density-based Clustering".

Especially, G-DBSCAN is used for realtime clustering of SLIC superpixels in demo application.

## Requirement

* OpenCV (for demo application)
* CUDA

## Build

  git clone https://github.com/ca1773130n/SLIC-DBSCAN-CUDA.git

  cd SLIC-DBSCAN-CUDA

  mkdir build

  cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLE=1 -DCUDA_ARCH=compute_50

  make -j8

* Make sure with CUDA_ARCH argument matches for your NVIDIA GPU architecture.

## Run

To run with webcam:

  ./demo

To run with video file (e.g. mp4):

  ./demo <video filename>

* Only 640x480 video files are available for now

## References

[1] Andrade, G., Ramos, G., Madeira, D., Sachetto, R., Ferreira, R., Rocha, L.: G-DBSCAN: A GPU Accelerated Algorithm for Density-based Clustering. Procedia Comput. Sci. 18, 369–378 (2013)

[2] C. Y. Ren, V. A. Prisacariu, and I.  D. Reid, “gSLICr: SLIC superpixels at over 250Hz,” ArXiv e-prints, Sep. 2015.

## License

MIT license

## Acknowledge

CUDA implementation of SLIC is from gSLICr below (slightly modified).

Project website: http://www.robots.ox.ac.uk/~victor/gslicr/

