# Image Compression using k-means clustering
Image compression using k-means clustering implemented in OpenMP and OpenCL. Project for the subject Parallel and Distributed Systems and Algorithms.

## Requirements
- OpenCL
- OpenMP
- LibFreeImage

## Build
1. Run `module load CUDA` so you can use OpenCL
2. Compile the file k-means-gpu.c using `gcc  k-means-gpu.c -fopenmp -O2 -lm -lOpenCL -Wl,-rpath,./ -L./ -l:"libfreeimage.so.3" -o k-means-gpu -Wall -Wextra`
3. Put the image you want to compress in images
4. Run `./k-means-gpu`

The repo is still in progress, better documentation is coming soon. Detailed report is avilable [here](https://github.com/in-droid/image-compression-k-means/blob/main/Project%20assignment.pdf)
