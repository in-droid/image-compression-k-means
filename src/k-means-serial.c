#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "FreeImage.h"
#include <time.h>
#include <math.h>
#include <limits.h> 
#include <CL/cl.h>
#include <omp.h>

#define CENTROID_NUM 64
#define ITERATIONS 50
#define OMP_THREADS 2

int find_min_dist(unsigned char *imageIn, int x, int y, int width, int* centroids) {

    int min_dist = INT_MAX;
    int min_index = 0;

    for (int i = 0; i < CENTROID_NUM; i++) {
        
        int centroid_R = centroids[i * 3];
        int centroid_G = centroids[(i * 3) + 1];
        int centroid_B = centroids[(i * 3) + 2];

        int sample_R = imageIn[(y * width + x) * 4];
        int sample_G = imageIn[(y * width + x) * 4 + 1];
        int sample_B = imageIn[(y * width + x) * 4 + 2];

        int dist = ((centroid_R - sample_R) * (centroid_R - sample_R)  + (centroid_G - sample_G) * (centroid_G - sample_G)  
                    + (centroid_B - sample_B) * (centroid_B - sample_B));

        if (dist < min_dist) {
            min_dist = dist;
            min_index = i;
        }
    }

    return min_index;
}

int find_min_dist_k(unsigned char *imageIn, int k, int* centroids) {

    int min_dist = INT_MAX;
    int min_index = 0;

    for (int i = 0; i < CENTROID_NUM; i++) {
        
        int centroid_R = centroids[i * 3];
        int centroid_G = centroids[(i * 3) + 1];
        int centroid_B = centroids[(i * 3) + 2];

        int sample_R = imageIn[k * 4];
        int sample_G = imageIn[k * 4 + 1];
        int sample_B = imageIn[k * 4 + 2];

        int dist = (centroid_R - sample_R) * (centroid_R - sample_R)  + (centroid_G - sample_G) * (centroid_G - sample_G) 
                    + (centroid_B - sample_B) * (centroid_B - sample_B);

        if (dist < min_dist) {
            min_dist = dist;
            min_index = i;
        }
    }

    return min_index;

}

void init_to_zero(int *array, int size) {
    for(int i = 0; i < size; i++) {
        array[i] = 0;
    }
}


int rand_num(int current, int max) {
    int result;
    while (1) {
        result = rand() % max;
        if (result != current) {
            break;
        }
    }
    return result;   
}

void init_centroids(unsigned char *imageIn, int width, int height, int numCentr,
                    int *centroids) {
    
    srand(time(NULL));

    for (int i = 0; i < numCentr; i ++) {
        int x = rand() % width; 
        int y = rand() % height;

        centroids[i * 3] = imageIn[(y * width + x) * 4];
        centroids[(i * 3) + 1] = imageIn[(y * width + x) * 4 + 1];
        centroids[(i * 3) + 2] = imageIn[(y * width + x) * 4 + 2];
    }

}


void k_means_serial(unsigned char *imageIn, unsigned char *imageOut, int width, 
                    int height, int numCentr, int iterations) {
    
    srand(time(NULL));

    
    int* centroidIndex = (int*) malloc(width * height * sizeof(int));
    int samplesInCluster[numCentr];

    int centroids[3 * numCentr];
    init_centroids(imageIn, width, height, numCentr, centroids);

    int centroidMean[3 * numCentr];

    int indx;


    for (int i = 0; i < iterations; i++) {
        init_to_zero(samplesInCluster, numCentr);
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                indx = find_min_dist(imageIn, x, y, width, centroids);
                centroidIndex[(y * width) + x] = indx;
                samplesInCluster[indx]++;
            }
            // }
        }

    
        for (int j = 0; j < numCentr; j++) {
            if(samplesInCluster[j] == 0) {
                int rand_x, rand_y, c_ind_old;
                while(1) {
                    rand_x = rand() % width;
                    rand_y = rand() % height;
                    c_ind_old = centroidIndex[(rand_y * height) + rand_x];
                    if (samplesInCluster[c_ind_old] > 1) {
                        break;
                    }
                }
                
                centroidIndex[(rand_y * height) + rand_x] = j;
                samplesInCluster[c_ind_old]--;
                samplesInCluster[j]++;
           }
        }
        
        init_to_zero(centroidMean, 3 * numCentr);
        int pxR, pxG, pxB;
        for(int k = 0; k < width * height; k++) {
            int centrInd = centroidIndex[k];
            
            pxR = imageIn[k * 4];
            pxG = imageIn[k * 4 + 1];
            pxB = imageIn[k * 4 + 2];

            centroidMean[centrInd * 3] += pxR;
            centroidMean[centrInd * 3 + 1] += pxG;
            centroidMean[centrInd * 3 + 2] += pxB;
        }
        for (int k = 0; k < numCentr; k++) {
            centroidMean[k * 3] /= samplesInCluster[k];
            centroidMean[k * 3 + 1] /= samplesInCluster[k];
            centroidMean[k * 3 + 2] /= samplesInCluster[k];

            centroids[k * 3] = centroidMean[k * 3];
            centroids[k * 3 + 1] = centroidMean[k * 3 + 1];
            centroids[k * 3 + 2] = centroidMean[k * 3 + 2];
        }
    }

    for (int k = 0; k < width * height; k++) {
        int index = centroidIndex[k];
        imageOut[k * 4] = centroids[index * 3];
        imageOut[(k * 4) + 1] = centroids[(index * 3) + 1];
        imageOut[(k * 4) + 2] = centroids[(index * 3) + 2];
        imageOut[(k * 4) + 3] = 255;
    }

}


int main (int argc, char** argv) {

    printf("K-means clustering image compression\n");
    printf("---------------------------------------\n");
    printf("K = %d\t ITERATIONS = %d\n", CENTROID_NUM, ITERATIONS);
    FIBITMAP *imageBitmap = FreeImage_Load(FIF_PNG, "../images/cat-640x480.png", 0);
	//Convert it to a 32-bit image
    FIBITMAP *imageBitmap32 = FreeImage_ConvertTo32Bits(imageBitmap);
    FIBITMAP *dst; 
	
    //Get image dimensions
    int width = FreeImage_GetWidth(imageBitmap32);
	int height = FreeImage_GetHeight(imageBitmap32);
	int pitch = FreeImage_GetPitch(imageBitmap32);

    printf("IMAGE SIZE: %d x %d\n", width, height);
	//Preapare room for a raw data copy of the image
    unsigned char *imageIn = (unsigned char *)malloc(height*pitch * sizeof(unsigned char));
    unsigned char *imageOut = (unsigned char *)malloc(height * pitch * sizeof(unsigned char));
    FreeImage_ConvertToRawBits(imageIn, imageBitmap32, pitch, 32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);

    //Free source image data
	FreeImage_Unload(imageBitmap32);
	FreeImage_Unload(imageBitmap);

    double startTime = omp_get_wtime();
    k_means_serial(imageIn, imageOut, width, height, CENTROID_NUM, ITERATIONS);
    double endTime = omp_get_wtime();
    double cpuTime = endTime - startTime;
	printf("CPU TIME: %f\n", cpuTime);
    dst = FreeImage_ConvertFromRawBits(imageOut, width, height, pitch,
		32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);
	FreeImage_Save(FIF_PNG, dst, "../serial-out/cat-640x480.png", 0);
    
    free(imageIn);
    free(imageOut);

    return 0;
	
}