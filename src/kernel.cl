#define LOCAL_SIZE 16

int eucledian_square(int A1, int A2,  int A3, int B1, int B2, int B3) {

    return (A1 - B1) * (A1 - B1) + (A2 - B2) * (A2 - B2) + (A3 - B3) * (A3 - B3);
}

int find_min_dist_k(int* sample, int k, 
                    int* centroids, int numCentr) {
    
    int sample_R = sample[0];
    int sample_G = sample[1];
    int sample_B = sample[2];

    int min_dist = eucledian_square(sample_R, sample_G, sample_B,
                                    centroids[0], centroids[numCentr], centroids[numCentr * 2]);
    int min_index = 0;

    for (int i = 1; i < numCentr; i++) {
        
        int centroid_R = centroids[i];
        int centroid_G = centroids[numCentr + i];
        int centroid_B = centroids[numCentr * 2 + i];

        int dist = eucledian_square(sample_R, sample_G, sample_B, centroid_R, centroid_G, centroid_B);
         
        if (dist < min_dist) {
            min_dist = dist;
            min_index = i;
        }
    }

    return min_index;

}

uint gen_random(uint randoms_x, uint randoms_y, uint globalID) {
    uint seed = randoms_x + globalID;
    uint t = seed ^ (seed << 11);
    uint result = randoms_y ^ (randoms_y >> 19) ^ (t ^ (t >> 8));
    return result;
}


__kernel void k_means_nearest(__global unsigned char *imageIn,
                            __global int* centroids,
                            __global int* centroidIndex,
                            __global int* samplesInCentroid,
                            __global int* centroidMean,
                            int width, int height,
                            int numCentr) {

    int iGlobal = get_global_id(1);
    int jGlobal = get_global_id(0);
    int kGlobal = iGlobal * width + jGlobal;

    int iLocal = get_local_id(1);
    int jLocal = get_local_id(0);
    int kLocal = iLocal * get_local_size(1) + jLocal;

    int indx;
    int size = width * height;
    __local int centroids_local[3 * C_SIZE];
    int pXR, pxG, pxB;
    if (iGlobal < height && jGlobal < width) {

        if (kLocal < numCentr) {
            centroids_local[kLocal] = centroids[kLocal];
            centroids_local[kLocal + numCentr] = centroids[kLocal + numCentr];
            centroids_local[kLocal + 2 * numCentr] = centroids[kLocal + 2 * numCentr];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int pxR = imageIn[kGlobal * 4];
        int pxG = imageIn[kGlobal * 4 + 1];
        int pxB = imageIn[kGlobal * 4 + 2];
        int sample[3] = {pxR, pxG, pxB};
        
        indx = find_min_dist_k(sample, kGlobal, centroids_local, numCentr);
        centroidIndex[kGlobal] = indx;
        atomic_inc(&samplesInCentroid[indx]);
        
        atomic_add(&centroidMean[indx], pxR);
        atomic_add(&centroidMean[numCentr + indx], pxG);
        atomic_add(&centroidMean[numCentr * 2 + indx], pxB);
    }
    
}


__kernel void k_means_c_mean(__global unsigned char* imageIn,
                            __global int* centroids,
                            __global int* centroidMean,
                            __global int* samplesInCentroid,
                            int width, int height,
                            int numCentr,
                            int randoms_x) 
{

    int iGlobal = get_global_id(0);
    if (iGlobal < numCentr) {
        int count = samplesInCentroid[iGlobal];
        int pxR, pxG, pxB;
        if (count == 0) {
            int rand_k = gen_random(randoms_x, 0, iGlobal) % (width * height);
        
            samplesInCentroid[iGlobal]++;
            pxR = imageIn[rand_k * 4];
            pxG = imageIn[rand_k * 4 + 1];
            pxB = imageIn[rand_k * 4 + 2];
        }
        else {
            pxR = centroidMean[iGlobal];
            pxG = centroidMean[numCentr + iGlobal];
            pxB = centroidMean[numCentr * 2 + iGlobal];

            pxR /= count;
            pxG /= count;
            pxB /= count;
        }
        centroids[iGlobal] = pxR;
        centroids[iGlobal + numCentr]  = pxG;
        centroids[iGlobal + 2 * numCentr] = pxB;

        centroidMean[iGlobal] = 0;
        centroidMean[numCentr + iGlobal] = 0;
        centroidMean[numCentr * 2 + iGlobal] = 0;
        samplesInCentroid[iGlobal] = 0;

    }   
}


__kernel void write_back(__global unsigned char* imageOut,
                        __global int* centroidIndex,
                        __global int* centroids,
                        int width, int height,
                        int numCentr) {

    int iGlobal = get_global_id(1);
    int jGlobal = get_global_id(0);

    if (iGlobal < height && jGlobal < width) {
        int kGlobal = iGlobal * width + jGlobal;
        int index = centroidIndex[kGlobal];
        imageOut[kGlobal * 4] = centroids[index];
        imageOut[kGlobal * 4 + 1] = centroids[index + numCentr];
        imageOut[kGlobal * 4 + 2] = centroids[index + numCentr * 2];
        imageOut[kGlobal * 4 + 3] = 255;



    }

}