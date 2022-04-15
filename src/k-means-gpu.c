#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "FreeImage.h"
#include <omp.h>
#include <math.h>
#include <time.h>

#define WORKGROUP_SIZE_WIDTH (16)
#define WORKGROUP_SIZE_HEIGHT (16)
#define WORKGROUP_SIZE_2 64
#define MAX_SOURCE_SIZE 16384
#define CENTROID_NUM 64
#define ITERATIONS 50 

double gpuTime;

void init_centroids(unsigned char *imageIn, int width, int height, int numCentr,
                    int *centroids) {
    
    srand(time(NULL));

    for (int i = 0; i < numCentr; i ++) {
        int x = rand() % width; 
        int y = rand() % height;

        centroids[i] = imageIn[(y * width + x) * 4];
        centroids[i + numCentr] = imageIn[(y * width + x) * 4 + 1];
        centroids[i + numCentr * 2] = imageIn[(y * width + x) * 4 + 2];
    }

}



void k_means_gpu(unsigned char *imageIn, unsigned char *imageOut, int width, int height,
                int pitch, int numCentr, int iterations) {

    srand(time(NULL));

    char ch;
    int i;
	cl_int ret;
    const size_t num_groups_width =  (int)ceil((float) width / (WORKGROUP_SIZE_WIDTH * 1.0));
    const size_t num_groups_height = (int)ceil((float) height / (WORKGROUP_SIZE_HEIGHT * 1.0));
    int seed = rand();

	int imageSize = height * width;

    
    cl_int* centroidIndex = (cl_int*) malloc(imageSize * sizeof(cl_int));
    cl_int* centroids = (cl_int*) malloc(3 * numCentr * sizeof(cl_int));
    cl_int* samplesInCentroid = (cl_int*) calloc(numCentr, sizeof(cl_int));
    cl_int* centroidMean = (cl_int*) calloc(3 * numCentr, sizeof(cl_int));
    //cl_int* localSums = (cl_int*) calloc(localSums_size, sizeof(cl_int));
    //cl_int* sums = (cl_int*) calloc(sums_size, sizeof(cl_int));
    
    init_centroids(imageIn, width, height, numCentr, centroids);

    // Branje datoteke
    FILE *fp;
    char *source_str;
    size_t source_size;


    fp = fopen("kernel.cl", "r");
    if (!fp) 
	{
		fprintf(stderr, ":-(#\n");
        exit(1);
    }

    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	source_str[source_size] = '\0';
    fclose( fp );
 
    // Podatki o platformi
    cl_platform_id	platform_id[10];
    cl_uint			ret_num_platforms;
	char			*buf;
	size_t			buf_len;
	ret = clGetPlatformIDs(10, platform_id, &ret_num_platforms);
	

	cl_device_id	device_id[10];
	cl_uint			ret_num_devices;

	ret = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 10,	
						 device_id, &ret_num_devices);				
	

    // Kontekst
    cl_context context = clCreateContext(NULL, 1, &device_id[0], NULL, NULL, &ret);
	

    // Ukazna vrsta
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id[0], 0, &ret);

	// Delitev dela
    const size_t local_item_size[2] = {WORKGROUP_SIZE_WIDTH, WORKGROUP_SIZE_HEIGHT};

    


    const size_t global_item_size[2] = {num_groups_width * WORKGROUP_SIZE_WIDTH, 
                                        num_groups_height * WORKGROUP_SIZE_HEIGHT};


    size_t datasize = sizeof(unsigned char) * pitch * height; 	
  	
  
    // Priprava programa
    cl_program program = clCreateProgramWithSource(context,	1, (const char **)&source_str,  
												   NULL, &ret);


    char compileArgs[64];
    sprintf(compileArgs, "-DC_SIZE=%d", numCentr);
    // Prevajanje
    ret = clBuildProgram(program, 1, &device_id[0], compileArgs, NULL, NULL);
    

	// Log
	size_t build_log_len;
	char *build_log;
	ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, 
								0, NULL, &build_log_len);

	

	build_log =(char *)malloc(sizeof(char)*(build_log_len+1));
	ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, 
							    build_log_len, build_log, NULL);
	printf("%s\n", build_log);
	free(build_log);
    
    // "s"cepec: priprava objekta
   
    cl_kernel kernel_nearest = clCreateKernel(program, "k_means_nearest", &ret);
 
    cl_kernel kernel_means = clCreateKernel(program, "k_means_c_mean", &ret);
   
    cl_kernel kernel_writeBack = clCreateKernel(program, "write_back", &ret);


    double startTime = omp_get_wtime();
    cl_mem image_mem_obj_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
									          datasize, NULL, &ret); 
    

    cl_mem centroids_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                            3 * numCentr * sizeof(cl_int), centroids, &ret);
  
    
    cl_mem centroidIndex_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                imageSize * sizeof(cl_int), NULL, &ret);


    
    cl_mem image_mem_obj_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
									        sizeof(cl_int) * imageSize, imageIn, &ret);

  
    cl_mem samplesInCentroid_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(cl_int) * numCentr, samplesInCentroid, &ret);
    
    cl_mem centroidMean_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                3 * numCentr * sizeof(cl_int), centroidMean, &ret);
   
    

     
    
    ret = clSetKernelArg(kernel_nearest, 0, sizeof(cl_mem), (void *)&image_mem_obj_in);
    printf("IMAGE PASS: %d\n", ret);

    ret |= clSetKernelArg(kernel_nearest, 1, sizeof(cl_mem), (void *)&centroids_mem_obj);
    printf("CENTROID PASS: %d\n", ret);

    ret |= clSetKernelArg(kernel_nearest, 2, sizeof(cl_mem), (void *)&centroidIndex_mem_obj);
    printf("CENTROID INDEX PASS: %d\n", ret);

    ret |= clSetKernelArg(kernel_nearest, 3, sizeof(cl_mem), (void *)&samplesInCentroid_mem_obj);

    ret |= clSetKernelArg(kernel_nearest, 4, sizeof(cl_mem), (void *)&centroidMean_mem_obj);

    ret |= clSetKernelArg(kernel_nearest, 5, sizeof(cl_int), (void *)&width);

    ret |= clSetKernelArg(kernel_nearest, 6, sizeof(cl_int), (void *)&height);

    ret |= clSetKernelArg(kernel_nearest, 7, sizeof(cl_int), (void *)&numCentr);


    ret |= clSetKernelArg(kernel_means, 0, sizeof(cl_mem), (void *)&image_mem_obj_in);



    ret |= clSetKernelArg(kernel_means, 1, sizeof(cl_mem), (void *)&centroids_mem_obj);


    ret |= clSetKernelArg(kernel_means, 2, sizeof(cl_mem), (void *)&centroidMean_mem_obj);
 
    ret |= clSetKernelArg(kernel_means, 3, sizeof(cl_mem), (void *)&samplesInCentroid_mem_obj);


    ret |= clSetKernelArg(kernel_means, 4, sizeof(cl_int), (void *)&width);
  
    ret |= clSetKernelArg(kernel_means, 5, sizeof(cl_int), (void *)&height);

    ret |= clSetKernelArg(kernel_means, 6, sizeof(cl_int), (void *)&numCentr);
  
    ret |= clSetKernelArg(kernel_means, 7, sizeof(cl_int), (void *)&seed);
   
  
    ret |= clSetKernelArg(kernel_writeBack, 0, sizeof(cl_mem), (void *)&image_mem_obj_out);
    ret |= clSetKernelArg(kernel_writeBack, 1, sizeof(cl_mem), (void *)&centroidIndex_mem_obj);
    ret |= clSetKernelArg(kernel_writeBack, 2, sizeof(cl_mem), (void *)&centroids_mem_obj);
    ret |= clSetKernelArg(kernel_writeBack, 3, sizeof(cl_int), (void *)&width);
    ret |= clSetKernelArg(kernel_writeBack, 4, sizeof(cl_int), (void *)&height);
    ret |= clSetKernelArg(kernel_writeBack, 5, sizeof(cl_int), (void *)&numCentr);


    // "s"cepec, "stevilka argumenta, velikost podatkov, kazalec na podatke

	// "s"cepec: zagon
    const size_t local_2 = WORKGROUP_SIZE_2;
    const size_t groups_2 = ((numCentr-1)/ local_2+1);
    const size_t global_2 = local_2 * groups_2;
    for (int i = 0; i < iterations; i++) {
        ret = clEnqueueNDRangeKernel(command_queue, kernel_nearest, 2, NULL,						
                                    global_item_size, local_item_size, 0, NULL, NULL);
        
        ret = clEnqueueNDRangeKernel(command_queue, kernel_means, 1, NULL,						
                                    &global_2, &local_2, 0, NULL, NULL);
    }

    ret = clEnqueueNDRangeKernel(command_queue, kernel_writeBack, 2, NULL,
                                global_item_size, local_item_size, 0, NULL, NULL);
    
    printf("WRITE BACK: %d\n", ret);
			// vrsta, "s"cepec, dimenzionalnost, mora biti NULL, 
			// kazalec na "stevilo vseh niti, kazalec na lokalno "stevilo niti, 
			// dogodki, ki se morajo zgoditi pred klicem																						


    ret = clEnqueueReadBuffer(command_queue, image_mem_obj_out, CL_TRUE, 0,						
							 sizeof(unsigned char) * pitch * height, imageOut, 0, NULL, NULL);

                            				

    double endTime = omp_get_wtime();
    gpuTime = endTime - startTime;

    free(centroidIndex);
    free(centroidMean);
    free(centroids);
    free(samplesInCentroid);
    
    
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel_nearest);
    ret = clReleaseKernel(kernel_means);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(image_mem_obj_in);
    ret = clReleaseMemObject(centroids_mem_obj);
    ret = clReleaseMemObject(image_mem_obj_out);
    ret = clReleaseMemObject(samplesInCentroid_mem_obj);
    ret = clReleaseMemObject(centroidMean_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
}



int main () {


    printf("K-means clustering image compression\n");
    printf("---------------------------------------\n");
    printf("K = %d\t ITERATIONS = %d\n", CENTROID_NUM, ITERATIONS);
    FIBITMAP *imageBitmap = FreeImage_Load(FIF_PNG, "../images/city-3840x2160.png", 0);
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



    k_means_gpu(imageIn, imageOut, width, height, pitch, CENTROID_NUM, ITERATIONS);
    dst = FreeImage_ConvertFromRawBits(imageOut, width, height, pitch,
		32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);
	FreeImage_Save(FIF_PNG, dst, "../gpu-out/city2-3840x2160.png", 0);


    printf("GPU time: %f\n", gpuTime);
    
    free(imageIn);
    free(imageOut);

    return 0;
}


