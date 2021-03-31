// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

// Includes, CUDA
#include <cuda_runtime.h>

// Includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples
#include "common.h"

// Define constants
#define line_max 200000
#define char_max 40
#define ascii_nb 128
#define threads_per_block 1024
#define borne_inf 32
#define borne_sup 126
#define first_up 65
#define last_up 90

////////////////////////////////////////////////////////////////////////////////
// Declarations
////////////////////////////////////////////////////////////////////////////////
void Histogram(char* inputFileName, char* outputFileName);

void writeOutputCSV(unsigned long long result[ascii_nb], char* outputFileName);

void processBatchInKernel(  char** d_data,
                            char h_data[line_max][char_max],
                            int nbLine,
                            size_t pitch,
                            int lineSize,
                            unsigned long long** d_result,
                            int resultSize,
                            unsigned long long resultStorage[ascii_nb]);

////////////////////////////////////////////////////////////////////////////////
//! Kernel function to execute the computation in threads using only Global Memory
//! @param d_data  input data in global memory
//! @param d_result  output result as array in global memory
//! @param nbLine  input size of the data in global memory
//! @param pitch  input pitch size of in the data global memory
////////////////////////////////////////////////////////////////////////////////

__global__ 
void kernelHistoGlobal(char* d_data, unsigned long long* d_result, int nbLine, size_t pitch) {
    
    const unsigned int tidb = threadIdx.x;
    const unsigned int ti = blockIdx.x*blockDim.x + tidb;
    unsigned long long unit = 1;
    
    // Each thread compute a single line of the data
    if (ti < nbLine) {
		char* line = (char *)((char*)d_data + ti * pitch);
		int index = 0;
		int currentLetter = line[index];

        // Each char is converted to int and adds a unit to the corresponding index in the global memory
		while (currentLetter > 0) {
	    	atomicAdd(&d_result[currentLetter], unit);
	    	index++;
	    	currentLetter = line[index];
		}
    }
}
                            

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
	
    // Process the arguments of the call
	int c;
	char *inputFileName = NULL;
	char *outputFileName = NULL;
	while ((c = getopt (argc, argv, "i:o:h")) != -1)
		switch(c) {
			case 'i':
				inputFileName = optarg;
				break;
			case 'o':
				outputFileName = optarg;
				break;
		}

	printf("\n%s Starting...\n\n", argv[0]);

    // Start timer
    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Do the computation
	Histogram(inputFileName, outputFileName);

    // Stop timer
    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    printf("Input file processed successfully.\n");
    if (outputFileName) {
        printf("Check results in %s.\n\n", outputFileName);
    } else {
        printf("Check results in out.csv.\n\n");
    }

	exit(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//! Generate the Histogram
//! @param inputFileName input name of the file to process
//! @param outputFileName input name of the file to output the histogram in
////////////////////////////////////////////////////////////////////////////////
void Histogram(char* inputFileName, char* outputFileName) {
    
    // Compute result and data sizes
    unsigned int resultSize = ascii_nb * sizeof(unsigned long long);
    unsigned int lineSize = char_max * sizeof(char);

    // Load input file
    FILE *inputFile = NULL;
    inputFile = fopen(inputFileName, "r");
    if (!inputFile) {
        printf("Wrong input file\n");
		exit(EXIT_FAILURE);
    }

    // Allocate device memory
    char* d_data;
    unsigned long long* d_result;
    size_t pitch;
    checkCudaErrors(cudaMallocPitch((void **) &d_data, &pitch, lineSize, line_max));
    checkCudaErrors(cudaMalloc((void **) &d_result, resultSize));

    // Allocate host memory
    char h_data[line_max][char_max];
    unsigned long long resultStorage[ascii_nb];
    char str[char_max];
    int nbLine = 0;
    int batchNum = 1;
    
    // Iterate over the file's lines
    while (fgets(str, char_max, inputFile)) {
	
        // Batch size reached, send data to kernel for process
		if (nbLine == line_max) {

            printf("Batch N°%i: %i lines. \n", batchNum, nbLine);
	    	processBatchInKernel(&d_data, h_data, nbLine, pitch, lineSize, &d_result, resultSize, resultStorage);
            
            nbLine = 0;
            batchNum++;
		}

        // Add current line to the Batch
        strcpy(h_data[nbLine], str);
        nbLine++;
    }
    
    // Process last Batch (< line_max lines)
    printf("Batch N°%i: %i lines. \n", batchNum, nbLine);
    processBatchInKernel(&d_data, h_data, nbLine, pitch, lineSize, &d_result, resultSize, resultStorage);
    
    fclose(inputFile);
    
    // Write the output
    writeOutputCSV(resultStorage, outputFileName);

    // Cleanup memory
    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaFree(d_result));
}

////////////////////////////////////////////////////////////////////////////////
//! Send batch data to kernel and store the output in resultStorage
//! @param d_data input pointer to the allocated memory for the input data on the device
//! @param h_data input the list of strings to process
//! @param nbLine input number of lines to process for the current batch
//! @param pitch input pitch size of the array in the device 
//! @param lineSize input size of a single line
//! @param d_result input pointer to the allo
//! @param resultSize input pointer to the allocated memory for the output data on the device
//! @param resultStorage output result of the computation as an array
////////////////////////////////////////////////////////////////////////////////

void processBatchInKernel(  char** d_data,
                            char h_data[line_max][char_max],
                            int nbLine,
                            size_t pitch,
                            int lineSize,
                            unsigned long long** d_result,
                            int resultSize,
                            unsigned long long resultStorage[ascii_nb]) {
    // Allocate host memory for result
    unsigned long long h_result[ascii_nb];

    // Setup execution parameters
    dim3  grid((nbLine + threads_per_block - 1) / threads_per_block, 1, 1);
    dim3  threads(threads_per_block, 1, 1);

    // Copy data to device
    checkCudaErrors(cudaMemcpy2D(*d_data, pitch, h_data, lineSize, lineSize, line_max, cudaMemcpyHostToDevice));
    
    // Execute the kernel
    kernelHistoGlobal<<< grid, threads, 0 >>>(*d_data, *d_result, nbLine, pitch);

    getLastCudaError("Kernel execution failed");
    
    // Copy result from device to host
    checkCudaErrors(cudaMemcpy(&h_result, *d_result, resultSize, cudaMemcpyDeviceToHost));

    // Copy the result into resultStorage
    for (int index = 0; index < ascii_nb; index++) {
        resultStorage[index] = h_result[index];
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Write the given output to the CSV file
//! @param result input the given ouput of the computations as an array of int
//! @param outputFileName input file name to write in
////////////////////////////////////////////////////////////////////////////////

void writeOutputCSV(unsigned long long result[ascii_nb], char* outputFileName) {

    // Load output file
	FILE *outputFile;
	char asciiChar;
    if (outputFileName) {
        outputFile = fopen(outputFileName, "w+");
    } else {
        outputFile = fopen("outputHisto.csv", "w+");
    }
	
    // Write the result
	for (int index = borne_inf; index <= borne_sup; index++) {

        if (index >= first_up && index <= last_up) {
            // Add uppercase count to char count
            result[index + borne_inf] += result[index];
        } else {
            // Write count in file
            asciiChar = index;
		    fprintf(outputFile, "%c: %llu\n", asciiChar, result[index]);
        }

	}

	fclose(outputFile);
}


