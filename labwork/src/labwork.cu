#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#define ACTIVE_THREADS 4

int main(int argc, char **argv) {
    printf("USTH ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;
    std::string inputFilename2 ;
    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2 ) {

	    if (lwNum == 6){

        	inputFilename =  (std::string(argv[2]));
	       	labwork.loadInputImage(inputFilename);
	        inputFilename2 = (std::string(argv[3]));
		labwork.loadInputImage2(inputFilename2);
	    }else {
        	inputFilename = std::string(argv[2]);
	       	labwork.loadInputImage(inputFilename);
	    }
    }
    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum) {
        case 1:
            labwork.labwork1_CPU();
            labwork.saveOutputImage("labwork2-cpu-out.jpg");
            printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
            labwork.labwork1_OpenMP();
	    labwork.saveOutputImage("labwork2-openmp-out.jpg");
	    printf("labwork 1 openMP ellapsed %.1fms\n",lwNum, timer.getElapsedTimeInMilliSec());
            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:
            labwork.labwork1_CPU();
            printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
	    timer.start() ;
            labwork.labwork3_GPU();
            labwork.saveOutputImage("labwork3-gpu-out.jpg");
	    printf("labwork 3 GPU ellapsed %.1fms\n",lwNum, timer.getElapsedTimeInMilliSec());

            break;
        case 4:
	    timer.start() ;
            labwork.labwork3_GPU();
	    printf("labwork 3 GPU ellapsed %.1fms\n",lwNum, timer.getElapsedTimeInMilliSec());

           timer.start() ;
	    labwork.labwork4_GPU();
            labwork.saveOutputImage("labwork4-gpu-out.jpg");
	    printf("labwork 4 GPU ellapsed %.1fms\n",lwNum, timer.getElapsedTimeInMilliSec());

            break;
        case 5:
	    timer.start() ;
            labwork.labwork5_CPU();
            labwork.saveOutputImage("labwork5-cpu-out.jpg");
	    printf("labwork 5 CPU ellapsed %.1fms\n",lwNum, timer.getElapsedTimeInMilliSec());
	    timer.start() ;
            labwork.labwork5_GPU();
            labwork.saveOutputImage("labwork5-gpu-out.jpg");
	    printf("labwork 5 GPU ellapsed %.1fms\n",lwNum, timer.getElapsedTimeInMilliSec());

            break;
        case 6:
            labwork.labwork6_GPU();
            labwork.saveOutputImage("labwork6-gpu-out.jpg");
            break;
        case 7:
            labwork.labwork7_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
            labwork.labwork8_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
            labwork.labwork9_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork9-gpu-out.jpg");
            break;
        case 10:
            labwork.labwork10_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork10-gpu-out.jpg");
            break;
    }
    printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
}

void Labwork::loadInputImage(std::string inputFileName){
    inputImage = jpegLoader.load(inputFileName);
}


void Labwork::loadInputImage2(std::string inputFileName){
    inputImage2 = jpegLoader.load(inputFileName);
}

void Labwork::saveOutputImage(std::string outputFileName) {
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));

   #pragma omp parallel for
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }                                                                                                                   }	
}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void Labwork::labwork2_GPU() {
    int nDevices = 0;
    // get all devices
    cudaGetDeviceCount(&nDevices);
    printf("Number total of GPU : %d\n\n", nDevices);
    for (int i = 0; i < nDevices; i++){
        // get informations from individual device
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

//	if !(prop == NULL) {
		printf("Device name : %s \n",prop.name);
		printf("  Core info :\n") ;
		printf("    Clock Rate (in kHz) : %d \n",prop.clockRate);
		printf("    Core counts : %d \n", getSPcores(prop));
		printf("    MP count : %d \n",prop.multiProcessorCount);
		printf("    warp size : %d \n", prop.warpSize);
		printf("  Memory info : \n");
		printf("    Clock Rate : %d",prop.memoryClockRate);
		printf("    Bus width : %d\n\n",prop.memoryBusWidth);
		//printf("    Bus bandwidth : ");
//	}
    }

}
__global__ void grayscale(uchar3 *input, uchar3 *output){
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	output[tidx].x = (input[tidx].x + input[tidx].y +input[tidx].z) / 3;
	output[tidx].z = output[tidx].y = output[tidx].x;	
}


void Labwork::labwork3_GPU() {

    // Calculate number of pixels
	int pixelCount =  inputImage->width * inputImage->height ; 
        //allocate memory for the output on the host
	outputImage = static_cast<char *>(malloc(pixelCount * 3));  
    // Allocate CUDA memory
	uchar3 *devInput ;
	uchar3 *devOutput ;    
	cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
	cudaMalloc(&devOutput, pixelCount * sizeof(uchar3));
    // Copy CUDA Memory from CPU to GPU
	cudaMemcpy(devInput, inputImage->buffer,pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);    // Processing
	int blockSize = 64 ;
	int numBlock = pixelCount / blockSize ;
	grayscale<<<numBlock,blockSize>>>(devInput,devOutput) ;
    // Copy CUDA Memory from GPU to CPU
	cudaMemcpy(outputImage, devOutput,pixelCount * sizeof(uchar3),cudaMemcpyDeviceToHost);

    // Cleaning
	cudaFree(devInput) ;
	cudaFree(devOutput) ;

}


__global__ void grayscale2(uchar3 *input, uchar3 *output, int width, int height){
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y;
	int rowIdx = tidy *width ;
	int tid = tidx + rowIdx ;
	if ((tidx < width) and (tidy< height)){
		output[tid].x = (input[tid].x + input[tid].y +input[tid].z) / 3;
 		output[tid].z = output[tid].y = output[tid].x;
	}
} 


void Labwork::labwork4_GPU() {

 	// Calculate number of pixels
	int pixelCount =  inputImage->width * inputImage->height ;
	 //allocate memory for the output on the host
	outputImage = static_cast<char *>(malloc(pixelCount * 3));

	// Allocate CUDA memory
	uchar3 *devInput ;
        uchar3 *devOutput ; 
	cudaMalloc(&devInput, pixelCount * sizeof(uchar3)); 
	cudaMalloc(&devOutput, pixelCount * sizeof(uchar3));

    // Copy CUDA Memory from CPU to GPU
        cudaMemcpy(devInput, inputImage->buffer,pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);
    // Processing
	dim3 blockSize = dim3(32,32);
	int numBlockx = inputImage-> width / (blockSize.x) ;
	int numBlocky = inputImage-> height / (blockSize.y) ;

	if ((inputImage-> width % (blockSize.x)) > 0) {
		numBlockx++ ;
	}
	if ((inputImage-> height % (blockSize.y)) > 0){
		numBlocky++ ;
	}
	dim3 gridSize = dim3(numBlockx,numBlocky) ;

        grayscale2<<<gridSize,blockSize>>>(devInput,devOutput, inputImage->width, inputImage->height) ;
	// Copy CUDA Memory from GPU to CPU
	cudaMemcpy(outputImage, devOutput,pixelCount * sizeof(uchar3),cudaMemcpyDeviceToHost);
	// Cleaning
	cudaFree(devInput) ;
	cudaFree(devOutput) ;

}


__global__ void gaussianBlur_ss_SM(uchar3 *input, uchar3 *output, int width, int height){
	const int kernel[7][7] = {
			{0,0,1,2,1,0,0},
			{0,3,13,22,13,3,0},
			{1,13,59,97,59,13,1},
			{2,22,97,159,97,22,2},
			{1,13,59,97,59,13,1},
			{0,3,13,22,13,3,0},
			{0,0,1,2,1,0,0}
			} ;

        int tidx = (threadIdx.x + blockIdx.x * blockDim.x)-3;
        int tidy = (threadIdx.y + blockIdx.y * blockDim.y)-3;
        int rowIdx = (tidy+3) *width ;
        int tid = tidx +3 + rowIdx ;
	int tempx = 0 ;
	int tempy = 0;
	int tempz = 0;
	int i ;
	int j ;
	int somme = 0 ;
    
	if ((tidx < width-4) and (tidx > 2) and (tidy< height-4) and (tidy>2)){
	for  (i=0;i<7;i++){
		for (j=0;j<7;j++){
			tempx += (kernel[i][j]) * (input[(tidx+i)+((tidy+j)*width)].x) ; 
			tempy += (kernel[i][j]) * (input[(tidx+i)+((tidy+j)*width)].y) ; 
			tempz += (kernel[i][j]) * (input[(tidx+i)+((tidy+j)*width)].z) ; 
			somme += kernel[i][j] ;
		}
	}
	tempx = tempx / somme ;
	tempy = tempy / somme ;
	tempz = tempz / somme ;
        output[tid].x = tempx ;                                                                         
	output[tid].y = tempy ;
	output[tid].z = tempz ;
        }
}


__global__ void gaussianBlur_avec_SM(uchar3 *input, uchar3 *output, int width, int height){
	 const int kernel[7][7] = {
			{0,0,1,2,1,0,0},
                        {0,3,13,22,13,3,0},
                        {1,13,59,97,59,13,1},
                        {2,22,97,159,97,22,2},
                        {1,13,59,97,59,13,1},
                        {0,3,13,22,13,3,0},
                        {0,0,1,2,1,0,0} 
	} ;  
	__shared__ int redtile[16][16] ;
	__shared__ int greentile[16][16] ;
	__shared__ int bluetile[16][16] ;

        int tidx = (threadIdx.x + blockIdx.x * blockDim.x)-3;
        int tidy = (threadIdx.y + blockIdx.y * blockDim.y)-3;
        int rowIdx = (tidy+3) *width ;
        int tid = tidx +3 + rowIdx ;
        int tempx = 0 ;
        int tempy = 0;
        int tempz = 0; 
        int i ;
        int j ;
        int somme = 0 ;
        redtile[threadIdx.x][threadIdx.y] = input[tid].x ;   
        greentile[threadIdx.x][threadIdx.y] = input[tid].y ;   
        bluetile[threadIdx.x][threadIdx.y] = input[tid].z ;   
	__syncthreads() ;
        if ((tidx < width-4) and (tidx > 2) and (tidy< height-4) and (tidy>2)){
	        for  (i=0;i<7;i++){
 	               for (j=0;j<7;j++){
	                      tempx += (kernel[i][j]) * (redtile[threadIdx.x+i][threadIdx.y+j]) ;
                              tempy += (kernel[i][j]) * (greentile[threadIdx.x+i][threadIdx.y +j ]) ;
                              tempz += (kernel[i][j]) * (bluetile[threadIdx.x+i][threadIdx.y + j]) ;
                              somme += kernel[i][j] ;
                       }
                 }
        	tempx = tempx / somme ;
	        tempy = tempy / somme ;
        	tempz = tempz / somme ;
	        output[tid].x = tempx ;
        	output[tid].y = tempy ;
	        output[tid].z = tempz ;
        }
	__syncthreads() ; 

}

void Labwork::labwork5_CPU() {
        int kernel[7][7] = {
                        {0,0,1,2,1,0,0},
                        {0,3,13,22,13,3,0},
                        {1,13,59,97,59,13,1},
                        {2,22,97,159,97,22,2},
                        {1,13,59,97,59,13,1},
                        {0,3,13,22,13,3,0},
                        {0,0,1,2,1,0,0}
                        } ;   

    int pixelCount = inputImage->width * inputImage->height;
    int width = inputImage -> width ;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int k = 0; k < 100; k++) {     // let's do it 100 times, otherwise it's too fast!
        for (int n = 0;n < pixelCount; n++) {
		 if ((n*3 > 3*width) and ( n % width > 3) and (n % width < width -4) and (n*3 < (inputImage->height)-(4*width))){
			 for  (int i=0;i<7;i++){
	        	         for (int j=0;j<7;j++){
					 int temp = (n*3 -9)-(3*inputImage->width) ;
		                         outputImage[n*3] += (kernel[i][j]) * ((int)(inputImage->buffer[temp +i + j*3*(inputImage->width)])) ;
		                         outputImage[n*3+1] += (kernel[i][j]) * ((int)(inputImage->buffer[temp+i +j*3*(inputImage->width)])) ;
	                        	 outputImage[n*3+2] += (kernel[i][j]) * ((int)(inputImage->buffer[temp+i +j*3*(inputImage->width)])) ;
        	        	}
	        	}        
		}
	}
    }

}


void Labwork::labwork5_GPU(){
        int pixelCount =  inputImage->width * inputImage->height ;
       //allocate memory for the output on the host
       outputImage = static_cast<char *>(malloc(pixelCount * 3));
       // Allocate CUDA memory
       uchar3 *devInput ;
       uchar3 *devOutput ;
       cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
       cudaMalloc(&devOutput, pixelCount * sizeof(uchar3));
       // Copy CUDA Memory from CPU to GPU
       cudaMemcpy(devInput, inputImage->buffer,pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);
       // Processing
       dim3 blockSize = dim3(16,16);
       int numBlockx = inputImage-> width / (blockSize.x) ;
       int numBlocky = inputImage-> height / (blockSize.y) ;
       if ((inputImage-> width % (blockSize.x)) > 0) {
	       numBlockx++ ;
       }
       if ((inputImage-> height % (blockSize.y)) > 0){
	       numBlocky++ ;
	}
       dim3 gridSize = dim3(numBlockx,numBlocky) ;
       gaussianBlur_avec_SM<<<gridSize,blockSize>>>(devInput,devOutput, inputImage->width, inputImage->height) ;
       // Copy CUDA Memory from GPU to CPU
       cudaMemcpy(outputImage, devOutput,pixelCount * sizeof(uchar3),cudaMemcpyDeviceToHost);
       // Cleaning
       cudaFree(devInput) ;
       cudaFree(devOutput) ;
}

__global__ void binarization(uchar3 *input,uchar3 *output,int width, int threshold){
	        int tidx = (threadIdx.x + blockIdx.x * blockDim.x);
                int tidy = (threadIdx.y + blockIdx.y * blockDim.y);
                int rowIdx = tidy *width ;
		int tid = tidx + rowIdx ;
		if (input[tid].x > threshold){
			output[tid].x = 255 ;
		} else {
			output[tid].x = 0 ;
		}
		output[tid].z = output[tid].y = output[tid].x ;
}

__global__ void brightness_control(uchar3 *input,uchar3 *output, int width, bool increase, int percentage){
                int tidx = (threadIdx.x + blockIdx.x * blockDim.x);
                int tidy = (threadIdx.y + blockIdx.y * blockDim.y);
                int rowIdx = tidy *width ;
                int tid = tidx + rowIdx ;
		int incValue = percentage*255/100 ;
		int temp ;
		if (increase){
			temp = input[tid].x + incValue ;
			if (temp > 255){ output[tid].x = 255 ;}
			else{ output[tid].x = temp ;}
		} else{
	                temp = input[tid].x + incValue ;
                        if (temp > 255){ output[tid].x = 255 ;}
			else{ output[tid].x = temp ;}
		}
		output[tid].z = output[tid].y = output[tid].x ;

}


__global__ void blending(uchar3 *input1, uchar3 *input2, uchar3 *output,int width,int weight){
                int tidx = (threadIdx.x + blockIdx.x * blockDim.x);
                int tidy = (threadIdx.y + blockIdx.y * blockDim.y);
                int rowIdx = tidy *width ;
                int tid = tidx + rowIdx ;
		output[tid].x =  (weight*input1[tid].x + (100 - weight)*input2[tid].x)/100 ;
		output[tid].y =  (weight*input1[tid].y + (100 - weight)*input2[tid].y)/100 ;
		output[tid].z =  (weight*input1[tid].z + (100 - weight)*input2[tid].z)/100 ;
}


void Labwork::labwork6_GPU() {

	int threshold = 128 ;
        int pixelCount = inputImage->width * inputImage->height ;
        //allocate memory for the output on the host
        outputImage = static_cast<char *>(malloc(pixelCount * 3));
        // Allocate CUDA memory
        uchar3 *devInput ;
	uchar3 *devOutput ;
	uchar3 * devInput2 ;
	cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
        cudaMalloc(&devOutput, pixelCount * sizeof(uchar3));
        cudaMalloc(&devInput2, pixelCount * sizeof(uchar3));
        // Copy CUDA Memory from CPU to GPU
        cudaMemcpy(devInput, inputImage->buffer,pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);
        cudaMemcpy(devInput2, inputImage2->buffer,pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);
        // Processing
        dim3 blockSize = dim3(16,16);
        int numBlockx = inputImage-> width / (blockSize.x) ;
        int numBlocky = inputImage-> height / (blockSize.y) ;
        if ((inputImage-> width % (blockSize.x)) > 0) {
	        numBlockx++ ;
	} 
        if ((inputImage-> height % (blockSize.y)) > 0){
		numBlocky++ ;
	}
       dim3 gridSize = dim3(numBlockx,numBlocky) ;
//       binarization<<<gridSize,blockSize>>>(devInput,devOutput, inputImage->width,threshold) ;
//       brightness_control<<<gridSize,blockSize>>>(devInput,devOutput, inputImage->width,1,50) ;
	blending<<<gridSize,blockSize>>>(devInput,devInput2,devOutput, inputImage->width, 50) ;
       // Copy CUDA Memory from GPU to CPU
       cudaMemcpy(outputImage, devOutput,pixelCount * sizeof(uchar3),cudaMemcpyDeviceToHost);
       // Cleaning
       cudaFree(devInput) ;
       cudaFree(devOutput) ;

}


__global__ void  grayscale_stretch(uchar3 *input,uchar3*output,int width){

}


void Labwork::labwork7_GPU() {
       int pixelCount = inputImage->width * inputImage->height ;
        //allocate memory for the output on the host
        outputImage = static_cast<char *>(malloc(pixelCount * 3));
        // Allocate CUDA memory
        uchar3 *devInput ;
	uchar3 *devOutput ;
	cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
        cudaMalloc(&devOutput, pixelCount * sizeof(uchar3));
        // Copy CUDA Memory from CPU to GPU
        cudaMemcpy(devInput, inputImage->buffer,pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);
        // Processing
        dim3 blockSize = dim3(16,16);
        int numBlockx = inputImage-> width / (blockSize.x) ;
        int numBlocky = inputImage-> height / (blockSize.y) ;
        if ((inputImage-> width % (blockSize.x)) > 0) {
	        numBlockx++ ;
	} 
        if ((inputImage-> height % (blockSize.y)) > 0){
		numBlocky++ ;
	}
       dim3 gridSize = dim3(numBlockx,numBlocky) ;
       //grayscale_strectch<<<gridSize,blockSize>>>(devInput,devOutput, inputImage->width) ;
       
       // Copy CUDA Memory from GPU to CPU
       cudaMemcpy(outputImage, devOutput,pixelCount * sizeof(uchar3),cudaMemcpyDeviceToHost);
       // Cleaning
       cudaFree(devInput) ;
       cudaFree(devOutput) ;

}

void Labwork::labwork8_GPU() {

}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU(){
}


























