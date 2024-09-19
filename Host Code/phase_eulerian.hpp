#include <chrono>

#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <fstream>
#include <cstdio>

#include "hls_x_complex.h"
#include "hls_stream.h"
#include "hls_math.h"

#include <ap_fixed.h>
#include "ap_int.h"


#define BUFFER_SIZE 32
#define DATAWIDTH 224
#define VECTOR_SIZE (DATAWIDTH / 32)
typedef ap_uint<DATAWIDTH> myuint_t;

const int N = 100;

#include "helpers.hpp"


#include "time.h"

#include <iostream>
#include <fstream>

#include "event_timer.hpp"






#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1


#include <vector>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <CL/cl2.hpp>


#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }

static const std::string error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %d Device result = %d\n";


// Function to convert ap_uint<32> to float without memcpy
float uint32_to_float(ap_uint<32> input) {

    union {
        uint32_t i;
        float f;
    } converter;
    converter.i = input.to_uint();
    return converter.f;
}

// Function to convert float to ap_uint<32> without memcpy
ap_uint<32> float_to_uint32(float input) {
    union {
        uint32_t i;
        float f;
    } converter;
    converter.f = input;
    return ap_uint<32>(converter.i);
}



// Function to convert x_complex<floatT> to myuint_t
myuint_t complex_to_uint320(hls::x_complex<floatT> real, hls::x_complex<floatT> imag) {
	myuint_t result = 0;
	#pragma omp parallel
    for (int i = 0; i < VECTOR_SIZE; i++) {
        ap_uint<32> real_val;
        ap_uint<32> imag_val;
        floatT real_part = real.real();
        floatT imag_part = imag.imag();
        memcpy(&real_val, &real_part, sizeof(real_val));
        memcpy(&imag_val, &imag_part, sizeof(imag_val));

        result.range(32 * (i + 1) - 1, 32 * i) = real_val;
        result.range(32 * (i + 1) - 1, 32 * i) = imag_val;

        //std::cout<< "complex to uint320 \nreal_val="<<real_val<<"\nimag_val="<<imag_val<<"\n";
    }
    return result;
}
/*
// Function to convert x_complex<floatT> to myuint_t
myuint_t complex_to_uint320R_2(hls::x_complex<floatT> real[VECTOR_SIZE]) {
	myuint_t result = 0;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        ap_uint<32> real_val;
        floatT real_part = real[i].real();
        memcpy(&real_val, &real_part, sizeof(real_val));
        result.range(32 * (i + 1) - 1, 32 * i) = real_val;


        //std::cout<< "complex to uint320 \nreal_val="<< real_val <<"\n";
    }
    return result;
}

// Function to convert x_complex<floatT> to myuint_t
myuint_t complex_to_uint320I_2(hls::x_complex<floatT> imag[VECTOR_SIZE]) {
	myuint_t result = 0;

	for (int i = 0; i < VECTOR_SIZE; i++) {
        ap_uint<32> imag_val;
        floatT imag_part = imag[i].imag();
        memcpy(&imag_val, &imag_part, sizeof(imag_val));

        result.range(32 * (i + 1) - 1, 32 * i) = imag_part;


        //std::cout<< "complex to uint320 \nimag_val="<< imag_val <<"\n";
    }
    return result;
}
*/

myuint_t complex_to_uint320R_2(hls::x_complex<floatT> real[VECTOR_SIZE]) {
    myuint_t result = 0;

    for (int i = 0; i < VECTOR_SIZE; i++) {
        ap_uint<32> real_val = float_to_uint32(real[i].real());
        result.range(32 * (i + 1) - 1, 32 * i) = real_val;
        // Optional debug output
        // std::cout << "complex to uint320 \nreal_val = " << real_val << "\n";
    }

    return result;
}

myuint_t complex_to_uint320I_2(hls::x_complex<floatT> imag[VECTOR_SIZE]) {
    myuint_t result = 0;

    for (int i = 0; i < VECTOR_SIZE; i++) {
        ap_uint<32> imag_val = float_to_uint32(imag[i].imag());
        result.range(32 * (i + 1) - 1, 32 * i) = imag_val;

        // Optional debug output
        // std::cout << "complex to uint320 \nimag_val = " << imag_val << "\n";
    }

    return result;
}

// Function to convert myuint_t to x_complex<floatT>
void myuint_to_complex(myuint_t input, hls::x_complex<floatT>& real, hls::x_complex<floatT>& imag) {

	#pragma omp parallel
    for (int i = 0; i < VECTOR_SIZE; i++) {
        ap_uint<32> real_val = input.range(32 * (i + 1) - 1, 32 * i);
        ap_uint<32> imag_val = input.range(32 * (i + 1) - 1, 32 * i);
        floatT real_part;
        floatT imag_part;
        memcpy(&real_part, &real_val, sizeof(real_part));
        memcpy(&imag_part, &imag_val, sizeof(imag_part));
        real.real() = real_part;
        imag.imag() = imag_part;
    }
}



/*
// Function to convert myuint_t to x_complex<floatT> (real and imag)
void myuint_to_complex2(myuint_t input, hls::x_complex<floatT> real[], hls::x_complex<floatT> imag[]) {

    for (int i = 0; i < VECTOR_SIZE; i++) {
        ap_uint<32> real_val = input.range(32 * (i + 1) - 1, 32 * i);
        ap_uint<32> imag_val = input.range(32 * (i + 1) - 1, 32 * i);

        floatT real_part, imag_part;
        memcpy(&real_part, &real_val, sizeof(real_part));
        memcpy(&imag_part, &imag_val, sizeof(imag_part));

        real[i].real(real_part);
        imag[i].imag(imag_part);

        std::cout <<"magn real_part = " << real_part <<"\n";
        std::cout <<"magn imag_part = " << imag_part <<"\n";
    }
}
*/

void myuint_to_complex2(myuint_t input, hls::x_complex<floatT> real[], hls::x_complex<floatT> imag[]) {

    for (int i = 0; i < VECTOR_SIZE; i++) {
        // Extract the real and imaginary parts from the input
        ap_uint<32> real_val = input.range(32 * (i + 1) - 1, 32 * i);
        ap_uint<32> imag_val = input.range(32 * (i + 1) - 1, 32 * i);

        // Convert the extracted ap_uint<32> values to float using uint32_to_float
        floatT real_part = uint32_to_float(real_val);
        floatT imag_part = uint32_to_float(imag_val);

        // Set the real and imaginary parts in the complex arrays
        real[i].real(real_part);
        imag[i].imag(imag_part);

        // Debug output (optional)
       // std::cout << "magn real_part = " << real_part << "\n";
       // std::cout << "magn imag_part = " << imag_part << "\n";
    }
}



// Function to convert std::vector<std::vector<int>> to std::vector<myuint_t>
std::vector<myuint_t> int_to_uint320(const std::vector<std::vector<int>>& intVector) {
    int numRows = intVector.size();
    int numCols = intVector[0].size(); // Assuming all rows have the same number of columns


    std::vector<myuint_t> uint320Array(numRows);
	#pragma omp parallel for collapse(2)
    for (int i = 0; i < numRows; i++) {
        myuint_t result = 0;
        for (int j = 0; j < numCols; j++) {
            ap_uint<32> val = intVector[i][j];
            result.range(32 * (j + 1) - 1, 32 * j) = val;
        }
        uint320Array[i] = result;
    }

    return uint320Array;
}

myuint_t int_to_uint320v2(int integer[VECTOR_SIZE]) {
	myuint_t result = 0;

    for (int i = 0; i < VECTOR_SIZE; i++) {
        ap_uint<32> uintv;
        int temp = integer[i];
        memcpy(&uintv, &temp, sizeof(uintv));
        result.range(32 * (i + 1) - 1, 32 * i) = uintv;

    }
    return result;
}

template<int W>
void print_ap_uint(ap_uint<W> value) {
    std::cout << "ap_uint<" << W << ">: ";
    for (int i = W - 1; i >= 0; --i) {
        std::cout << value[i];
    }
    std::cout << std::endl;
}


static const int size1 = 100;

void phase_eulerian(std::string xclbinFilename,
			std::vector<std::vector<std::vector<uint8_t>>>& vid,
		    double& alpha,
		    double& cutoff_freq_low,
		    double& cutoff_freq_high,
		    std::vector<std::vector<std::vector<uint8_t>>>& final_out, // final results out
		    double& sigma,
		    bool& attenuateOtherFreq,
		    std::string& pyrType,
		    std::string& temporalFilter)
{
	int width = 100;//100;
	int height = 100;//100; // here not * 3
	int numberOfFrames =  60;

 	
    	printf("Computing spatial filters\n");
    	
    	// Create a matrix of zeros with dimensions height x width
    	std::vector<std::vector<int>> zeros = createZerosMatrix<int>(height, width * 3);
    	
    	// Calculate maxSCFpyrHt
    	int ht = maxSCFpyrHt(zeros);

    	height--;
    	width--;
    	
    	
    	std::vector<int> size = {height, width};
    	std::vector<double> rVals;
    	std::vector<std::vector<std::vector<double>>> filters; 
    	
    	type_time stime_sw;
    	type_time etime_sw;
    	type_time_calc sw_duration;

    	stime_sw = std::chrono::high_resolution_clock::now();
    	/* Supporting only Octave , halfOctave */
    	if (pyrType == "octave") {
    		for (int i = 0; i <= ht; ++i) {
    			rVals.push_back(std::pow(2.0, -i));
		}
    		filters = getFilters(size, rVals, 4,1.0);
    		printf("Using octave bandwidth pyramid\n");
    	} else if (pyrType == "halfOctave") {
    		for (int i = 0; i <= ht * 2; ++i) {
		    rVals.push_back(std::pow(2.0, -0.5 * i));
		}
       	filters = getFilters(size, rVals, 8, 0.75);
       	printf("Using half octave bandwidth pyramid\n");
    	} 
    	else {
       	std::cerr << "Invalid Filter Type" << std::endl;
        	
    	}
    	
    	   
    	
    	auto result = getFilterIDX(filters);
       
    	// Access the result
    	std::vector<std::vector<std::vector<double>>> croppedFilters = std::get<0>(result);
    	std::vector<std::vector<std::vector<int>>> filtIDX = std::get<1>(result);

		#pragma omp parallel for collapse(2)
    	for(int i = 0; i < croppedFilters.size(); i++){
    		croppedFilters[i].resize(height);
    		for(int j =0; j < croppedFilters[i].size(); j++){
    			croppedFilters[i][j].resize(width ,0.0);
    		}
    	}


		#pragma omp parallel for collapse(2)
    	for(int i = 0; i < filtIDX.size(); i++){
    		for(int j =0; j < filtIDX[i].size(); j++){
    			filtIDX[i][j].resize(width ,-1);
    		}
    	}


    	std::vector<std::vector<std::vector<std::complex<double>>>> magnifiedLumaFFT(numberOfFrames,std::vector<std::vector<std::complex<double>>>(height, std::vector<std::complex<double>>(width,{0.0,0.0})));


    	int numLevels = filters.size();
    	//numLevels = 1;
        

        
    	printf("Moving video to Fourier domain\n");
        
    	std::vector<std::vector<std::vector<double>>> doubleVid;
    	doubleVid = im2single2(vid);
    	//exportFramesdouble(doubleVid,width, height);
	
	
    	std::vector<std::vector<std::vector<double>>>  originalFrames = rgb2ntsc(doubleVid);
    	//rgb2ntsc(im2single(vid(:,:,:,k)));
	
	
	
    	// Convert RGB images to complex-valued images
    	std::vector<std::vector<std::vector<std::complex<double>>>> tVid = YtoComplex(originalFrames); // originalFramesComplex = originalFrames
    	std::vector<std::vector<std::complex<double>>> tvid2(height,std::vector<std::complex<double>>(width, {0.0,0.0}));
    	std::vector<std::vector<std::complex<double>>> tvidin2(height,std::vector<std::complex<double>>(width, {0.0,0.0}));


    	std::vector<std::vector<std::vector<std::complex<double>>>> vidFFT;
		#pragma omp parallel for
    	for(int i=0; i < numberOfFrames; i++){
    		tvidin2 = tVid[i];
    		fft2(tvidin2,tvid2);
    		tVid[i] = tvid2;
    		fftshift(tVid[i]);
    		vidFFT.push_back(tVid[i]);

   	 }
    etime_sw = std::chrono::high_resolution_clock::now();
    sw_duration = etime_sw - stime_sw;
    std::cout << "SW_1: " << sw_duration.count() << " seconds." << std::endl;
    	 

    	
    	
    std::vector<std::vector<std::complex<double>>> absPyrRef;
    //std::vector<std::vector<std::complex<double>>> pyrRef;
    std::vector<std::vector<std::complex<double>>> pyrRefPhaseOrig;
	std::vector<std::vector<double>> pyrRefAngle;
	
	//std::vector<std::vector<std::complex<double>>> filterResponse;
	std::vector<std::vector<double>> pyrCurrent;
	std::vector<std::vector<double>> phaseOfFrame;
	std::vector<std::vector<std::complex<double>>> originalLevel;
	std::vector<std::vector<std::complex<double>>> tempOrig;
	
	std::vector<std::vector<std::complex<double>>> expPhaseOfFrame;
	std::vector<std::vector<std::complex<double>>> tempTransformOut;
	std::vector<std::vector<std::complex<double>>> curLevelFrame;
	
	/*-----------------------------------------------------------*/
	/* Init the timers for measurement*/
	
	type_time stime_per_level;
	type_time etime_per_level;
	type_time_calc per_level_duration;
	

	type_time stime_per_frame_fpga;
	type_time etime_per_frame_fpga;
	type_time_calc per_frame_duration_fpga;

	type_time stime_buildLevel;
	type_time etime_buildLevel;
	type_time_calc buildLevel_duration;

	type_time stime_delta;
	type_time etime_delta;
	type_time_calc delta_duration;
	
	type_time stime_filter; // not per frame
	type_time etime_filter; // not per frames
	type_time_calc filter_duration;// not per frames
	// need to add in filters time for perframe (both of them) inside the functions
	
	type_time stime_magnification;
	type_time etime_magnification;
	type_time_calc magnification_duration;
	
	type_time stime_lowpass;
	type_time etime_lowpass;
	type_time_calc lowpass_duration;
	
	type_time stime_finalResults;
	type_time etime_finalResults;
	type_time_calc finalResults_duration;

	type_time stime_hw;
	type_time etime_hw;
	type_time_calc hw_duration;
	/*-----------------------------------------------------------*/



	std::vector<std::vector<std::complex<double>>> filterResponse;
	std::vector<std::vector<std::complex<double>>> pyrRef;


	// Define size2 appropriately as per your use case.
	const int tsize2 = (100*100-1)/ VECTOR_SIZE;
	const int size2v2 = (2*100-1)/ VECTOR_SIZE;

	EventTimer et;
	 // Compute the size of array in bytes
	size_t size_vidFFT = tsize2 * sizeof(int);
	size_t size_cropped = tsize2 * sizeof(int);;
	size_t size_vidfftidx = tsize2 * sizeof(int);

	size_t temporal_filter_size = 1*sizeof(bool);
	size_t freq_low_size = 1*sizeof(floatT);
	size_t freq_high_size = 1*sizeof(floatT);
	size_t first_size = 1*sizeof(bool);
	size_t alpha_size = 1*sizeof(floatT);
	size_t attenuateOtherFreq_size = 1*sizeof(bool);

	size_t filtidx_size = size2v2 *  sizeof(int);

	size_t magnifiedLumaFFT_size = tsize2 * sizeof(int);

   std::vector<cl::Device> devices;
   cl_int err;
   cl::Context context;
   cl::CommandQueue q;
   cl::Kernel top_kernel;
   cl::Program program;
   std::vector<cl::Platform> platforms;
   bool found_device = false;

   // traversing all Platforms To find Xilinx Platform and targeted
   // Device in Xilinx Platform
   cl::Platform::get(&platforms);
   for (size_t i = 0; (i < platforms.size()) & (found_device == false); i++) {
       cl::Platform platform = platforms[i];
       std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
       if (platformName == "Xilinx") {
           devices.clear();
           platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
           if (devices.size()) {
               found_device = true;
               break;
           }
       }
   }
   if (found_device == false) {
       std::cout << "Error: Unable to find Target Device " << std::endl;
       //return EXIT_FAILURE;
       exit(EXIT_FAILURE);
   }

   std::cout << "INFO: Reading " << xclbinFilename << std::endl;
   FILE* fp;
   if ((fp = fopen(xclbinFilename.c_str(), "r")) == nullptr) {
       printf("ERROR: %s xclbin not available please build\n", xclbinFilename.c_str());
       exit(EXIT_FAILURE);
   }
   // Load xclbin
   std::cout << "Loading: '" << xclbinFilename << "'\n";
   std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
   bin_file.seekg(0, bin_file.end);
   unsigned nb = bin_file.tellg();
   bin_file.seekg(0, bin_file.beg);
   char* buf = new char[nb];
   bin_file.read(buf, nb);

   // Creating Program from Binary File
   cl::Program::Binaries bins;
   bins.push_back({buf, nb});
   bool valid_device = false;
   for (unsigned int i = 0; i < devices.size(); i++) {
       auto device = devices[i];
       // Creating Context and Command Queue for selected Device
       OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
       OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
       std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
       cl::Program program(context, {device}, bins, nullptr, &err);
       if (err != CL_SUCCESS) {
           std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
       } else {
           std::cout << "Device[" << i << "]: program successful!\n";
           OCL_CHECK(err, top_kernel = cl::Kernel(program, "top", &err));
           valid_device = true;
           break; // we break because we found a valid device
       }
   }
   if (!valid_device) {
       std::cout << "Failed to program any device found, exit!\n";
       exit(EXIT_FAILURE);
   }

   // These commands will allocate memory on the Device. The cl::Buffer objects can
   // be used to reference the memory locations on the device.
   et.add("Allocate Buffer in Global Memory");

   OCL_CHECK(err, cl::Buffer buffer_vidFFT_real(context, CL_MEM_READ_ONLY, size_vidFFT, NULL, &err));
   OCL_CHECK(err, cl::Buffer buffer_vidFFT_imag(context, CL_MEM_READ_ONLY, size_vidFFT, NULL, &err));
   OCL_CHECK(err, cl::Buffer buffer_cropped(context, CL_MEM_READ_ONLY, size_cropped, NULL, &err));
   OCL_CHECK(err, cl::Buffer buffer_vidFFTidx_real(context, CL_MEM_READ_ONLY, size_vidfftidx, NULL, &err));
   OCL_CHECK(err, cl::Buffer buffer_vidFFTidx_imag(context, CL_MEM_READ_ONLY, size_vidfftidx, NULL, &err));
   OCL_CHECK(err, cl::Buffer buffer_temporalFilter(context, CL_MEM_READ_ONLY, temporal_filter_size, NULL, &err));
   OCL_CHECK(err, cl::Buffer buffer_cutoffFreqLow(context, CL_MEM_READ_ONLY, freq_low_size, NULL, &err));
   OCL_CHECK(err, cl::Buffer buffer_cutoffFreqHigh(context, CL_MEM_READ_ONLY, freq_high_size, NULL, &err));
   OCL_CHECK(err, cl::Buffer buffer_first(context, CL_MEM_READ_ONLY, first_size, NULL, &err));
   OCL_CHECK(err, cl::Buffer buffer_alpha(context, CL_MEM_READ_ONLY, alpha_size, NULL, &err));
   OCL_CHECK(err, cl::Buffer buffer_attenuateOtherFreq(context, CL_MEM_READ_ONLY, attenuateOtherFreq_size, NULL, &err));
   OCL_CHECK(err, cl::Buffer buffer_filtIDX(context, CL_MEM_READ_ONLY, filtidx_size, NULL, &err));

   OCL_CHECK(err, cl::Buffer buffer_magnifiedLumaFFT_real(context, CL_MEM_WRITE_ONLY, magnifiedLumaFFT_size, NULL, &err));
   OCL_CHECK(err, cl::Buffer buffer_magnifiedLumaFFT_imag(context, CL_MEM_WRITE_ONLY, magnifiedLumaFFT_size, NULL, &err));

   et.finish();


   // set the kernel Arguments
   et.add("Set the Kernel Arguments");
   int narg = 0;
   OCL_CHECK(err, err = top_kernel.setArg(narg++, buffer_vidFFT_real));
   OCL_CHECK(err, err = top_kernel.setArg(narg++, buffer_vidFFT_imag));
   OCL_CHECK(err, err = top_kernel.setArg(narg++, buffer_cropped));
   OCL_CHECK(err, err = top_kernel.setArg(narg++, buffer_vidFFTidx_real));
   OCL_CHECK(err, err = top_kernel.setArg(narg++, buffer_vidFFTidx_imag));
   OCL_CHECK(err, err = top_kernel.setArg(narg++, buffer_temporalFilter));
   OCL_CHECK(err, err = top_kernel.setArg(narg++, buffer_cutoffFreqLow));
   OCL_CHECK(err, err = top_kernel.setArg(narg++, buffer_cutoffFreqHigh));
   OCL_CHECK(err, err = top_kernel.setArg(narg++, buffer_first));
   OCL_CHECK(err, err = top_kernel.setArg(narg++, buffer_alpha));
   OCL_CHECK(err, err = top_kernel.setArg(narg++, buffer_attenuateOtherFreq));
   OCL_CHECK(err, err = top_kernel.setArg(narg++, buffer_filtIDX));

   OCL_CHECK(err, err = top_kernel.setArg(narg++, buffer_magnifiedLumaFFT_real));
   OCL_CHECK(err, err = top_kernel.setArg(narg++, buffer_magnifiedLumaFFT_imag));

   et.finish();


   // We then need to map our OpenCL buffers to get the pointers
   int *ptr_vidFFT_zero_real;
   int *ptr_vidFFT_zero_imag;
   int *ptr_croppedFilters;
   int *ptr_vidFFT_idx_real;
   int *ptr_vidFFT_idx_imag;

   bool *ptr_temporalFilter;
   floatT *ptr_cutoff_freq_low;
   floatT *ptr_cutoff_freq_high;
   bool *ptr_first;
   floatT *ptr_alpha;
   bool *ptr_attenuateOtherFreq;
   int *ptr_filtIDX;

   int *ptr_magnifiedLumaFFT_real;
   int *ptr_magnifiedLumaFFT_imag;

   OCL_CHECK(err,
		   ptr_vidFFT_zero_real = (int*)q.enqueueMapBuffer(buffer_vidFFT_real, CL_TRUE, CL_MAP_WRITE, 0, size_vidFFT, NULL, NULL, &err));
   OCL_CHECK(err,
   		   ptr_vidFFT_zero_imag = (int*)q.enqueueMapBuffer(buffer_vidFFT_imag, CL_TRUE, CL_MAP_WRITE, 0, size_vidFFT, NULL, NULL, &err));
   OCL_CHECK(err,
   		   ptr_croppedFilters = (int*)q.enqueueMapBuffer(buffer_cropped, CL_TRUE, CL_MAP_WRITE, 0, size_cropped, NULL, NULL, &err));
   OCL_CHECK(err,
   		   ptr_vidFFT_idx_real = (int*)q.enqueueMapBuffer(buffer_vidFFTidx_real, CL_TRUE, CL_MAP_WRITE, 0, size_vidfftidx, NULL, NULL, &err));
   OCL_CHECK(err,
   		   ptr_vidFFT_idx_imag = (int*)q.enqueueMapBuffer(buffer_vidFFTidx_imag, CL_TRUE, CL_MAP_WRITE, 0, size_vidfftidx, NULL, NULL, &err));
   OCL_CHECK(err,
   		   ptr_temporalFilter = (bool*)q.enqueueMapBuffer(buffer_temporalFilter, CL_TRUE, CL_MAP_WRITE, 0, temporal_filter_size, NULL, NULL, &err));
   OCL_CHECK(err,
   		   ptr_cutoff_freq_low = (floatT*)q.enqueueMapBuffer(buffer_cutoffFreqLow, CL_TRUE, CL_MAP_WRITE, 0, freq_low_size, NULL, NULL, &err));
   OCL_CHECK(err,
      		   ptr_cutoff_freq_high = (floatT*)q.enqueueMapBuffer(buffer_cutoffFreqHigh, CL_TRUE, CL_MAP_WRITE, 0, freq_high_size, NULL, NULL, &err));
   OCL_CHECK(err,
   		   ptr_first = (bool*)q.enqueueMapBuffer(buffer_first, CL_TRUE, CL_MAP_WRITE, 0, first_size, NULL, NULL, &err));
   OCL_CHECK(err,
   		   ptr_alpha = (floatT*)q.enqueueMapBuffer(buffer_alpha, CL_TRUE, CL_MAP_WRITE, 0, alpha_size, NULL, NULL, &err));
   OCL_CHECK(err,
		   ptr_attenuateOtherFreq = (bool  *)q.enqueueMapBuffer(buffer_attenuateOtherFreq, CL_TRUE, CL_MAP_WRITE, 0, attenuateOtherFreq_size, NULL, NULL, &err));
   OCL_CHECK(err,
		   ptr_filtIDX = (int*)q.enqueueMapBuffer(buffer_filtIDX, CL_TRUE, CL_MAP_WRITE, 0, filtidx_size, NULL, NULL, &err));

   OCL_CHECK(err,
   		   ptr_magnifiedLumaFFT_real = (int*)q.enqueueMapBuffer(buffer_magnifiedLumaFFT_real, CL_TRUE, CL_MAP_READ, 0, magnifiedLumaFFT_size, NULL, NULL, &err));
   OCL_CHECK(err,
		   ptr_magnifiedLumaFFT_imag = (int*)q.enqueueMapBuffer(buffer_magnifiedLumaFFT_imag, CL_TRUE, CL_MAP_READ, 0, magnifiedLumaFFT_size, NULL, NULL, &err));



   ptr_temporalFilter[0] = (temporalFilter == "differenceOfIIR") ? false : true;
   ptr_cutoff_freq_low[0] = cutoff_freq_low;
   ptr_cutoff_freq_high[0] = cutoff_freq_high;
   ptr_alpha[0] = alpha;
   ptr_attenuateOtherFreq[0] = attenuateOtherFreq;

   hls::x_complex<floatT> temp_real[VECTOR_SIZE]; // 16 elements
   hls::x_complex<floatT> temp_imag[VECTOR_SIZE]; // 16 elements

   hls::x_complex<floatT> temp_real2[VECTOR_SIZE]; // 16 elements
   hls::x_complex<floatT> temp_imag2[VECTOR_SIZE]; // 16 elements

   hls::x_complex<floatT> temp_real3[VECTOR_SIZE]; // 16 elements

   int temp_filtIDX[VECTOR_SIZE];

   hls::x_complex<floatT> real_part[VECTOR_SIZE]; // Array to hold 16 real parts
   hls::x_complex<floatT> imag_part[VECTOR_SIZE]; // Array to hold 16 imaginary parts

   /*-----------------------------------------------------------*/


   for (int level = 0; level < numLevels; level++) {
       stime_per_level = std::chrono::high_resolution_clock::now();

       std::vector<std::vector<std::complex<double>>> vidFFT_zero = vidFFT[0];
       std::vector<std::vector<double>> croppedFilters_level = croppedFilters[level];
       std::vector<std::vector<int>> filtidx = filtIDX[level];
       std::vector<std::vector<std::complex<double>>> magnifiedLumaFFT_temp(height, std::vector<std::complex<double>>(width, {0.0, 0.0}));

       for (int frameIDX = 0; frameIDX < numberOfFrames; frameIDX++) {
           stime_per_frame_fpga = std::chrono::high_resolution_clock::now();

           ptr_first[0] = (frameIDX == 0) ? true : false;

           std::vector<std::vector<std::complex<double>>> vidFFT_idx = vidFFT[frameIDX];

           et.add("Fill the buffers ");

		   int vec_count = 0;


           for (int i = 0; i < height; i++) {
               for (int j = 0; j < width; j++) {

            	   int index = vec_count / VECTOR_SIZE;
            	   int vec_idx = vec_count % VECTOR_SIZE;

            	   if(i < 2){
            		   temp_filtIDX[vec_idx]=filtidx[i][j];
            	   }

            	   temp_real[vec_idx].real((floatT)vidFFT_zero[i][j].real());
            	   temp_imag[vec_idx].imag((floatT)vidFFT_zero[i][j].imag());

            	   temp_real2[vec_idx].real((floatT)vidFFT_idx[i][j].real());
            	   temp_imag2[vec_idx].imag((floatT)vidFFT_idx[i][j].imag());

            	   temp_real3[vec_idx].real((floatT)croppedFilters_level[i][j]);

            	   vec_count++;

            	   if (vec_idx == VECTOR_SIZE - 1) {
            		   ptr_vidFFT_zero_real[index] = complex_to_uint320R_2(temp_real);
            		   ptr_vidFFT_zero_imag[index] = complex_to_uint320I_2(temp_imag);


            		   ptr_vidFFT_idx_real[index] = complex_to_uint320R_2(temp_real2);
            		   ptr_vidFFT_idx_imag[index] = complex_to_uint320I_2(temp_imag2);

            		   ptr_croppedFilters[index] = complex_to_uint320R_2(temp_real3);

            		   if(i < 2) {
            			   ptr_filtIDX[index] = int_to_uint320v2(temp_filtIDX);
            		   }

            		  // std::cout << "host index = " << index<< "\n";

            		   vec_idx = 0;
            	   }

            	   if (i == height - 1 && j == width - 1 && vec_idx != 0) {
            		   for (int k = vec_idx + 1; k < VECTOR_SIZE; k++) {
            			   temp_real[k].real(0.0);
            			   temp_imag[k].imag(0.0);

            			   temp_real2[k].real(0.0);
            			   temp_imag2[k].imag(0.0);

            			   temp_real3[k].real(0.0);

            		   }

            		   ptr_vidFFT_zero_real[index] = complex_to_uint320R_2(temp_real);
            		   ptr_vidFFT_zero_imag[index] = complex_to_uint320I_2(temp_imag);

            		   ptr_vidFFT_idx_real[index] = complex_to_uint320R_2(temp_real2);
					   ptr_vidFFT_idx_imag[index] = complex_to_uint320I_2(temp_imag2);

					   ptr_croppedFilters[index] = complex_to_uint320R_2(temp_real3);

            	   }
               }
           }

           et.finish();



           // Now all data is filled, move it to the FPGA memory
           et.add("Copy input data to device global memory ");
           OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_vidFFT_real}, 0));
           OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_vidFFT_imag}, 0));
           OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_cropped}, 0));
           OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_vidFFTidx_real}, 0));
           OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_vidFFTidx_imag}, 0));
           OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_temporalFilter}, 0));
           OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_cutoffFreqLow}, 0));
           OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_cutoffFreqHigh}, 0));
           OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_first}, 0));
           OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_alpha}, 0));
           OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_attenuateOtherFreq}, 0));
           OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_filtIDX}, 0));
           et.finish();

           // Launch the Kernel after all data is copied
           et.add("Launch the Kernel ");
           stime_hw = std::chrono::high_resolution_clock::now();

           OCL_CHECK(err, err = q.enqueueTask(top_kernel));

           etime_hw = std::chrono::high_resolution_clock::now();
           hw_duration = etime_hw - stime_hw;
           profilling_file << "Itteration " << counter << " HW Execution time: " << hw_duration.count() << " seconds." << std::endl;

           et.finish();

           // Retrieve the result of the kernel execution
           et.add("Copy Result from Device Global Memory to Host Local Memory ");
           OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_magnifiedLumaFFT_real}, CL_MIGRATE_MEM_OBJECT_HOST));
           OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_magnifiedLumaFFT_imag}, CL_MIGRATE_MEM_OBJECT_HOST));

           OCL_CHECK(err, q.finish());

           et.finish();

           // Write results back to the host's local memory
           et.add("Write the results back in Host ");

           vec_count = 0;
           for (int i = 0; i < height; i++) {
               for (int j = 0; j < width; j += VECTOR_SIZE) {
                   // Compute the index in the flattened 1D array
                   int index = vec_count / VECTOR_SIZE;

                   // Convert the packed myuint_t to arrays of real and imaginary parts
                   myuint_to_complex2(ptr_magnifiedLumaFFT_real[index], real_part, imag_part);
                   myuint_to_complex2(ptr_magnifiedLumaFFT_imag[index], real_part, imag_part);

                   // Assign to the 2D array
                   for (int k = 0; k < VECTOR_SIZE && (j + k) < width; k++) {
                	   double real_value = (double)real_part[k].real();
                	   double imag_value = (double)imag_part[k].imag();

                   	   // Assign clamped values to the 2D array
                	   magnifiedLumaFFT_temp[i][j + k] = std::complex<double>(real_value, imag_value);

                	   // Optional Debug output for tracing
                	  // std::cout << "i = " << i << " k = " << k << " j + k = " << (j + k) << "\n";
                	  // std::cout << "real_value = " << real_value
                	     //        << " imag_value = " << imag_value << "\n";


                   }

                   vec_count++;
               }
           }

           et.finish();
           magnifiedLumaFFT[frameIDX] = magnifiedLumaFFT_temp;



           //etime_per_frame_fpga = std::chrono::high_resolution_clock::now();
           //per_frame_duration_fpga = etime_per_frame_fpga - stime_per_frame_fpga;
           //profilling_file << "FPGA per Frame " << frameIDX << " Execution time: " << per_frame_duration_fpga.count() << " seconds." << std::endl;
       }

       etime_per_level = std::chrono::high_resolution_clock::now();
       per_level_duration = etime_per_level - stime_per_level;
       profilling_file << "Level " << level << " Execution time: " << per_level_duration.count() << " seconds." << std::endl;
   }


   	std::cout <<"----------------- Key execution times -----------------" << std::endl;
   	et.print();
   	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_vidFFT_real, ptr_vidFFT_zero_real));
   	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_vidFFT_imag, ptr_vidFFT_zero_imag));
   	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_cropped, ptr_croppedFilters));
   	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_vidFFTidx_real, ptr_vidFFT_idx_real));
   	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_vidFFTidx_imag, ptr_vidFFT_idx_imag));
   	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_temporalFilter, ptr_temporalFilter));
   	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_cutoffFreqLow, ptr_cutoff_freq_low));
   	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_cutoffFreqHigh, ptr_cutoff_freq_high));
   	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_first, ptr_first));
   	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_alpha, ptr_alpha));
   	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_attenuateOtherFreq, ptr_attenuateOtherFreq));
   	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_filtIDX, ptr_filtIDX));

   	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_magnifiedLumaFFT_real, ptr_magnifiedLumaFFT_real));
	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_magnifiedLumaFFT_imag, ptr_magnifiedLumaFFT_imag));

   	OCL_CHECK(err, err = q.finish());
   	delete [] buf;


    stime_sw = std::chrono::high_resolution_clock::now();

	//Add unmolested lowpass residual
	numLevels--;
	
	std::vector<std::vector<double>> croppedFiltersLastTemp = croppedFilters.back();
	std::vector<std::vector<double>> croppedFiltersLast(vidFFT[0].size(), std::vector<double>(vidFFT[0][0].size(),0.0));
	
	#pragma omp parallel for collapse(2)
	for(int i=0; i < croppedFiltersLast.size(); i++){
		for(int j=0; j<croppedFiltersLast[0].size(); j++){
			if(i < croppedFiltersLastTemp.size() && j < croppedFiltersLastTemp[i].size()){
			
				croppedFiltersLast[i][j] = croppedFiltersLastTemp[i][j];	
			}
		}
	}	

	
	std::vector<std::vector<double>> two(croppedFiltersLast.size(),std::vector<double>(croppedFiltersLast[0].size(),2.0));
	std::vector<std::vector<double>> croppedFiltersLastPowOfTwo = elementwisePower(croppedFiltersLast,two);
	std::vector<std::vector<std::complex<double>>> lowpassFrame (croppedFiltersLast.size(), std::vector<std::complex<double>>(croppedFiltersLast[0].size(),0.0));

	//#pragma omp parallel for
	for(int frameIDX = 0; frameIDX < numberOfFrames; frameIDX++){
		stime_lowpass = std::chrono::high_resolution_clock::now();
		updateLowpassFrame(lowpassFrame,vidFFT,filtIDX,croppedFiltersLastPowOfTwo,numLevels,frameIDX);
		updateMagnifiedLumaFFTLowpassFrame(magnifiedLumaFFT, filtIDX, lowpassFrame, numLevels, frameIDX);
		
		etime_lowpass = std::chrono::high_resolution_clock::now();
		lowpass_duration = etime_lowpass - stime_lowpass;
		profilling_file << "Frame " << frameIDX << " lowpass Execution time: " << lowpass_duration.count() << " seconds." << std::endl;
		
	}
	
	
	std::vector<std::vector<std::vector<uint8_t>>> res;
	std::vector<std::vector<double>> outFrame(height, std::vector<double>(width * 3));
	std::vector<std::vector<std::complex<double>>> magnifiedLumaFFT2(height,std::vector<std::complex<double>>(width,{0.0,0.0}));

	//#pragma omp parallel for
	for(int k = 0; k < numberOfFrames; k++){
		stime_finalResults = std::chrono::high_resolution_clock::now();



		fftshift(magnifiedLumaFFT[k]);
		std::vector<std::vector<std::complex<double>>> input = magnifiedLumaFFT[k];
		ifft2(input,magnifiedLumaFFT2);
		magnifiedLumaFFT[k] = magnifiedLumaFFT2;

		//#pragma omp parallel for collapse(2)
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				outFrame[i][j * 3] = std::real(magnifiedLumaFFT[k][i][j]); //channel 0
				outFrame[i][j * 3 + 1] = originalFrames[k][i][j * 3 + 1]; // channel 1
				outFrame[i][j * 3 + 2] = originalFrames[k][i][j * 3 + 2]; // channel 2



			}
		}

		outFrame = ntsc2rgb(outFrame); // per Frame

       	 // Put frame in output
	 	 //#pragma omp critical
       	 res.push_back(im2uint8(outFrame));
       	 
 		 etime_finalResults = std::chrono::high_resolution_clock::now();
 		 finalResults_duration = etime_finalResults - stime_finalResults;
 		 profilling_file << "Frame " << k << " FinalResults Execution time: " << finalResults_duration.count() << " seconds." << std::endl;
       	 
	}
	
	etime_sw = std::chrono::high_resolution_clock::now();
	sw_duration = etime_sw - stime_sw;
	std::cout << "SW_2: " << sw_duration.count() << " seconds." << std::endl;
	final_out = res;
}



