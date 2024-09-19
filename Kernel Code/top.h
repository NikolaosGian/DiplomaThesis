#ifndef _TOP_H_
#define _TOP_H_


#include <ap_fixed.h>
#include "ap_int.h"
#include "hls_x_complex.h"
#include "hls_stream.h"
#include "hls_math.h"

const int size2 = 100;

#define PI 3.14159265358979323846
#define floatT float

//#define BUFFER_SIZE 625
#define BUFFER_SIZE 32
#define DATAWIDTH 224
#define VECTOR_SIZE (DATAWIDTH / 32)
typedef ap_uint<DATAWIDTH> myuint_t;



myuint_t temp_real, temp_imag;


myuint_t tmpCropped_imag = 0;	// always imaginery part of croppedFilters -> 0


void elementWiseMultiplyBuildLevel(
		myuint_t &vidFFT_real,
		myuint_t &vidFFT_imag,
		myuint_t &croppedFilter,
		myuint_t &results_real,
		myuint_t &results_imag)
    ;

void elementWiseMultiply(
		myuint_t &in_real,
		myuint_t &in_imag,
		myuint_t &in2_real,
		myuint_t &in2_imag,
		myuint_t &results_real,
		myuint_t &results_imag)
    ;

myuint_t temp;

void circularShiftRow(myuint_t &row, int shift);


int shiftRows = VECTOR_SIZE / 2;
int shiftCols = VECTOR_SIZE / 2;

void fftshift(myuint_t &matrix_real,
			  myuint_t &matrix_imag) ;


void polar(myuint_t &magn, myuint_t &angle, myuint_t &realResult, myuint_t &imagResult);
// Perform 1D DFT
void dft1d(myuint_t &data_real, myuint_t &data_imag, bool inverse) ;

void dft2d(myuint_t &data_real, myuint_t &data_imag, bool inverse);

void fft2(myuint_t &in_real, myuint_t &in_imag, myuint_t &out_real, myuint_t &out_imag);
void ifft2(myuint_t &in_real, myuint_t &in_imag, myuint_t &out_real, myuint_t &out_imag);

void buildlevel(	myuint_t &vidFFT_real,
					myuint_t &vidFFT_imag,
					myuint_t &croppedFilter,
					myuint_t &results_real,
					myuint_t &results_imag);

// Function to convert ap_uint<32> to float without memcpy
float uint32_to_float(ap_uint<32> input) ;
// Function to convert float to ap_uint<32> without memcpy
ap_uint<32> float_to_uint32(float input) ;


void computeAbsoluteValue(myuint_t &input_real, myuint_t &input_imag, myuint_t &results_real, myuint_t &results_imag) ;

void elementWiseDivideComplex(
		myuint_t &results_real,
		myuint_t &results_imag,

		myuint_t &mat1_real,
		myuint_t &mat1_imag,

		myuint_t &mat2_real,
		myuint_t &mat2_imag) ;


void cordic_atan2(float y, float x, float &atan_out);

void calculatePhaseAngles(myuint_t &phaseAngles, myuint_t &complexMatrix_real, myuint_t &complexMatrix_imag );


ap_uint<32> customMod(ap_uint<32>  a, ap_uint<32>  m) ;

void computeDeltaForFrame(
		myuint_t &pyrRefAngle,
		myuint_t &pyrCurrent,
		myuint_t &delta) ;

void differenceOfIIR(myuint_t &delta, floatT rl, floatT rh , bool first) ;



void scalePhaseOfFrame(myuint_t &phaseOfFrame, floatT alpha) ;


void exp_custom( ap_uint<32> &z_real, ap_uint<32> &z_imag, ap_uint<32> &result_real, ap_uint<32> &result_imag);


void calculateExp1i( myuint_t &phaseOfFrame, myuint_t &result_real, myuint_t &result_imag ) ;


void reconLevel( myuint_t &im_dft_real, myuint_t  &im_dft_imag, myuint_t &croppedFilter,myuint_t &result_real, myuint_t &result_imag) ;



void updateMagnifiedLumaFFT(
    myuint_t &magnifiedLumaFFT_real,
    myuint_t &magnifiedLumaFFT_imag,
    myuint_t &filtIDX,
    myuint_t &curLevelFrame_real,
    myuint_t &curLevelFrame_imag
);






extern "C"{


void top(
        // In
        myuint_t *vidFFT_zero_real,
        myuint_t *vidFFT_zero_imag,
        myuint_t *croppedFilters, //non complex
        myuint_t *vidFFT_idx_real,
        myuint_t *vidFFT_idx_imag,
        bool *temporalFilter,
        floatT *cutoff_freq_low,
        floatT *cutoff_freq_high,
        bool *first,
        floatT *alpha,
        bool *attenuateOtherFreq,
        myuint_t *filtIDX,
        // Out
        myuint_t *magnifiedLumaFFT_real,
        myuint_t *magnifiedLumaFFT_imag
    ) ;



}





#endif
