#include "top.h"
/*
void printHexValue(const ap_uint<32>& value) {
    std::cout << "0x" << std::hex << value.to_uint() << std::dec << std::endl;
}

template<int W>
void print_ap_uint(ap_uint<W> value) {
    std::cout << "ap_uint<" << W << ">: ";
    for (int i = W - 1; i >= 0; --i) {
        std::cout << value[i];
    }
    std::cout << std::endl;
}

*/


// Function to convert ap_uint<32> to float without memcpy
float uint32_to_float(ap_uint<32> input) {
#pragma HLS inline

    union {
        uint32_t i;
        float f;
    } converter;
    converter.i = input.to_uint();
    return converter.f;
}

// Function to convert float to ap_uint<32> without memcpy
ap_uint<32> float_to_uint32(float input) {
#pragma HLS inline

    union {
        uint32_t i;
        float f;
    } converter;
    converter.f = input;
    return ap_uint<32>(converter.i);
}



void elementWiseMultiply(
		myuint_t &in_real,
		myuint_t &in_imag,
		myuint_t &in2_real,
		myuint_t &in2_imag,
		myuint_t &results_real,
		myuint_t &results_imag)
    {

//#pragma HLS inline
	for (int vector = 0; vector < VECTOR_SIZE; vector++) {
		#pragma HLS UNROLL factor=2

		ap_uint<32> tmpInR1 = in_real.range(32 * (vector + 1) - 1, vector * 32);
		ap_uint<32> tmpInI1 = in_imag.range(32 * (vector + 1) - 1, vector * 32);
		ap_uint<32> tmpInR2 = in2_real.range(32 * (vector + 1) - 1, vector * 32);
		ap_uint<32> tmpInI2 = in2_imag.range(32 * (vector + 1) - 1, vector * 32);

		ap_uint<64> tt1uint = (tmpInR1 * tmpInR2) - (tmpInI1 * tmpInI2);
		ap_uint<64> tt2uint = (tmpInR1*tmpInI2) + (tmpInI1 * tmpInR2);

		tt1uint = tt1uint >> 16;
		tt2uint = tt2uint >> 16;

		results_real.range(32 * (vector + 1) - 1, vector * 32) = tt1uint.range(31, 0);
		results_imag.range(32 * (vector + 1) - 1, vector * 32) = tt2uint.range(31, 0);

	}

}



void circularShiftRow(myuint_t &row, int shift) {
	temp = row;

    ap_uint<32> elements[VECTOR_SIZE];

    for (int vector = 0; vector < VECTOR_SIZE; vector++) {
#pragma HLS UNROLL factor=2
        elements[vector] = temp.range(32 * (vector + 1) - 1, 32 * vector);
    }

    ap_uint<32> shifted_elements[VECTOR_SIZE];
    for (int i = 0; i < VECTOR_SIZE; i++) {
#pragma HLS UNROLL factor=2
        int idx = (i + shift) % VECTOR_SIZE;
        if (idx < 0) idx += VECTOR_SIZE; // Handle negative indices
        shifted_elements[i] = elements[idx];
    }

   // myuint_t shifted_row = 0;
    for (int vector = 0; vector < VECTOR_SIZE; vector++) {
#pragma HLS UNROLL factor=2
    	temp.range(32 * (vector + 1) - 1, 32 * vector) = shifted_elements[vector];
    }

    // Update the original row with the shifted data
    row = temp;
}





void fftshift(myuint_t &matrix_real,
			  myuint_t &matrix_imag) {


	circularShiftRow(matrix_real,shiftCols);
	circularShiftRow(matrix_imag,shiftCols);

    ap_uint<32> temp[VECTOR_SIZE];
    ap_uint<32> temp2[VECTOR_SIZE];


    // Shift columns
    for (int j = 0; j < VECTOR_SIZE; j++) {
		#pragma HLS UNROLL factor=2
        for (int i = 0; i < VECTOR_SIZE; i++) {
			#pragma HLS UNROLL factor=2
            temp[i] = matrix_real.range(32 * (i + 1) - 1, 32 * i);
            temp2[i] = matrix_imag.range(32 * (i + 1) - 1, 32 * i);
        }


        // Perform circular shift on the temporary column array
        ap_uint<32> shifted_temp[VECTOR_SIZE];
        ap_uint<32> shifted_temp2[VECTOR_SIZE];

        for (int i = 0; i < VECTOR_SIZE; i++) {
			#pragma HLS UNROLL factor=2
            int idx = (i + shiftRows) % VECTOR_SIZE;
            if (idx < 0) idx += VECTOR_SIZE; // Handle negative indices
            shifted_temp[i] = temp[idx];
            shifted_temp2[i] = temp2[idx];

        }


        for (int i = 0; i < VECTOR_SIZE; i++) {
			#pragma HLS UNROLL factor=2
        	matrix_real.range(32 * (i + 1) - 1, 32 * i) = shifted_temp[i];
        	matrix_imag.range(32 * (i + 1) - 1, 32 * i) = shifted_temp2[i];
        }

    }
}



void polar(myuint_t &magn, myuint_t &angle, myuint_t &realResult, myuint_t &imagResult) {

#pragma HLS inline

    for (int vector = 0; vector < VECTOR_SIZE; vector++) {
#pragma HLS UNROLL factor=2
        ap_uint<32> tmpMagn = magn.range(32 * (vector + 1) - 1, vector * 32);
        ap_uint<32> tmAngle = angle.range(32 * (vector + 1) - 1, vector * 32);

        float costmp = hls::cos(uint32_to_float(tmAngle));
        float sintmp = hls::sin(uint32_to_float(tmAngle));
        ap_uint<32> cosuint = float_to_uint32(costmp);
        ap_uint<32> sinuint = float_to_uint32(sintmp);

        ap_uint<64> tt1 = (tmpMagn * cosuint);
        ap_uint<64> tt2 = (tmpMagn * sinuint);

        tt1 = tt1 >> 16;
        tt2 = tt2 >> 16;
        realResult.range(32 * (vector + 1) - 1, vector * 32) = tt1.range(31, 0);
        imagResult.range(32 * (vector + 1) - 1, vector * 32) = tt2.range(31, 0);
    }
}

void dft1d(myuint_t &data_real, myuint_t &data_imag, bool inverse) {

    myuint_t one = 1;
    myuint_t polarRealResult;
    myuint_t polarImagResult;
    myuint_t sign_mult_angle;

    for (int k = 0; k < VECTOR_SIZE; k++) {
		#pragma HLS UNROLL factor=2
        for (int n = 0; n < VECTOR_SIZE; n++) {
		#pragma HLS UNROLL factor=2


            float ff = (inverse ? 1.0 : -1.0) * ((2.0 * PI * k * n) / VECTOR_SIZE);

            for (int l = 0; l < VECTOR_SIZE; l++){
				#pragma HLS UNROLL factor=2
            	sign_mult_angle.range(32 * (l + 1) - 1, l * 32) = float_to_uint32(ff);
            	one.range(32 * (l + 1) - 1, l * 32) = float_to_uint32(1.0);
            }


            polar(one, sign_mult_angle, polarRealResult, polarImagResult);

            ap_uint<32> temp1 = data_real.range(32 * (k + 1) - 1, k * 32);
            ap_uint<32> temp2 = data_imag.range(32 * (k + 1) - 1, k * 32);

            ap_uint<32> temp3 = data_real.range(32 * (n + 1) - 1, n * 32);
            ap_uint<32> temp4 = data_imag.range(32 * (n + 1) - 1, n * 32);

            ap_uint<64> tt1 = temp1 + (temp3 * polarRealResult);
            ap_uint<64> tt2 = temp2 + (temp4 * polarImagResult);

            tt1 = tt1 >> 16;
            tt2 = tt2 >> 16;

            data_real.range(32 * (k + 1) - 1, k * 32) = tt1.range(31, 0);
            data_imag.range(32 * (k + 1) - 1, k * 32) = tt2.range(31, 0);
        }

        if (inverse) {
            ap_uint<32> temp1 = data_real.range(32 * (k + 1) - 1, k * 32);
            ap_uint<32> temp2 = data_imag.range(32 * (k + 1) - 1, k * 32);
            ap_uint<32> vecSize = VECTOR_SIZE;

            ap_uint<64> uintfinal1 = temp1 / vecSize;
            ap_uint<64> uintfinal2 = temp2 / vecSize;
            uintfinal1 = uintfinal1 >> 16;
            uintfinal2 = uintfinal2 >> 16;

            //print_ap_uint(uintfinal1);
            //print_ap_uint(uintfinal2);

            data_real.range(32 * (k + 1) - 1, k * 32) = uintfinal1.range(31, 0);
            data_imag.range(32 * (k + 1) - 1, k * 32) = uintfinal2.range(31, 0);
        }
    }
}

void dft2d(myuint_t &data_real, myuint_t &data_imag, bool inverse) {

    // DFT of rows
    for (int i = 0; i < VECTOR_SIZE; i++) {
		#pragma HLS UNROLL factor=2
        myuint_t row_real = data_real.range(32 * (i + 1) - 1, i * 32);
        myuint_t row_imag = data_imag.range(32 * (i + 1) - 1, i * 32);
        dft1d(row_real, row_imag, inverse);
        data_real.range(32 * (i + 1) - 1, i * 32) = row_real;
        data_imag.range(32 * (i + 1) - 1, i * 32) = row_imag;
    }

    // DFT of columns
    for (int j = 0; j < VECTOR_SIZE; j++) {
		#pragma HLS UNROLL factor=2
        myuint_t column_real = 0;
        myuint_t column_imag = 0;
        for (int i = 0; i < VECTOR_SIZE; i++) {
			#pragma HLS UNROLL factor=2
            column_real.range(32 * (i + 1) - 1, i * 32) = data_real.range(32 * (i * VECTOR_SIZE + j + 1) - 1, (i * VECTOR_SIZE + j) * 32);
            column_imag.range(32 * (i + 1) - 1, i * 32) = data_imag.range(32 * (i * VECTOR_SIZE + j + 1) - 1, (i * VECTOR_SIZE + j) * 32);
        }
        dft1d(column_real, column_imag, inverse);
        for (int i = 0; i < VECTOR_SIZE; i++) {
			#pragma HLS UNROLL factor=2
            data_real.range(32 * (i * VECTOR_SIZE + j + 1) - 1, (i * VECTOR_SIZE + j) * 32) = column_real.range(32 * (i + 1) - 1, i * 32);
            data_imag.range(32 * (i * VECTOR_SIZE + j + 1) - 1, (i * VECTOR_SIZE + j) * 32) = column_imag.range(32 * (i + 1) - 1, i * 32);
        }
    }
}

void fft2(myuint_t &in_real, myuint_t &in_imag, myuint_t &out_real, myuint_t &out_imag){
	//#pragma HLS inline


	//bool reserver = false;
	dft1d(in_real, in_imag, false);
	out_real = in_real;
	out_imag = in_imag;

}

void ifft2(myuint_t &in_real, myuint_t &in_imag, myuint_t &out_real, myuint_t &out_imag){
	//#pragma HLS inline


	//bool reserver = true;
	dft1d(in_real, in_imag, true);
	out_real = in_real;
	out_imag = in_imag;

}




void buildlevel(	myuint_t &vidFFT_real,
					myuint_t &vidFFT_imag,
					myuint_t &croppedFilter,
					myuint_t &results_real,
					myuint_t &results_imag){

	myuint_t temp_real2, temp_imag2, temp_croppedImag;
	temp_croppedImag = 0;


	elementWiseMultiply(vidFFT_real, vidFFT_imag, croppedFilter,temp_croppedImag, temp_real2,temp_imag2);
	fftshift(temp_real2,temp_imag2);
	ifft2(temp_real2,temp_imag2,results_real,results_imag);

}





void computeAbsoluteValue(myuint_t &input_real, myuint_t &input_imag, myuint_t &results_real, myuint_t &results_imag) {
//#pragma HLS inline


    for (int vector = 0; vector < VECTOR_SIZE; vector++) {
#pragma HLS UNROLL factor=2

        ap_uint<32> tmpInput_real1 = input_real.range(32 * (vector + 1) - 1, vector * 32);
        ap_uint<32> tmpInput_imag1 = input_imag.range(32 * (vector + 1) - 1, vector * 32);

        ap_uint<32> l = hls::hypot(tmpInput_real1, tmpInput_imag1);

        results_real.range(32 * (vector + 1) - 1, vector * 32) = l;
        results_imag.range(32 * (vector + 1) - 1, vector * 32) = 0;
    }
}

void elementWiseDivideComplex(
		myuint_t &results_real,
		myuint_t &results_imag,

		myuint_t &mat1_real,
		myuint_t &mat1_imag,

		myuint_t &mat2_real,
		myuint_t &mat2_imag) {



	for (int vector = 0; vector < VECTOR_SIZE; vector++) {
		#pragma HLS UNROLL factor=2

		ap_uint<32> tmpInput_real1 = mat1_real.range(32 * (vector + 1) - 1, vector * 32);
		ap_uint<32> tmpInput_imag1 = mat1_imag.range(32 * (vector + 1) - 1, vector * 32);

		ap_uint<32> tmpInput_real2 = mat2_real.range(32 * (vector + 1) - 1, vector * 32);
		ap_uint<32> tmpInput_imag2 = mat2_imag.range(32 * (vector + 1) - 1, vector * 32);


		ap_uint<32> t1 = (tmpInput_real1*tmpInput_real2) + (tmpInput_imag1*tmpInput_imag2); //(a*c +b*d)
		ap_uint<32> t2 = (tmpInput_imag1*tmpInput_real2) - (tmpInput_real1*tmpInput_imag2); // b*c -a*d)
		ap_uint<32> t3 = (tmpInput_real2*tmpInput_real2) + (tmpInput_imag2*tmpInput_imag2); // c^2 + d^2
		ap_uint<32> realuint = float_to_uint32(uint32_to_float(t1)/uint32_to_float(t3));
		ap_uint<32> imaguint = float_to_uint32(uint32_to_float(t2)/uint32_to_float(t3));

		results_real.range(32 * (vector + 1) - 1, vector * 32) = realuint;
		results_imag.range(32 * (vector + 1) - 1, vector * 32) = imaguint;

	}

}

void calculatePhaseAngles(myuint_t &phaseAngles, myuint_t &complexMatrix_real, myuint_t &complexMatrix_imag ) {
//#pragma HLS inline

	for (int vector = 0; vector < VECTOR_SIZE; vector++) {

#pragma HLS UNROLL factor=2
		ap_uint<32> tmpInput_real1 = complexMatrix_real.range(32 * (vector + 1) - 1, vector * 32);
		ap_uint<32> tmpInput_imag1 = complexMatrix_imag.range(32 * (vector + 1) - 1, vector * 32);

		tmpInput_real1 = tmpInput_real1 << 16;
		tmpInput_imag1 = tmpInput_imag1 << 16;

		float real_part = uint32_to_float(tmpInput_real1);
		float imag_part = uint32_to_float(tmpInput_imag1);
		float angle = hls::atan2(imag_part, real_part);
		ap_uint<32> final = float_to_uint32(angle);

		phaseAngles.range(32 * (vector + 1) - 1, vector * 32) = final;

	}

}



ap_uint<32> customMod(ap_uint<32>  a, ap_uint<32>  m) {
	#pragma HLS inline

    if (m == 0) {
        return a;
    }

    ap_uint<32>  temp = float_to_uint32(uint32_to_float(a) / uint32_to_float(m));
    ap_uint<32>  temp1 = hls::floor(temp);
    ap_uint<32>  temp2 = m * temp1;
    ap_uint<32>  result = a - temp2;
    ap_uint<32> final = result;

    return final;
}

void computeDeltaForFrame(
		myuint_t &pyrRefAngle,
		myuint_t &pyrCurrent,
		myuint_t &delta) {

	for (int vector = 0; vector < VECTOR_SIZE; vector++) {
#pragma HLS UNROLL factor=2

		ap_uint<32> tmpInput1 = pyrRefAngle.range(32 * (vector + 1) - 1, vector * 32);
		ap_uint<32> tmpInput2 = pyrCurrent.range(32 * (vector + 1) - 1, vector * 32);

		ap_uint<32> pi = float_to_uint32(3.1416);
		ap_uint<32> in1 = pi + tmpInput2 - tmpInput1;
		ap_uint<32> in2 = pi*2;
		ap_uint<32> finaluint = customMod(in1.range(31,0),in2.range(31,0)) - pi;
		delta.range(32 * (vector + 1) - 1, vector * 32) = finaluint;

	}

}


void differenceOfIIR(myuint_t &delta, floatT rl, floatT rh , bool first) {

	myuint_t lowpass1 = 0;
	myuint_t lowpass2 = 0;

    if (first){

    	for (int vector = 0; vector < VECTOR_SIZE; vector++) {
#pragma HLS UNROLL factor=2
    		ap_uint<32> tmpInput1 = delta.range(32 * (vector + 1) - 1, vector * 32);
    		lowpass1.range(32 * (vector + 1) - 1, vector * 32) = tmpInput1;
    		lowpass2.range(32 * (vector + 1) - 1, vector * 32) = tmpInput1;
    		delta.range(32 * (vector + 1) - 1, vector * 32) = 0;
    		}

    }

    for (int vector = 0; vector < VECTOR_SIZE; vector++) {
#pragma HLS UNROLL factor=2

    	ap_uint<32> tmplowpass1 = lowpass1.range(32 * (vector + 1) - 1, vector * 32);
    	ap_uint<32> tmplowpass2 = lowpass1.range(32 * (vector + 1) - 1, vector * 32);
    	ap_uint<32> tmpdelta = delta.range(32 * (vector + 1) - 1, vector * 32);

    	ap_uint<32> rhUint = float_to_uint32(rh);
    	ap_uint<32> rlUint = float_to_uint32(rl);
    	ap_uint<32> one = 1;
    	ap_uint<64> tt1 = (one - rhUint) * tmplowpass1 + rhUint * tmpdelta;
    	ap_uint<64> tt2 = (one - rlUint) * tmplowpass2 + rlUint * tmpdelta;
    	ap_uint<64> finalDelta = tt1 - tt2;

    	tt1 = tt1 >> 16;
    	tt2 = tt2 >> 16;
    	finalDelta = finalDelta >> 16;

    	lowpass1.range(32 * (vector + 1) - 1, vector * 32) = tt1.range(31,0);
    	lowpass2.range(32 * (vector + 1) - 1, vector * 32) = tt2.range(31,0);

    	delta.range(32 * (vector + 1) - 1, vector * 32) = finalDelta.range(31,0);

    	}


}



void scalePhaseOfFrame(myuint_t &phaseOfFrame, floatT alpha) {

	for (int vector = 0; vector < VECTOR_SIZE; vector++) {
#pragma HLS UNROLL factor=2

		ap_uint<32> tmpAlpha = float_to_uint32(alpha);
		ap_uint<32> tmpPhase = phaseOfFrame.range(32 * (vector + 1) - 1, vector * 32);
		ap_uint<64> tmpResult = tmpPhase * tmpAlpha;

		tmpResult = tmpResult >> 16;

		phaseOfFrame.range(32 * (vector + 1) - 1, vector * 32) = tmpResult.range(31,0);

		}
}


void exp_custom( ap_uint<32> &z_real, ap_uint<32> &z_imag, ap_uint<32> &result_real, ap_uint<32> &result_imag) {

	ap_uint<64> exp_x = float_to_uint32(hls::exp(uint32_to_float(z_real)));
	ap_uint<64> cos_y = float_to_uint32(hls::cos(uint32_to_float(z_imag)));
	ap_uint<64> sin_y = float_to_uint32(hls::sin(uint32_to_float(z_imag)));
	ap_uint<64> tresult_real = exp_x * cos_y;
	ap_uint<64> tresult_imag = exp_x * sin_y;

	tresult_real = tresult_real >> 16;
	tresult_imag = tresult_imag >> 16;

	result_real = tresult_real.range(31,0);
	result_imag = tresult_imag.range(31,0);

}

void calculateExp1i( myuint_t &phaseOfFrame, myuint_t &result_real, myuint_t &result_imag ) {

	for (int vector = 0; vector < VECTOR_SIZE; vector++) {
#pragma HLS UNROLL factor=2
		ap_uint<32> tmpPhase = phaseOfFrame.range(32 * (vector + 1) - 1, vector * 32);

		result_real.range(32 * (vector + 1) - 1, vector * 32) = 0;
		result_imag.range(32 * (vector + 1) - 1, vector * 32) = tmpPhase;

		ap_uint<32> tmpResultR = result_real.range(32 * (vector + 1) - 1, vector * 32);
		ap_uint<32> tmpResultI = result_imag.range(32 * (vector + 1) - 1, vector * 32);
		ap_uint<32> tmpFinalResultR;
		ap_uint<32> tmpFinalResultI;

		exp_custom(tmpResultR, tmpResultI, tmpFinalResultR, tmpFinalResultI);

		result_real.range(32 * (vector + 1) - 1, vector * 32) = tmpFinalResultR;
		result_imag.range(32 * (vector + 1) - 1, vector * 32) = tmpFinalResultI;

		}

}


void reconLevel( myuint_t &im_dft_real, myuint_t  &im_dft_imag, myuint_t &croppedFilter,myuint_t &result_real, myuint_t &result_imag) {

	ap_uint<32> two = 2;
	myuint_t temp_real1;
	myuint_t temp_imag1;
	myuint_t temp_croppedImag = 0;
	myuint_t temp_resultReal;
	myuint_t temp_resultImag;

	fft2(im_dft_real, im_dft_imag, temp_real1, temp_imag1);
	fftshift(temp_real1, temp_imag1);
	elementWiseMultiply(croppedFilter,temp_croppedImag, temp_real1, temp_imag1, temp_resultReal, temp_resultImag);


	for (int vector = 0; vector < VECTOR_SIZE; vector++) {
#pragma HLS UNROLL factor=2

		ap_uint<32> tmpReal = temp_resultReal.range(32 * (vector + 1) - 1, vector * 32);
		ap_uint<32> tmpImag = temp_resultImag.range(32 * (vector + 1) - 1, vector * 32);

		ap_uint<32> tt1 = (tmpReal * two);
		ap_uint<32> tt2 = (tmpImag* two);

		result_real.range(32 * (vector + 1) - 1, vector * 32) = tt1;
		result_imag.range(32 * (vector + 1) - 1, vector * 32) = tt2;

		}

}


const int filtIDX_size = (2 * size2 - 1) / (VECTOR_SIZE);

void updateMagnifiedLumaFFT(
    myuint_t &magnifiedLumaFFT_real,
    myuint_t &magnifiedLumaFFT_imag,
    myuint_t &filtIDX,
    myuint_t &curLevelFrame_real,
    myuint_t &curLevelFrame_imag
) {

    for (int vector = 0; vector < VECTOR_SIZE; vector++) {
#pragma HLS UNROLL factor=2


    	int idx = filtIDX.range(32 * (vector + 1) - 1, vector * 32).to_int() % VECTOR_SIZE;

    	if(idx >= 0){
    		ap_uint<32> magReal = magnifiedLumaFFT_real.range(32 * (idx + 1) - 1, 32 * idx);
    		ap_uint<32> magImag = magnifiedLumaFFT_imag.range(32 * (idx + 1) - 1, 32 * idx);
    		ap_uint<32> curReal = curLevelFrame_real.range(32 * (idx + 1) - 1, 32 * idx);
    		ap_uint<32> curImag = curLevelFrame_imag.range(32 * (idx + 1) - 1, 32 * idx);

    		ap_uint<32> updatedReal = magReal + curReal;
    		ap_uint<32> updatedImag = magImag + curImag;

    		//updatedReal = updatedReal >> 16;
    		//updatedImag = updatedImag >> 16;

    		magnifiedLumaFFT_real.range(32 * (idx + 1) - 1, 32 * idx) = updatedReal.range(31,0);
    		magnifiedLumaFFT_imag.range(32 * (idx + 1) - 1, 32 * idx) = updatedImag.range(31,0);
    	}

    }
}


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
    ) {
	/*
	#pragma HLS INTERFACE m_axi port = vidFFT_zero_real offset = slave bundle = gmem0
	#pragma HLS INTERFACE m_axi port = vidFFT_zero_imag offset = slave bundle = gmem1
	#pragma HLS INTERFACE m_axi port = croppedFilters offset = slave bundle = gmem2
	#pragma HLS INTERFACE m_axi port = vidFFT_idx_real offset = slave bundle = gmem3
	#pragma HLS INTERFACE m_axi port = vidFFT_idx_imag offset = slave bundle = gmem4
	#pragma HLS INTERFACE m_axi port = temporalFilter offset = slave bundle = gmem5
	#pragma HLS INTERFACE m_axi port = cutoff_freq_low offset = slave bundle = gmem6
	#pragma HLS INTERFACE m_axi port = cutoff_freq_high offset = slave bundle = gmem7
	#pragma HLS INTERFACE m_axi port = first offset = slave bundle = gmem8
	#pragma HLS INTERFACE m_axi port = alpha offset = slave bundle = gmem9
	#pragma HLS INTERFACE m_axi port = attenuateOtherFreq offset = slave bundle = gmem10
	#pragma HLS INTERFACE m_axi port = filtIDX offset = slave bundle = gmem11
	#pragma HLS INTERFACE m_axi port = magnifiedLumaFFT_real offset = slave bundle = gmem12
	#pragma HLS INTERFACE m_axi port = magnifiedLumaFFT_imag offset = slave bundle = gmem13
	*/
	#pragma HLS INTERFACE m_axi port = vidFFT_zero_real bundle = gmem0
	#pragma HLS INTERFACE m_axi port = vidFFT_zero_imag bundle = gmem1
	#pragma HLS INTERFACE m_axi port = croppedFilters bundle = gmem2
	#pragma HLS INTERFACE m_axi port = vidFFT_idx_real bundle = gmem3
	#pragma HLS INTERFACE m_axi port = vidFFT_idx_imag bundle = gmem4
	#pragma HLS INTERFACE m_axi port = temporalFilter bundle = gmem5
	#pragma HLS INTERFACE m_axi port = cutoff_freq_low bundle = gmem6
	#pragma HLS INTERFACE m_axi port = cutoff_freq_high bundle = gmem7
	#pragma HLS INTERFACE m_axi port = first  bundle = gmem8
	#pragma HLS INTERFACE m_axi port = alpha bundle = gmem9
	#pragma HLS INTERFACE m_axi port = attenuateOtherFreq bundle = gmem10
	#pragma HLS INTERFACE m_axi port = filtIDX bundle = gmem11
	#pragma HLS INTERFACE m_axi port = magnifiedLumaFFT_real bundle = gmem12
	#pragma HLS INTERFACE m_axi port = magnifiedLumaFFT_imag bundle = gmem13

	#pragma HLS INTERFACE s_axilite port = vidFFT_zero_real bundle = control
	#pragma HLS INTERFACE s_axilite port = vidFFT_zero_imag bundle = control
	#pragma HLS INTERFACE s_axilite port = croppedFilters bundle = control
	#pragma HLS INTERFACE s_axilite port = vidFFT_idx_real bundle = control
	#pragma HLS INTERFACE s_axilite port = vidFFT_idx_imag bundle = control
	#pragma HLS INTERFACE s_axilite port = temporalFilter bundle = control
	#pragma HLS INTERFACE s_axilite port = cutoff_freq_low bundle = control
	#pragma HLS INTERFACE s_axilite port = cutoff_freq_high bundle = control
	#pragma HLS INTERFACE s_axilite port = first bundle = control
	#pragma HLS INTERFACE s_axilite port = alpha bundle = control
	#pragma HLS INTERFACE s_axilite port = attenuateOtherFreq bundle = control
	#pragma HLS INTERFACE s_axilite port = filtIDX bundle = control
	#pragma HLS INTERFACE s_axilite port = magnifiedLumaFFT_real bundle = control
	#pragma HLS INTERFACE s_axilite port = magnifiedLumaFFT_imag bundle = control
	#pragma HLS INTERFACE s_axilite port = return bundle = control


	floatT local_freq = 500.0;
	floatT local_cutoff_freq_low = cutoff_freq_low[0];
	floatT local_cutoff_freq_high =cutoff_freq_high[0];

	float low = local_cutoff_freq_low / local_freq;
	float high = local_cutoff_freq_high / local_freq;

	bool local_temporalFilter = temporalFilter[0];
	bool local_first = first[0];

	floatT local_alpha = alpha[0];

    myuint_t magnifiedLumaFFT_real_local[BUFFER_SIZE];
    myuint_t magnifiedLumaFFT_imag_local[BUFFER_SIZE];
    myuint_t vidFFT_zero_real_local[BUFFER_SIZE];
    myuint_t vidFFT_zero_imag_local[BUFFER_SIZE];
    myuint_t vidFFT_idx_real_local[BUFFER_SIZE];
    myuint_t vidFFT_idx_imag_local[BUFFER_SIZE];
    myuint_t croppedFilters_local[BUFFER_SIZE];
    myuint_t filtIDX_local[BUFFER_SIZE];

    /*
   	myuint_t tmpVidFFT_zeroR;
   	myuint_t tmpVidFFT_zeroI;
   	myuint_t tmpVidFFT_idxR;
   	myuint_t tmpVidFFT_idxI;
   	myuint_t tmpCroppedFilter;
   	myuint_t tmpFiltIDX_selected;
   	myuint_t tmpPyrRef_real;
   	myuint_t tmpPyrRef_imag;
   	myuint_t tmpAbsPyrRef_real;
   	myuint_t tmpAbsPyrRef_imag;
   	myuint_t tmpPyrRefPhaseOrig_real;
   	myuint_t tmpPyrRefPhaseOrig_imag;
   	myuint_t tmpPyrRefAngle;
   	myuint_t tmpFilterResponse_real;
   	myuint_t tmpFilterResponse_imag;
   	myuint_t tmpPyrCurrent;
   	myuint_t tmpDelta;
   	myuint_t tmpOriginalLevel_real;
   	myuint_t tmpOriginalLevel_imag;
   	myuint_t tmpExpPhaseOfFrame_real;
   	myuint_t tmpExpPhaseOfFrame_imag;
   	myuint_t tmpTransformOut_real;
   	myuint_t tmpTransformOut_imag;
   	myuint_t tmpCurLevelFrame_real;
   	myuint_t tmpCurLevelFrame_imag;
   	myuint_t tmpMagnifiedLumaFFT_real;
   	myuint_t tmpMagnifiedLumaFFT_imag;
	*/
    int size_in100 = (size2 * size2 - 1) / VECTOR_SIZE + 1;
    int size_in100v2 = (2 * size2 - 1) / VECTOR_SIZE + 1;
    /*
    outer_loop_filtIDX:for (int i = 0; i < size_in100v2; i += BUFFER_SIZE) {
		#pragma HLS DATAFLOW
		#pragma HLS stream variable = filtIDX_local depth = 224 type=FIFO

    	int chunk_size2 = BUFFER_SIZE;

    	if ((i + BUFFER_SIZE) > size_in100v2)
    		chunk_size2 = size_in100v2 - i;

        filtIDX_READ:for (int j = 0; j < chunk_size2; j++) {
    			#pragma hls pipeline
				//#pragma HLS LOOP_TRIPCOUNT min = 512 max = 512

            	filtIDX_local[j] = filtIDX[i + j];
            }
    }
*/
    outer_loop_processing:for (int i = 0; i < size_in100; i += BUFFER_SIZE) {
    	/*
		#pragma HLS DATAFLOW
        #pragma HLS stream variable = vidFFT_zero_real_local depth = 32 type=FIFO
        #pragma HLS stream variable = vidFFT_zero_imag_local depth = 32 type=FIFO
        #pragma HLS stream variable = vidFFT_idx_real_local depth = 32 type=FIFO
        #pragma HLS stream variable = vidFFT_idx_imag_local depth = 32 type=FIFO
        #pragma HLS stream variable = croppedFilters_local depth = 32 type=FIFO
       	#pragma HLS stream variable = magnifiedLumaFFT_real_local depth = 32 type=FIFO
        #pragma HLS stream variable = magnifiedLumaFFT_imag_local depth = 32 type=FIFO
        #pragma HLS stream variable = filtIDX_local depth = 32 type=FIFO
        */

    	int chunk_size = BUFFER_SIZE;

        if ((i + BUFFER_SIZE) > size_in100)
            chunk_size = size_in100 - i;

        int counter =0;
        int effective_j = 0;

       fill_locals:for (int j = 0; j < chunk_size; j++) {
			#pragma hls pipeline
			#pragma HLS LOOP_TRIPCOUNT min=1 max=32

        	vidFFT_zero_real_local[j] = vidFFT_zero_real[i+j];
        	vidFFT_zero_imag_local[j] = vidFFT_zero_imag[i+j];
        	vidFFT_idx_real_local[j] = vidFFT_idx_real[i+j];
        	vidFFT_idx_imag_local[j] = vidFFT_idx_imag[i+j];
        	croppedFilters_local[j] = croppedFilters[i+j];

        	 if (j < size_in100v2) {
        		 filtIDX_local[j] = filtIDX[i + j];
        	 }
       }

       processing:for (int j = 0; j < chunk_size; j++) {
			//#pragma hls pipeline
			#pragma HLS LOOP_TRIPCOUNT min=1 max=32

    	   	myuint_t tmpVidFFT_zeroR ;
    	    myuint_t tmpVidFFT_zeroI ;
    	    myuint_t tmpVidFFT_idxR;
    	    myuint_t tmpVidFFT_idxI;
    	    myuint_t tmpCroppedFilter;
    	    myuint_t tmpFiltIDX_selected;
    	    myuint_t tmpPyrRef_real;
    	    myuint_t tmpPyrRef_imag;
    	    myuint_t tmpAbsPyrRef_real;
    	    myuint_t tmpAbsPyrRef_imag;
    	    myuint_t tmpPyrRefPhaseOrig_real;
    	    myuint_t tmpPyrRefPhaseOrig_imag;
    	    myuint_t tmpPyrRefAngle;
    	    myuint_t tmpFilterResponse_real;
    	    myuint_t tmpFilterResponse_imag;
    	    myuint_t tmpPyrCurrent;
    	    myuint_t tmpDelta;
    	    myuint_t tmpOriginalLevel_real;
    	    myuint_t tmpOriginalLevel_imag;
    	    myuint_t tmpExpPhaseOfFrame_real;
    	    myuint_t tmpExpPhaseOfFrame_imag;
    	    myuint_t tmpTransformOut_real;
    	    myuint_t tmpTransformOut_imag;
    	    myuint_t tmpCurLevelFrame_real;
    	    myuint_t tmpCurLevelFrame_imag;
    	    myuint_t tmpMagnifiedLumaFFT_real;
    	    myuint_t tmpMagnifiedLumaFFT_imag;



    	    /*
        	tmpVidFFT_zeroR = vidFFT_zero_real[i + j];
        	tmpVidFFT_zeroI = vidFFT_zero_imag[i + j];
        	tmpVidFFT_idxR = vidFFT_idx_real[i + j];
        	tmpVidFFT_idxI = vidFFT_idx_imag[i + j];
        	tmpCroppedFilter =  croppedFilters[i + j];
        	*/
          	tmpVidFFT_zeroR = vidFFT_zero_real_local[j];
          	tmpVidFFT_zeroI = vidFFT_zero_imag_local[j];
          	tmpVidFFT_idxR = vidFFT_idx_real_local[j];
          	tmpVidFFT_idxI = vidFFT_idx_imag_local[j];
          	tmpCroppedFilter =  croppedFilters_local[j];

        	if(counter == 13){	// after 13*7 elements get the next filtIDX
        		effective_j++;
        		counter = 0;
        	}else{
        		counter++;		// count up

        		if(effective_j == 13){ // reset to 0 again. MAX elements per position of buffers.
        			effective_j = 0;   // 14 * 7 elements = 98 so 7*7 -> 49 half of them are X other are Y.
        		}
        	}

        	if(effective_j < 7){	// 0-6 -> X , 7-13 -> Y. Doing mod because we have real range 0-6 for each fildIDX_local (can't be out of bounds!).
        		tmpFiltIDX_selected=filtIDX_local[(effective_j%7)];
        	}else{
        		tmpFiltIDX_selected=filtIDX_local[(effective_j%7)];
        	}



        	buildlevel(tmpVidFFT_zeroR, tmpVidFFT_zeroI, tmpCroppedFilter, tmpPyrRef_real, tmpPyrRef_imag);
        	computeAbsoluteValue(tmpPyrRef_real, tmpPyrRef_imag, tmpAbsPyrRef_real, tmpAbsPyrRef_imag);
        	elementWiseDivideComplex(tmpPyrRefPhaseOrig_real, tmpPyrRefPhaseOrig_imag, tmpPyrRef_real, tmpPyrRef_imag, tmpAbsPyrRef_real, tmpAbsPyrRef_imag);
        	calculatePhaseAngles(tmpPyrRefAngle, tmpPyrRef_real, tmpPyrRef_imag);
        	//
        	buildlevel(tmpVidFFT_idxR, tmpVidFFT_idxI, tmpCroppedFilter, tmpFilterResponse_real, tmpFilterResponse_imag);
        	calculatePhaseAngles(tmpPyrCurrent, tmpFilterResponse_real, tmpFilterResponse_imag);
        	computeDeltaForFrame(tmpPyrRefAngle, tmpPyrCurrent, tmpDelta);


        	//if (local_temporalFilter == false) { //false -> differenceOfIIR

        	differenceOfIIR(tmpDelta, low, high, local_first);

        	//}

        	buildlevel(tmpVidFFT_idxR, tmpVidFFT_idxI, tmpCroppedFilter, tmpOriginalLevel_real, tmpOriginalLevel_imag);
        	scalePhaseOfFrame(tmpDelta, local_alpha);
        	calculateExp1i(tmpDelta, tmpExpPhaseOfFrame_real, tmpExpPhaseOfFrame_imag);
        	elementWiseMultiply(tmpExpPhaseOfFrame_real, tmpExpPhaseOfFrame_imag, tmpOriginalLevel_real, tmpOriginalLevel_imag, tmpTransformOut_real, tmpTransformOut_imag);
        	reconLevel(tmpTransformOut_real, tmpTransformOut_imag, tmpCroppedFilter, tmpCurLevelFrame_real, tmpCurLevelFrame_imag);
        	updateMagnifiedLumaFFT(tmpMagnifiedLumaFFT_real, tmpMagnifiedLumaFFT_imag, tmpFiltIDX_selected, tmpCurLevelFrame_real, tmpCurLevelFrame_imag);

        	magnifiedLumaFFT_real_local[j] = tmpMagnifiedLumaFFT_real;
        	magnifiedLumaFFT_imag_local[j] = tmpMagnifiedLumaFFT_imag;

        	//magnifiedLumaFFT_real[i + j] = tmpMagnifiedLumaFFT_real;
        	//magnifiedLumaFFT_imag[i + j] = tmpMagnifiedLumaFFT_imag;
        }

       out_write:
	   for (int j = 0; j < chunk_size; j++) {
		   #pragma hls pipeline
		   #pragma HLS LOOP_TRIPCOUNT min=1 max=32
		   magnifiedLumaFFT_real[i + j] = magnifiedLumaFFT_real_local[j];
		   magnifiedLumaFFT_imag[i + j] = magnifiedLumaFFT_imag_local[j];
	   }


    }



	}
}




