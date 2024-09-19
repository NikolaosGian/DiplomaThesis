#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <tuple>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <valarray>
#include <complex> 


#include "hls_x_complex.h"
#include "hls_stream.h"
#include "hls_math.h"

#include <ap_fixed.h>
#include "ap_int.h"





//--------------------Function for Matrix and other stuff----------------------------------------

int factorial(int n){

     return (n==0) || (n==1) ? 1 : n* factorial(n-1);
}


template <typename T>
std::vector<std::vector<T>> createZerosMatrix(int rows, int cols) {
    return std::vector<std::vector<T>>(rows, std::vector<T>(cols, T(0)));
}

// Function to calculate maxSCFpyrHt
template <typename T>
int maxSCFpyrHt(const std::vector<std::vector<T>>& im) {
    int minHeight = std::min(im.size(), im[0].size());
    int maxHeight = static_cast<int>(std::floor(std::log2(minHeight))) - 2;
    return maxHeight;
}

// Templated function to perform element-wise multiplication of two matrices
template <typename T>
std::vector<std::vector<T>> elementWiseMultiply( std::vector<std::vector<T>>& matrix1,  std::vector<std::vector<T>>& matrix2) {
    // Determine the dimensions of the result matrix
    size_t maxRows = std::max(matrix1.size(), matrix2.size());
    size_t maxCols = std::max(matrix1[0].size(), matrix2[0].size());

    // Resize the smaller matrix to have the same dimensions as the larger matrix
    if (matrix1.size() < maxRows) {
        matrix1.resize(maxRows, std::vector<T>(maxCols, static_cast<T>(0)));
    }
    if (matrix2.size() < maxRows) {
        matrix2.resize(maxRows, std::vector<T>(0));
    }
    for (size_t i = 0; i < matrix1.size(); ++i) {
        if (matrix1[i].size() < maxCols) {
            matrix1[i].resize(maxCols, static_cast<T>(0));
        }
    }
    for (size_t i = 0; i < matrix2.size(); ++i) {
        if (matrix2[i].size() < maxCols) {
            matrix2[i].resize(maxCols, static_cast<T>(0));
        }
    }

    // Create a result matrix with the same dimensions as the input matrices
    std::vector<std::vector<T>> result(matrix1.size(), std::vector<T>(matrix1[0].size(), static_cast<T>(0)));

    
    // Perform element-wise multiplication
    for (size_t i = 0; i < matrix1.size(); ++i) {
        for (size_t j = 0; j < matrix1[0].size(); ++j) {
            result[i][j] = matrix1[i][j] * matrix2[i][j];
            
        }
    }

    return result;
}

// Function to compute element-wise power for vectors of vectors (2D vectors)
std::vector<std::vector<double>> elementwisePower(const std::vector<std::vector<double>>& base,
                                                  const std::vector<std::vector<double>>& exponent) {
    // Check if the vectors have the same size
    if (base.size() != exponent.size() || base.empty() || base[0].size() != exponent[0].size() || exponent.empty()) {
        std::cerr << "Vectors must have the same non-empty size." << std::endl;
        return {};
    }

    // Result vector
    std::vector<std::vector<double>> result(base.size(), std::vector<double>(base[0].size(), 0.0));

    // Perform element-wise power
    for (size_t i = 0; i < base.size(); ++i) {
        for (size_t j = 0; j < base[0].size(); ++j) {
            result[i][j] = std::pow(base[i][j], exponent[i][j]);
        }
    }

    return result;
}

std::vector<std::vector<std::complex<double>>> elementWiseDivideComplex(const std::vector<std::vector<std::complex<double>>>& vec1,
                                                               const std::vector<std::vector<std::complex<double>>>& vec2) {
    // Assuming vec1 and vec2 have the same dimensions

    std::vector<std::vector<std::complex<double>>> result;

    for (size_t i = 0; i < vec1.size(); ++i) {
        std::vector<std::complex<double>> rowResult;

        for (size_t j = 0; j < vec1[i].size(); ++j) {
            // Perform element-wise division
            std::complex<double> divisionResult = vec1[i][j] / vec2[i][j];
            rowResult.push_back(divisionResult);
        }

        result.push_back(rowResult);
    }

    return result;
}

std::vector<std::vector<std::complex<double>>> elementWiseDivideComplexDouble(const std::vector<std::vector<std::complex<double>>>& vec1,
                                                               const std::vector<std::vector<double>>& vec2) {
    // Assuming vec1 and vec2 have the same dimensions
    
    if( vec1.size() != vec2.size() || vec1[0].size() != vec2[0].size()){
    	printf("ERROR at elementWiseDivideComplexDouble\n");	
    	printf("vec1.size() = %d vec2.size() = %d \n vec1[0].size() = %d vec2[0].size() = %d\n", vec1.size(), vec2.size(), vec1[0].size(), vec1[0].size());
    }

    std::vector<std::vector<std::complex<double>>> result;

    for (size_t i = 0; i < vec1.size(); ++i) {
        std::vector<std::complex<double>> rowResult;

        for (size_t j = 0; j < vec1[i].size(); ++j) {
            // Perform element-wise division
            std::complex<double> divisionResult = vec1[i][j] / vec2[i][j];
            rowResult.push_back(divisionResult);
        }

        result.push_back(rowResult);
    }

    return result;
}

// Function to perform element-wise division of a matrix by a scalar
template <typename T>
std::vector<std::vector<T>> elementWiseDivide(const std::vector<std::vector<T>>& matrix, T divisor) {
    // Create a result matrix with the same dimensions as the input matrix
    std::vector<std::vector<T>> result(matrix.size(), std::vector<T>(matrix[0].size(), static_cast<T>(0)));

    // Perform element-wise division
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[0].size(); ++j) {
            result[i][j] = matrix[i][j] / divisor;
        }
    }

    return result;
}

std::vector<std::vector<std::complex<double>>> computeAbsoluteValue(const std::vector<std::vector<std::complex<double>>>& input) {
    std::vector<std::vector<std::complex<double>>> result;

    for (const auto& row : input) {
        std::vector<std::complex<double>> rowResult;
        for (const auto& complexNumber : row) {
            // Compute the absolute value of the complex number
            std::complex<double> absValue = std::abs(complexNumber);
            rowResult.push_back(absValue);
        }
        result.push_back(rowResult);
    }
    return result;
}


std::vector<std::vector<std::complex<double>>> transposeComplex(const std::vector<std::vector<std::complex<double>>>& matrix) {
    // Get the dimensions of the original matrix
    size_t rows = matrix.size();
    size_t cols = matrix[0].size() ;
   
    // Create a matrix with swapped dimensions for the result
    std::vector<std::vector<std::complex<double>>> result(cols, std::vector<std::complex<double>>(rows));

    // Perform the transpose
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[j][i] = matrix[i][j];
        }
    }

    return result;
}


// Function to perform circular shift on a vector
template<typename T>
void circularShift(std::vector<T>& vec, int shift) {
    std::rotate(vec.begin(), vec.begin() + shift, vec.end());
}


// Function to perform fftshift on a 2D complex matrix
void fftshift(std::vector<std::vector<std::complex<double>>>& matrix) {
    const size_t numRows = matrix.size();
    const size_t numCols = matrix[0].size();

    // Calculate the shift amounts for rows and columns
    int shiftRows = numRows / 2;
    int shiftCols = numCols / 2;

    // Apply circular shift to rows
    for (auto& row : matrix) {
        circularShift(row, shiftCols);
    }

    // Apply circular shift to columns
    for (size_t j = 0; j < numCols; ++j) {
        std::vector<std::complex<double>> column;
        column.reserve(numRows);
        for (size_t i = 0; i < numRows; ++i) {
            column.push_back(matrix[i][j]);
        }
        circularShift(column, shiftRows);
        for (size_t i = 0; i < numRows; ++i) {
            matrix[i][j] = column[i];
        }
    }
}



#define floatT float

#define PI 3.14159265358979323846


// Perform 1D DFT
void dft1d(std::vector<std::complex<double>>& data, int size, bool inverse) {
    std::vector<std::complex<double>> result(size);
    double sign = inverse ? 1.0 : -1.0;
    for (int k = 0; k < size-1; ++k) {
        result[k] = 0;
        for (int n = 0; n < size-1; ++n) {
            double angle = 2 * PI * k * n / size;
            result[k] += data[n] * std::polar(1.0, sign * angle);
        }
        if (inverse) {
            result[k] /= size;
        }
    }
    for (int i = 0; i < size-1; ++i) {
        data[i] = result[i];
    }
}

// Perform 2D DFT
void dft2d(std::vector<std::vector<std::complex<double>>>& data, bool inverse) {
    // DFT of rows
    for (int i = 0; i < N-1; ++i) {
        dft1d(data[i], N, inverse);
    }

    // DFT of columns
    std::vector<std::complex<double>> column(N);
    for (int j = 0; j < N-1; ++j) {
        for (int i = 0; i < N-1; ++i) {
            column[i] = data[i][j];
        }
        dft1d(column, N, inverse);
        for (int i = 0; i < N-1; ++i) {
            data[i][j] = column[i];
        }
    }
}

// Perform 2D IDFT
void ifft2(std::vector<std::vector<std::complex<double>>>& in, std::vector<std::vector<std::complex<double>>>& out) {
    bool inverse = true;
    dft2d(in, inverse);

    int l = N;
    for (int i = 0; i < l-1; ++i) {
        for (int j = 0; j < l-1; ++j) {
            out[i][j] = in[i][j];
        }
    }
}

void fft2(std::vector<std::vector<std::complex<double>>>& in, std::vector<std::vector<std::complex<double>>>& out) {

	bool inverse = false;
    dft2d(in, inverse);
    /*
    int l = N;
    for (int i = 0; i < l-1; ++i) {
        for (int j = 0; j < l-1; ++j) {
            out[i][j] = in[i][j];
        }
    }
    */
    out =in;
}



std::vector<std::vector<double>> calculatePhaseAngles(const std::vector<std::vector<std::complex<double>>>& complexVector) {
    

    size_t numRows = complexVector.size();
    size_t numCols = complexVector[0].size();
    
    std::vector<std::vector<double>> phaseAngles(numRows, std::vector<double>(numCols , 0.0));


    for (size_t i = 0; i < numRows; ++i) {
        for (size_t j = 0; j < numCols; ++j) {
            phaseAngles[i][j] = std::arg(complexVector[i][j]);           
        }
    }

    return phaseAngles;
}

double customMod(double a, double m) {
    // Check if m is zero to follow the convention
    if (m == 0) {
        return a;
    }

    // Compute the remainder after division of a by m
    double result = a - m * std::floor(a / m);
    
    return result;
}

// Function to compute delta based on pyrRef and pyrCurrent for a specific frameIDX
void computeDeltaForFrame(const std::vector<std::vector<double>>& pyrRefAngle,
                           const std::vector<std::vector<double>>& pyrCurrent,
                           std::vector<std::vector<std::vector<double>>>& delta,
                           int frameIDX) {
    // Assuming pyrRefAngle, pyrCurrent, and delta have the same dimensions

    size_t numRows = delta[0].size();
    size_t numCols = delta[0][0].size();
    
   
    for (size_t i = 0; i < numRows; ++i) {
        for (size_t j = 0; j < numCols; ++j) {
            // Compute delta for each element 3.1416 
            delta[frameIDX][i][j] = static_cast<float>(customMod((3.1416+ pyrCurrent[i][j] - pyrRefAngle[i][j]), (3.1416*2) ) - 3.1416);
        }
    }
}


void differenceOfIIR(std::vector<std::vector<std::vector<double>>>& delta, double rl, double rh) {
   

    std::vector<std::vector<double>> lowpass1(delta[0].size(), std::vector<double>(delta[0][0].size()));
    std::vector<std::vector<double>> lowpass2(delta[0].size(), std::vector<double>(delta[0][0].size()));

    for (size_t i = 0; i < delta[0].size(); ++i) {
        for (size_t k = 0; k < delta[0][0].size(); ++k) {
            lowpass1[i][k] = delta[0][i][k];
            lowpass2[i][k] = lowpass1[i][k];
            delta[0][i][k] = 0.0;
        }
    }


    for (size_t i = 0; i < delta[0][0].size(); ++i) {
        for (size_t j = 0; j < delta.size(); ++j) { 
            for (size_t k = 0; k < delta[j].size(); ++k) {
                lowpass1[k][i] = (1.0 - rh) * lowpass1[k][i] + rh * delta[j][k][i];
                lowpass2[k][i] = (1.0 - rl) * lowpass2[k][i] + rl * delta[j][k][i];
                delta[j][k][i] = lowpass1[k][i] - lowpass2[k][i];
   
            }
        }
    }
}

    
    // Butterworth filter function
std::pair<std::vector<double>, std::vector<double>> butter(int order, double cutoff, const std::string& type) {
    double wc = std::tan(cutoff * 3.1416);
    std::vector<double> a(order + 1, 0.0);
    std::vector<double> b(order + 1, 0.0);

    if (type == "low") {
        for (int k = 0; k <= order; ++k) {
            b[k] = std::tgamma(order + 1) / (std::tgamma(k + 1) * std::tgamma(order - k + 1)) * std::pow(wc, order - k);
            a[k] = std::tgamma(order + 1) / (std::tgamma(k + 1) * std::tgamma(order - k + 1)) * std::pow(wc, k);
        }
    } else if (type == "high") {
        for (int k = 0; k <= order; ++k) {
            b[k] = std::pow(-1, k) * std::tgamma(order + 1) / (std::tgamma(k + 1) * std::tgamma(order - k + 1)) * std::pow(wc, order - k);
            a[k] = std::tgamma(order + 1) / (std::tgamma(k + 1) * std::tgamma(order - k + 1)) * std::pow(wc, k);
        }
    } else {
        throw std::invalid_argument("Invalid filter type. Use 'low' or 'high'.");
    }

    return std::make_pair(a, b);
}

void differenceOfButterworths(std::vector<std::vector<std::vector<double>>>& delta, double fl, double fh) {


    auto [low_a, low_b] = butter(1, fl, "low");
    auto [high_a, high_b] = butter(1, fh, "low");

    size_t len = delta[0][0].size();

    std::vector<std::vector<std::vector<double>>> lowpass1 = delta;
    std::vector<std::vector<std::vector<double>>> lowpass2 = lowpass1;
    std::vector<std::vector<std::vector<double>>> prev = lowpass1;

    for (size_t i = 1; i < len; ++i) {
        for (size_t j = 0; j < delta.size(); ++j) {
            for (size_t k = 0; k < delta[j].size(); ++k) {
                lowpass1[j][k][i] = (-high_b[1] * lowpass1[j][k][i] + high_a[0] * delta[j][k][i] + high_a[1] * prev[j][k][i]) / high_b[0];
                lowpass2[j][k][i] = (-low_b[1] * lowpass2[j][k][i] + low_a[0] * delta[j][k][i] + low_a[1] * prev[j][k][i]) / low_b[0];
                prev[j][k][i] = delta[j][k][i];
                delta[j][k][i] = lowpass1[j][k][i] - lowpass2[j][k][i];
            }
        }
    }
}


// C++ equivalent of MATLAB meshgrid function for 2-D grid
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> meshgrid(const std::vector<double>& x, const std::vector<double>& y) {
    std::vector<std::vector<double>> X(y.size(), std::vector<double>(x.size()));
    std::vector<std::vector<double>> Y(y.size(), std::vector<double>(x.size()));

    for (size_t i = 0; i < y.size(); ++i) {
        for (size_t j = 0; j < x.size(); ++j) {
            X[i][j] = x[j]; // Assign x directly
            Y[i][j] = y[i]; // Assign y directly
        }
    }

    return std::make_pair(X, Y);
}

void getPolarGrid(const std::vector<int>& dimension ,std::vector<std::vector<double>>& angle, std::vector<std::vector<double>>& rad) {
    std::vector<int> center{ static_cast<int>(std::ceil((dimension[0] + 0.5) / 2)), static_cast<int>(std::ceil((dimension[1] + 0.5) / 2)) };

    std::vector<double> x(dimension[1]);
    std::vector<double> y(dimension[0]);

    // Correct variable names for center_x and center_y
    int center_x = center[1];
    int center_y = center[0];

    for (int i = 0; i < dimension[1]; ++i) {
        x[i] = (i + 1 - center_x) / (static_cast<double>(dimension[1]) / 2.0); // Correct dimension[1]
    }

    for (int i = 0; i < dimension[0]; ++i) {
        y[i] = (i + 1 - center_y) / (static_cast<double>(dimension[0]) / 2.0); // Correct dimension[0]
    }

    // Call meshgrid function
    auto result = meshgrid(x, y);

    // Convert to polar coordinates
    angle.resize(dimension[0], std::vector<double>(dimension[1], 0.0));
    rad.resize(dimension[0], std::vector<double>(dimension[1], 0.0));

    for (int i = 0; i < dimension[0]; ++i) {
        for (int j = 0; j < dimension[1]; ++j) {
            angle[i][j] = std::atan2(result.second[i][j], result.first[i][j]); // Use result instead of result2D
            rad[i][j] = std::sqrt(result.first[i][j] * result.first[i][j] + result.second[i][j] * result.second[i][j]); // Use result instead of result2D

        }
    }

    rad[center_y][center_x] = rad[center_y][center_x - 1];
} 


// C++ equivalent of MATLAB clip function
std::vector<std::vector<double>> clip(const std::vector<std::vector<double>>& im, double minValOrRange, double maxVal) {
    // Initialize minVal and maxVal based on input
    double minVal, range;
    if (minValOrRange == 0.0 && maxVal == 0.0) {
        minVal = 0.0;
        maxVal = 1.0;
    } else if (maxVal == 0.0) {
        minVal = minValOrRange;
        maxVal = minVal + 1.0;
    } else {
        minVal = minValOrRange;
    }

    // Check if maxVal is less than minVal
    if (maxVal < minVal) {
        // Handle the error as needed, e.g., throw an exception
        // You can customize this part based on your error handling strategy
        throw std::invalid_argument("MAXVAL should be less than MINVAL");
    }

    // Create a copy of the input matrix to store the result
    std::vector<std::vector<double>> res = im;

    // Apply clipping
    for (size_t i = 0; i < im.size(); ++i) {
        for (size_t j = 0; j < im[0].size(); ++j) {
            if (im[i][j] < minVal) {
                res[i][j] = minVal;
            } else if (im[i][j] > maxVal) {
                res[i][j] = maxVal;
            }
        }
    }

    return res;
}

// Function to get radial mask pair
void getRadialMaskPair(double r, const std::vector<std::vector<double>>& rad, double tWidth ,std::vector<std::vector<double>>& himask , std::vector<std::vector<double>>& lomask ) {
    // Calculate log_rad
    
    std::vector<std::vector<double>> logRad(rad.size(), std::vector<double>(rad[0].size(), 0.0));
    
    for (size_t i = 0; i < rad.size(); ++i) {
        for (size_t j = 0; j < rad[0].size(); ++j) {
            logRad[i][j] = std::log2(rad[i][j]) - std::log2(r);
        }
    }

    
    std::vector<std::vector<double>> temp = clip(logRad, -tWidth, 0.0);

    for (size_t i = 0; i < rad.size(); ++i) {
        for (size_t j = 0; j < rad[0].size(); ++j) {       
            himask[i][j] = temp[i][j] * 3.1416 / (2 * tWidth);
            himask[i][j] = std::abs(std::cos(temp[i][j]));
        }
    }

    
    
    for (size_t i = 0; i < rad.size(); ++i) {
        for (size_t j = 0; j < rad[0].size(); ++j) {
            lomask[i][j] = std::sqrt(1.0 - himask[i][j] * himask[i][j]);
        }
    }

    
   }



// Function to get angle mask
std::vector<std::vector<double>> getAngleMask(int b, int orientations, const std::vector<std::vector<double>>& angle) {
    int order = orientations - 1;
    double constValue = std::pow(2.0, 2 * order) * std::pow(std::tgamma(order + 1), 2) / (orientations * std::tgamma(2 * order + 1));

    std::vector<std::vector<double>> anglemask(angle.size(), std::vector<double>(angle[0].size(), 0.0));

    for (size_t i = 0; i < angle.size(); ++i) {
        for (size_t j = 0; j < angle[0].size(); ++j) {
            double adjustedAngle = std::fmod(3.1416 + angle[i][j] - 3.1416 * (b - 1) / orientations, 2 * 3.1416) - 3.1416;
            anglemask[i][j] = 2 * std::sqrt(constValue) * std::pow(std::cos(adjustedAngle), order) * (std::abs(adjustedAngle) < 3.1416 / 2);
        }
    }

    return anglemask;
}

void printFilters(const std::vector<std::vector<std::vector<double>>>& filters) {
    for (size_t layerIdx = 0; layerIdx < filters.size(); ++layerIdx) {
        const auto& layer = filters[layerIdx];
        for (size_t rowIdx = 0; rowIdx < layer.size(); ++rowIdx) {
            const auto& row = layer[rowIdx];
            for (size_t colIdx = 0; colIdx < row.size(); ++colIdx) {
                std::cout << "Layer: " << layerIdx << " Row: " << rowIdx << " Col: " << colIdx << " Value: " << row[colIdx] << std::endl;
            }
            std::cout << std::endl;  // Separate rows with an empty line
        }
        std::cout << std::endl;  // Separate layers with an empty line
    }
}



std::vector<std::vector<std::vector<double>>> getFilters(const std::vector<int>& dimension, const std::vector<double>& rVals, int orientations, double tWidth) {

    std::vector<std::vector<std::vector<double>>> filters;

    std::vector<std::vector<double>> angle;
    std::vector<std::vector<double>> logRad;

   
    getPolarGrid(dimension,angle,logRad);

    std::vector<std::vector<double>> himask(logRad.size(), std::vector<double>(logRad[0].size(), 0.0));
    std::vector<std::vector<double>> lomaskPrev(logRad.size(), std::vector<double>(logRad[0].size(), 0.0));


     // Call getRadialMaskPair function
    getRadialMaskPair(rVals[0], logRad, tWidth, himask, lomaskPrev);
    
    filters.push_back(himask);
  
    std::vector<std::vector<double>> lomask(logRad.size(), std::vector<double>(logRad[0].size(), 0.0));

    for (size_t k = 1; k < rVals.size(); ++k) {
        std::vector<std::vector<double>> himask(logRad.size(), std::vector<double>(logRad[0].size(), 0.0));
        
        // Call your getRadialMaskPair implementation
        //std::tie(himask, lomask) = getRadialMaskPair(rVals[k], logRad, tWidth);
        getRadialMaskPair(rVals[k], logRad, tWidth,himask,lomask);

        std::vector<std::vector<double>> radMask;
        radMask = elementWiseMultiply(himask, lomask);


        for (int j = 0; j < orientations; ++j) {
            std::vector<std::vector<double>> anglemask;
            
            anglemask = getAngleMask(j, orientations, angle);

            
            // Perform element-wise division by 2
    	    std::vector<std::vector<double>> result = elementWiseDivide(anglemask, 2.0);
    	    
    	    std::vector<std::vector<double>> filter = elementWiseMultiply(radMask, result);    
    	          
            filters.push_back(filter);

        }

        lomaskPrev = lomask;
        
    }

    filters.push_back(lomask);
    return filters;
}



std::vector<bool> sumColumns(const std::vector<std::vector<double>>& matrix) {

    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    std::vector<bool> result(cols, false);

    for (size_t j = 0; j < cols; ++j) {
        for (size_t i = 0; i < rows; ++i) {
            result[j] = result[j] || (matrix[i][j] > 0.0);
        }
    }

    return result;
    
}

std::vector<int> find(const std::vector<bool>& vector) {
    
    std::vector<int> result;

    for (size_t i = 0; i < vector.size(); ++i) {
        if (vector[i]) {
            result.push_back(i);
        }
    }

    return result;
}



std::vector<int> findRange(const std::vector<int>& vector) {
    if (vector.empty()) {
        return {};
    }

     return { *std::min_element(vector.begin(), vector.end()), *std::max_element(vector.begin(), vector.end()) };
}






std::vector<bool> rotateBool(const std::vector<bool>& vector, int times) {
 
    size_t size = vector.size();
    std::vector<bool> result(size, false);

    for (size_t i = 0; i < size; ++i) {
        result[i] = vector[(i + times) % size];
    }

    return result;
}

std::vector<bool> logicalOr(const std::vector<bool>& vector1, const std::vector<bool>& vector2) {

    
    size_t size = std::max(vector1.size(), vector2.size());
    std::vector<bool> result(size, false);

    for (size_t i = 0; i < size; ++i) {
        if (i < vector1.size()) {
            result[i] =result[i] || vector1[i];
        }
        if (i < vector2.size()) {
            result[i] = result[i] || vector2[i];
        }
    }

    return result;
}

std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& matrix) {
    
    
    std::vector<std::vector<double>> result(matrix[0].size(), std::vector<double>(matrix.size(),0.0));

    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[0].size(); ++j) {
            result[j][i] = matrix[i][j];
        }
    }

    return result;
}




std::vector<std::vector<double>> getSubmatrix(const std::vector<std::vector<double>>& filter,
                                           const std::vector<int>& indices1,
                                           const std::vector<int>& indices2) {
    std::vector<std::vector<double>> croppedFilter;
    for (int i : indices1) {
    	//printf("i = %d filters.size() = %d \n", i, static_cast<int>(filter.size()));
        if (i >= 0 && i <= static_cast<int>(filter.size())) {
            std::vector<double> row;
            for (int j : indices2) {
                if (j >= 0 && j <= static_cast<int>(filter[0].size())) {
                    row.push_back(filter[i][j]);
                } else {
                    // Handle column index out of bounds
                    // You can add appropriate error handling or return a default value.
                    std::cerr << "Column index out of bounds!" << std::endl;
                }
            }
            croppedFilter.push_back(row);
        } else {
            // Handle row index out of bounds
            // You can add appropriate error handling or return a default value.
            std::cerr << "Row index out of bounds!" << std::endl;
        }
    }

    return croppedFilter;
}



std::vector<std::vector<int>> getIDXFromFilter(const std::vector<std::vector<double>>& filter) {
    std::vector<std::vector<int>> filtIDX(2);

    // Create aboveZero matrix
    std::vector<std::vector<int>> aboveZero(filter.size(), std::vector<int>(filter[0].size(), 0));

    for (size_t i = 0; i < filter.size(); ++i) {
        for (size_t j = 0; j < filter[0].size(); ++j) {
            aboveZero[i][j] = filter[i][j] > 1e-10 ? 1 : 0;
        }
    }

    // Compute dim1 and dim2
    std::vector<int> dim1(filter.size(), 0);
    std::vector<int> dim2(filter[0].size(), 0);

    for (size_t i = 0; i < filter.size(); ++i) {
        dim1[i] = std::any_of(aboveZero[i].begin(), aboveZero[i].end(), [](int val) { return val == 1; });
    }

    for (size_t j = 0; j < filter[0].size(); ++j) {
        dim2[j] = std::any_of(aboveZero.begin(), aboveZero.end(), [j](const std::vector<int>& row) { return row[j] == 1; });
    }

    // Perform element-wise OR
    for (size_t i = 0; i < dim1.size(); ++i) {
        dim1[i] |= dim1[dim1.size() - 1 - i];
    }

    for (size_t j = 0; j < dim2.size(); ++j) {
        dim2[j] |= dim2[dim2.size() - 1 - j];
    }

    // Compute idx1 and idx2
    std::vector<int> idx1(filter.size());
    std::vector<int> idx2(filter[0].size());

    std::iota(idx1.begin(), idx1.end(), 0);
    std::iota(idx2.begin(), idx2.end(), 0);

    idx1.erase(std::remove_if(idx1.begin(), idx1.end(), [&dim1](int i) { return dim1[i] != 1; }), idx1.end());
    idx2.erase(std::remove_if(idx2.begin(), idx2.end(), [&dim2](int i) { return dim2[i] != 1; }), idx2.end());

    // Populate filtIDX
    filtIDX[0] = idx1;
    filtIDX[1] = idx2;

    return filtIDX;
}


// Function to crop a matrix based on indices
std::vector<std::vector<double>> cropMatrix(const std::vector<std::vector<double>>& matrix,
                                            const std::vector<int>& rowIndices,
                                            const std::vector<int>& colIndices) {
    std::vector<std::vector<double>> croppedMatrix;

    for (size_t i : rowIndices) {
        std::vector<double> row;

        for (size_t j : colIndices) {
            row.push_back(matrix[i][j]);
        }

        croppedMatrix.push_back(row);
    }

    return croppedMatrix;
}


// Function to get the cropped filters and corresponding indices
std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<std::vector<int>>>> getFilterIDX(const std::vector<std::vector<std::vector<double>>>& filters) {

    size_t nFilts = filters.size();		
 
    std::vector<std::vector<std::vector<int>>> filtIDX;
    filtIDX.resize(nFilts, std::vector<std::vector<int>>(2));
    std::vector<std::vector<std::vector<double>>> croppedFilters(nFilts);

    for (size_t k = 0; k < nFilts; ++k) {
         
         std::vector<std::vector<int>> indices = getIDXFromFilter(filters[k]);
         
         // Print the vector with info
    	 //print2DVectorWithInfo(indices);

        // Assign to filtIDX
        filtIDX[k][0] = indices[0];
        filtIDX[k][1] = indices[1];
        
            // Call getSubmatrix
        croppedFilters[k] = cropMatrix(filters[k], indices[0], indices[1]);

       
    }
    
    return std::make_tuple(croppedFilters, filtIDX);
}



std::vector<std::vector<std::complex<double>>> buildLevel( std::vector<std::vector<std::complex<double>>>& im_dft,
                        const std::vector<std::vector<std::vector<double>>>& croppedFilters,
                        const std::vector<std::vector<std::vector<int>>>& filtIDX,
                        int k) {

	//type_time stime_buildLevel = std::chrono::high_resolution_clock::now();

    size_t height = im_dft.size();
    size_t width = (height > 0) ? im_dft[0].size(): 0;
	
    std::vector<std::vector<std::vector<std::complex<double>>>>  resCropped(croppedFilters.size(), std::vector<std::vector<std::complex<double>>>(height,std::vector<std::complex<double>>(width,{0.0,0.0})));
 
    for (size_t frame = 0; frame < resCropped.size(); ++frame) {
        for (size_t i = 0; i < height; ++i) {
            for (size_t j = 0; j < width; ++j) {
                if (i < croppedFilters[frame].size() && j < croppedFilters[frame][i].size()) {
                    std::complex<double> complexCropped(croppedFilters[frame][i][j], 0.0);
                    resCropped[frame][i][j] = complexCropped;
                }
            }
        }
    }	
    
    
	
    std::vector<std::vector<std::complex<double>>> result(height, std::vector<std::complex<double>>(width));
    result = elementWiseMultiply(resCropped[k],im_dft);

    
    fftshift(result);

    ifft2(result,result);


	//type_time etime_buildLevel = std::chrono::high_resolution_clock::now();
	//type_time_calc buildLevel_duration = etime_buildLevel - stime_buildLevel;
	//profilling_file << "Frame "<< k <<" BuildLevel Execution time: " << buildLevel_duration.count() << " seconds." << std::endl;

    return result;
}



std::vector<std::vector<std::complex<double>>> reconLevel(
    std::vector<std::vector<std::complex<double>>>& im_dft,
    const std::vector<std::vector<double>>& croppedFilter) {

    size_t numRows = im_dft.size();
    size_t numCols = im_dft[0].size();
    
   
    

    std::vector<std::vector<std::complex<double>>> result(numRows, std::vector<std::complex<double>>(numCols));
    
    
    std::vector<std::vector<std::complex<double>>> resCropped(numRows, std::vector<std::complex<double>>(numCols,{0.0,0.0}));

   
    for (size_t i = 0; i < numRows; ++i) {
       for (size_t j = 0; j < numCols; ++j) {
          if (i < croppedFilter.size() && j < croppedFilter[i].size()) {
             std::complex<double> complexCropped(croppedFilter[i][j], 0.0);
             resCropped[i][j] = complexCropped;
         }

       }
    }
    /*
    std::vector<std::vector<std::complex<double>>> temp_in(numRows, std::vector<std::complex<double>>(numCols));
    std::vector<std::vector<std::complex<double>>> temp(numRows, std::vector<std::complex<double>>(numCols));

    for (size_t i = 0; i < numRows; ++i) {
       for (size_t j = 0; j < numCols; ++j) {
    	   temp_in[i][j] = im_dft[i][j];
    	   std::cout<< temp_in[i][j] << " ";
       }
       std::cout <<"\n";
    }
    */
    std::vector<std::vector<std::complex<double>>> temp(numRows, std::vector<std::complex<double>>(numCols));

    	
    fft2(im_dft ,temp);
    fftshift(temp);


    std::complex<double> complexScalarTwo(2.0, 0.0);
    
    std::vector<std::vector<std::complex<double>>> elementWiseTemp(numRows, std::vector<std::complex<double>>(numCols));
    elementWiseTemp = elementWiseMultiply(resCropped, temp);
    
    
    for (size_t i = 0; i < numRows -1 ; ++i) {
        for (size_t j = 0; j < numCols - 1; ++j) {
            
           result[i][j] = complexScalarTwo * elementWiseTemp[i][j];
           
        }
    }
    

    return result;
}


void updateMagnifiedLumaFFT(std::vector<std::vector<std::vector<std::complex<double>>>>& magnifiedLumaFFT,
                             const std::vector<std::vector<std::vector<int>>>& filtIDX,
                             const std::vector<std::vector<std::complex<double>>>& curLevelFrame,
                             int level,
                             int frameIDX) {
                             

     	    if(level >= 0 && level < filtIDX.size()){
	    	for(int i = 0; i < filtIDX[level][0].size(); i++){
	    		for(int j = 0; j < filtIDX[level][1].size(); j++){
	    		  	magnifiedLumaFFT[frameIDX][filtIDX[level][0][i]][filtIDX[level][1][j]] += curLevelFrame[filtIDX[level][0][i]][filtIDX[level][1][j]];
	    		}
	    	}
	    
	    }
}



// Function to print a 2D vector of doubles
void print2DCompVector(std::vector<std::vector<std::complex<double>>>& vec) {
    for (const auto& row : vec) {
        for (const auto& element : row) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    }
}



void updateMagnifiedLumaFFTLowpassFrame(std::vector<std::vector<std::vector<std::complex<double>>>>& magnifiedLumaFFT,
                             const std::vector<std::vector<std::vector<int>>>& filtIDX,
							 const std::vector<std::vector<std::complex<double>>>& lowpassFrame,
                             int level,
                             int frameIDX) {
                             
	    if(level >= 0 && level < filtIDX.size()){
	    	for(size_t i = 0; i < filtIDX[level][0].size(); i++){
	    		for(size_t j = 0; j < filtIDX[level][1].size(); j++){
	    			if(filtIDX[level][0][i] != -1 && filtIDX[level][1][j] != -1){
	    				magnifiedLumaFFT[frameIDX][filtIDX[level][0][i]][filtIDX[level][1][j]] += lowpassFrame[filtIDX[level][0][i]][filtIDX[level][1][j]];
	    			}
	    		}
	    	}
	    
	    }
	    

}


void updateLowpassFrame(std::vector<std::vector<std::complex<double>>>& lowpassFrame,
			const std::vector<std::vector<std::vector<std::complex<double>>>>& vidFFT,
			const std::vector<std::vector<std::vector<int>>>& filtIDX,
			const std::vector<std::vector<double>>& croppedFiltersLastPowOfTwo,
			const int level,
			const int frameIDX){
	  
	     if(level >= 0 && level < filtIDX.size()){
	    	for(size_t i = 0; i < filtIDX[level][0].size(); i++){
	    		for(size_t j = 0; j < filtIDX[level][1].size(); j++){
	    		  
	    			if(filtIDX[level][0][i] != -1 && filtIDX[level][1][j] != -1){
	    		  	// is the same dims with lowpassframe
		                std::complex<double> croppedComplex = std::complex<double>(croppedFiltersLastPowOfTwo[i][j], 0.0); // or the second 0.0?
		                
		                lowpassFrame[filtIDX[level][0][i]][filtIDX[level][1][j]] = vidFFT[frameIDX][filtIDX[level][0][i]][filtIDX[level][1][j]] * croppedComplex;
	    			}
	    		}
	    	}
	    
	    }else{
	    	printf("updateLowpassFrame level >> filtIDX.size()\n");
	    }
	    
}

std::vector<std::vector<std::complex<double>>> calculateAbsPlusEps(const std::vector<std::vector<std::complex<double>>>& originalLevel, double eps) {
    size_t height = originalLevel.size();
    size_t width = (height > 0) ? originalLevel[0].size() : 0;

    std::vector<std::vector<std::complex<double>>> result(height, std::vector<std::complex<double>>(width, {0.0,0.0}));

    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            result[i][j] = std::abs(originalLevel[i][j]) + eps;
        }
    }

    return result;
}


// Function to convert an integer-based image to a double-based image
std::vector<std::vector<std::vector<double>>> im2single2(const std::vector<std::vector<std::vector<uint8_t>>>& intImage) {
    size_t numFrames = intImage.size();
    size_t height = (numFrames > 0) ? intImage[0].size() : 0;
    size_t width = (height > 0) ? intImage[0][0].size() / 3 : 0;

    std::vector<std::vector<std::vector<double>>> doubleImage(numFrames, std::vector<std::vector<double>>(height, std::vector<double>(width * 3, 0.0)));

    for (size_t k = 0; k < numFrames; ++k) {
        for (size_t i = 0; i < height; ++i) {
            for (size_t j = 0; j < width; ++j) {
                // Assuming the image has 3 channels (RGB)
                uint8_t r = intImage[k][i][j * 3];
                uint8_t g = intImage[k][i][j * 3 + 1];
                uint8_t b = intImage[k][i][j * 3 + 2];

                // Convert each pixel to double and store in the output vector
                doubleImage[k][i][j * 3] = static_cast<double>(r) / 255.0;
                doubleImage[k][i][j * 3 + 1] = static_cast<double>(g) / 255.0;
                doubleImage[k][i][j * 3 + 2] = static_cast<double>(b) / 255.0;  
            }
        }
    }

    return doubleImage;
}


// Function to calculate exp(1i * phase) for each element in a matrix
std::vector<std::vector<std::complex<double>>> calculateExp1i(const std::vector<std::vector<double>>& phaseOfFrame) {
    // Ensure the dimensions match
    size_t numRows = phaseOfFrame.size();
    size_t numCols = (numRows > 0) ? phaseOfFrame[0].size() : 0;

    // Resulting matrix
    std::vector<std::vector<std::complex<double>>> result(numRows, std::vector<std::complex<double>>(numCols, {0.0, 0.0}));

    // Perform exp(1i * phase) calculation
    for (size_t i = 0; i < numRows; ++i) {
        for (size_t j = 0; j < numCols; ++j) {
            // Calculate exp(1i * phase[i][j])
            //result[i][j] = std::polar(1.0f, phase[i][j]);
            std::complex<double> phaseOfFrameComplex (0.0, phaseOfFrame[i][j]);
             result[i][j] = std::exp(phaseOfFrameComplex);
        }
    }

    return result;
}


void scalePhaseOfFrame(std::vector<std::vector<double>>& phaseOfFrame, double alpha) {

    for (size_t i = 0; i < phaseOfFrame.size(); ++i) {
        for (size_t j = 0; j < phaseOfFrame[i].size(); ++j) {
            phaseOfFrame[i][j] = phaseOfFrame[i][j] * alpha;
        }
    }
}


struct RGBPixel {
    double r, g, b;
};

RGBPixel rgb2ntscPixel(const RGBPixel& rgb) {
    double y = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
    double i = 0.595 * rgb.r - 0.274 * rgb.g - 0.321 * rgb.b;
    double q = 0.211 * rgb.r - 0.522 * rgb.g + 0.311 * rgb.b;
    
    y = std::max(0.0, std::min(1.0, y));
    i = std::max(-0.5959, std::min(0.5959, i));
    q = std::max(-0.5229, std::min(0.5229, q));

    return {y, i, q};
}

// Function to convert an image from RGB to NTSC
std::vector<std::vector<std::vector<double>>> rgb2ntsc(const std::vector<std::vector<std::vector<double>>>& vid) {
    size_t numFrames = vid.size();
    size_t height = (numFrames > 0) ? vid[0].size() : 0;
    size_t width = (height > 0) ? vid[0][0].size() / 3 : 0;

    std::vector<std::vector<std::vector<double>>> result(numFrames, std::vector<std::vector<double>>(height, std::vector<double>(width * 3, 0.0)));

   
    for (size_t k = 0; k < numFrames; ++k) {
        for (size_t i = 0; i < height; ++i) {
            for (size_t j = 0; j < width; ++j) {
                // Assuming the image has 3 channels (RGB)
                double r = vid[k][i][j * 3];
                double g = vid[k][i][j * 3 + 1];
                double b = vid[k][i][j * 3 + 2];
               
                RGBPixel rgbPixel = {r, g ,b};
                RGBPixel ntscPixel = rgb2ntscPixel(rgbPixel);

                // Store the result in the output vector
                result[k][i][j * 3] = ntscPixel.r;
                result[k][i][j * 3 + 1] = ntscPixel.g;
                result[k][i][j * 3 + 2] = ntscPixel.b;

                //std::cout<< "result[" << k << "][" << i <<" ]["<<j <<"]= " << result[k][i][j * 3] << "\n";
                //std::cout<< "result[" << k << "][" << i <<" ]["<<j <<" *3 +1]= " << result[k][i][j *3 +1] << "\n";
                //std::cout<< "result[" << k << "][" << i <<" ]["<<j <<"*3+2]= " << result[k][i][j *3 +2] << "\n";
               
                
            }
        }
    }

    return result;
}



// Function to convert RGB to complex
std::vector<std::vector<std::vector<std::complex<double>>>> rgbToComplex(const std::vector<std::vector<std::vector<double>>>& rgbFrames) {
    size_t numFrames = rgbFrames.size();
    size_t height = (numFrames > 0) ? rgbFrames[0].size() : 0;
    size_t width = (height > 0) ? rgbFrames[0][0].size() / 3 : 0;

    // Initialize a vector of vectors of vectors of complex numbers
    std::vector<std::vector<std::vector<std::complex<double>>>> complexFrames(numFrames, std::vector<std::vector<std::complex<double>>>(height, std::vector<std::complex<double>>(width, std::complex<double>(0.0f, 0.0f))));

    // Convert each pixel from RGB to complex for each frame
    for (size_t k = 0; k < numFrames; ++k) {
        for (size_t i = 0; i < height; ++i) {
            for (size_t j = 0; j < width; ++j) {
                // Assuming the image has 3 channels (RGB)
                double r = rgbFrames[k][i][j * 3];
                double g = rgbFrames[k][i][j * 3 + 1];
                double b = rgbFrames[k][i][j * 3 + 2];

                // Convert RGB to complex number
                std::complex<double> complexPixel(r, g + b);
                
                // Store the result in the output vector
                complexFrames[k][i][j] = complexPixel;
            }
        }
    }

    return complexFrames;
}



// Function to convert RGB to complex
std::vector<std::vector<std::vector<std::complex<double>>>> YtoComplex(const std::vector<std::vector<std::vector<double>>>& originalFrames) {
    size_t numFrames = originalFrames.size();
    size_t height = (numFrames > 0) ? originalFrames[0].size() : 0;
    size_t width = (height > 0) ? originalFrames[0][0].size() / 3 : 0;

    // Initialize a vector of vectors of vectors of complex numbers
    std::vector<std::vector<std::vector<std::complex<double>>>> complexFrames(numFrames, std::vector<std::vector<std::complex<double>>>(128, std::vector<std::complex<double>>(128, std::complex<double>(0.0, 0.0))));

    // Convert each pixel from RGB to complex for each frame
    for (size_t k = 0; k < numFrames; ++k) {
        for (size_t i = 0; i < height; ++i) {
            for (size_t j = 0; j < width; ++j) {
                // Assuming the image has 3 channels (YIQ)
            	if(i < 100 && j < 100){
                double y = originalFrames[k][i][j * 3];


                // Convert Y to complex number
                std::complex<double> complexPixel(y, 0.0);
                
                // Store the result in the output vector
                complexFrames[k][i][j] = complexPixel;
            	}else{
            		std::complex<double> complexPixel(0.0, 0.0);

            		 complexFrames[k][i][j] = complexPixel;
            	}
            }
        }
    }

    return complexFrames;
}

// Function to convert a single pixel from NTSC to RGB
RGBPixel ntsc2rgbPixel(const double y, const double i, const double q) {

    double r = y + 0.956f * i + 0.621f * q;
    double g = y - 0.272f * i - 0.647f * q;
    double b = y - 1.106f * i +  1.703f * q;
   
    // Clip values to the valid range [0, 1]
    r = std::max(0.0, std::min(1.0, r));
    g = std::max(0.0, std::min(1.0, g));
    b = std::max(0.0, std::min(1.0, b));
    
    return {r, g, b};
}

// Function to print a 2D vector of doubles
void print2DVector(const std::vector<std::vector<double>>& vec) {
    for (const auto& row : vec) {
        for (const auto& element : row) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    }
}



// Function to convert an image from NTSC to RGB
std::vector<std::vector<double>> ntsc2rgb(const std::vector<std::vector<double>>& ntscImage) {
    
    //size_t numFrames = ntscImage.size();
    size_t height = ntscImage.size();
    size_t width = (height > 0) ? ntscImage[0].size() / 3 : 0;

    std::vector<std::vector<double>> result(height, std::vector<double>(width * 3, 0.0f));
    

     for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
             // Assuming the image has 3 channels (RGB)
             double ntscY = ntscImage[i][j * 3];
             double ntscI = ntscImage[i][j * 3 + 1];
             double ntscq = ntscImage[i][j * 3 + 2];
                
             RGBPixel rgbPixel = ntsc2rgbPixel(ntscY,ntscI,ntscq);

             // Store the result in the output vector
             result[i][j * 3] = rgbPixel.r;
             result[i][j * 3 + 1] = rgbPixel.g;
             result[i][j * 3 + 2] = rgbPixel.b;
        }
     }
 
    return result;
}


// Function to convert image to 8-bit unsigned integers
std::vector<std::vector<uint8_t>> im2uint8(const std::vector<std::vector<double>>& image) {

    size_t height = image.size();
    size_t width = (height > 0) ? image[0].size() / 3 : 0;

    // Create the result vector with uint8_t data type
    std::vector<std::vector<uint8_t>> result(height, std::vector<uint8_t>(width * 3, 0));


    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
             // Assuming the image has 3 channels (RGB)
             double r = image[i][j * 3];
             double g = image[i][j * 3 + 1];
             double b = image[i][j * 3 + 2];

            // if( r < 0) {r = (-1) *r;}
            // if( g < 0) {g = (-1) *g;}
            // if( b < 0) {b = (-1) *b;}


              // Convert each pixel to uint8_t
              result[i][j * 3] = static_cast<uint8_t>(r * 255.0);
              result[i][j * 3 + 1] = static_cast<uint8_t>(g * 255.0);
              result[i][j * 3 + 2] = static_cast<uint8_t>(b * 255.0);
            }
        }
    

    return result;
}

/* Below this section having function for time calculation */




