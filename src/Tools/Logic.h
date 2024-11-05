#pragma once
#include "../config.h"

namespace Tools {

    Matrix CrossCorrelation(const Matrix& input, const Matrix& kernel);


    Matrix Convolution(const Matrix& input, const Matrix& kernel); // to add

}
