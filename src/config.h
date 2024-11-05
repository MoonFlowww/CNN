#pragma once

#include <Eigen/Dense>
#include <vector>

typedef double Scalar;
typedef Eigen::Matrix < Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::Matrix < Scalar, Eigen::Dynamic, 1> ColumnVector;



namespace NN {
	struct Size {
		int Rows = -1;
		int Columns = -1;

	};
}