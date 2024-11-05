#include "Logic.h"


namespace Tools {



	Matrix CrossCorrelation(const Matrix& input, const Matrix& kernel) {
		Matrix output(input.rows(), input.cols());

		int paddingRows = kernel.rows() - 1;
		int paddingCols = kernel.cols() - 1;


		int paddedSizeRows = input.rows() + paddingRows;
		int paddedSizeCols = input.cols() + paddingCols;

		Matrix inpPadded(paddedSizeRows, paddedSizeCols); // input Padded
		inpPadded.setZero();

		inpPadded.block(paddingRows / 2, paddingCols / 2, input.rows(), input.cols()) = input;

		for (int r = 0; r < input.rows(); ++r) {
			for (int c = 0; c < input.cols(); ++c) {
				output(r, c) = (inpPadded.block(r, c, kernel.rows(), kernel.cols()).array() * kernel.array()).sum();


			}
		}
		return output;
	}
}