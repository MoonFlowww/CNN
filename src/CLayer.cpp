#include "CLayer.h"
#include "Tools/Logic.h"


NN::CLayer::CLayer(const Size inputSize, const unsigned inputDepth, const Size kernelSize, const unsigned layerDepth, std::vector<std::vector<Matrix>>* kernels, std::vector<Matrix>* biases)
{
	Init(inputSize, inputDepth, kernelSize, layerDepth, kernels, biases);

}

void NN::CLayer::Init(const Size inputSize, const unsigned inputDepth, const Size kernelSize, const unsigned layerDepth, std::vector<std::vector<Matrix>>* kernels, std::vector<Matrix>* biases)
{


	mInputSize = inputSize;
	mInputDepth = inputDepth;

	mKernelSize = kernelSize;
	mLayerDepth = layerDepth;
	mKernels = kernels;

	mBiases = biases;





	mInput.reserve(inputDepth);

	mOutputSize = inputSize; // cross Correlation


	if (kernels && kernels->size() == 0) {
		// kernels->clear(); if size 0 = clear
		kernels->reserve(layerDepth);
	}
	if (biases && biases->size() == 0) {
		// biases->clear(); if size 0 = clear
		biases->reserve(layerDepth);
	}
	if ((biases && biases->size() == 0) && (kernels && kernels->size() == 0)) {
		for (int iKernel = 0; iKernel < layerDepth; ++iKernel) {
			biases->emplace_back(Matrix::Zero(mOutputSize.Rows, mOutputSize.Columns)); // init b = 0

			std::vector<Matrix> skernel;

			for (int i = 0; i < inputDepth; ++i) { // kernels = rdn + normalized
				skernel.emplace_back(Matrix::Random(kernelSize.Rows, kernelSize.Columns).normalized()); // norm to avoid overflow
			}

			kernels->emplace_back(std::move(skernel));
		}
	}
}


std::vector<Matrix> NN::CLayer::Calculate(std::vector<Matrix>& input) {
	if (!mKernels || !mBiases) std::domain_error("Pointers not defined.\n   Need kernels and biases.");


	mInput.clear();
	mInput = std::move(input);

	std::vector<Matrix> output;
	output.reserve(mLayerDepth);

	for (int k = 0; k < mLayerDepth; k++) {
		Matrix subOutput(mOutputSize.Rows, mOutputSize.Columns);
		subOutput.setZero();
		output.emplace_back(std::move(subOutput));
		for (int i = 0; i < mInputDepth; ++i) {
			output[k] += Tools::CrossCorrelation(mInput[i], (*mKernels)[k][i]) + (*mBiases)[k];
		}
	}

	for (auto& out : output) {
		out = out.normalized();
	}
	return output;
}

std::vector<Matrix> NN::CLayer::BackProp(std::vector<Matrix>& gradient) {

	mGradient = std::move(gradient);

	std::vector<Matrix> inputGradients;
	inputGradients.reserve(mInputDepth);

	for (int i = 0; i < mInputDepth; ++i) {
		Matrix inputGrad_tmp(mInputSize.Rows, mInputSize.Columns);
		inputGrad_tmp.setZero();
		inputGradients.emplace_back(std::move(inputGrad_tmp));
	}
	for (int k = 0; k < mInputDepth; ++k) {
		for (int i = 0; i < mInputDepth; ++i) {
			inputGradients[i] += Tools::CrossCorrelation(gradient[k], (*mKernels)[k][i].reverse());
		}
	}

	return inputGradients;
}