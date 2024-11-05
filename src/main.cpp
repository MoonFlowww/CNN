#include <Eigen/dense>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "config.h"
#include "CLayer.h"
#include "Tools/debug.h"




int main() {
    typedef NN::CLayer ConvLayer; ConvLayer cLayer;

    std::vector<std::vector<Matrix>> kernels;
    std::vector<Matrix> biases;

    Matrix testInput(5, 5);
    testInput << 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2,
        3, 3, 3, 3, 3,
        4, 4, 4, 4, 4,
        5, 5, 5, 5, 5;

    Matrix testKernel1(3, 3);
    testKernel1 << 2, 2, 2,
        3, 3, 3,
        4, 4, 4;

    Matrix testKernel2(3, 3);
    testKernel2 << 5, 5, 5,
        6, 6, 6,
        7, 7, 7;



    Matrix bias1(5, 5);
    bias1 << 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2,
        3, 3, 3, 3, 3,
        4, 4, 4, 4, 4,
        5, 5, 5, 5, 5;

    Matrix bias2(5, 5);
    bias2 << 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2,
        3, 3, 3, 3, 3,
        4, 4, 4, 4, 4,
        5, 5, 5, 5, 5;

    std::vector<Matrix> layerInput;
    layerInput.emplace_back(testInput);
    layerInput.emplace_back(testInput);
    layerInput.emplace_back(testInput);

    kernels.emplace_back(std::vector<Matrix>({ testKernel1, testKernel1, testKernel1 }));
    kernels.emplace_back(std::vector<Matrix>({ testKernel2, testKernel2 ,testKernel2 }));

    biases.emplace_back(bias1);
    biases.emplace_back(bias2);

    cLayer.Init({ 5,5 }, 3, { 3,3 }, 2, &kernels, &biases);
    std::vector<Matrix> output = cLayer.Calculate(layerInput);
    std::cout << "output size : " << output.size() << std::endl;
    int i = 0;
    for (auto& out : output) {
        std::string title = "CNN Output " + std::to_string(++i);;
        std::cout << Array::MatrixArray(out, true, true, title) << std::endl;
        std::cout << "\n\n\n";
    }



    return 0;
}