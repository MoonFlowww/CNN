#pragma once
#ifndef C_LAYER_H
#define C_LAYER_H

#include "config.h"
#include <vector>
#include <stdexcept>



namespace NN {

    class CLayer {
    private:
        typedef NN::Size Size;


        Size mInputSize;
        unsigned mInputDepth;

        Size mKernelSize;
        unsigned mLayerDepth;

        std::vector<std::vector<Matrix>>* mKernels;
        std::vector<Matrix>* mBiases;

        std::vector<Matrix> mInput;
        Size mOutputSize;
        std::vector<Matrix> mGradient;

    public:
        CLayer() = default;
        CLayer(const Size inputSize, unsigned inputDepth, const Size kernelSize, unsigned layerDepth,
            std::vector<std::vector<Matrix>>* kernels, std::vector<Matrix>* biases);

        void Init(const Size inputSize, unsigned inputDepth, const Size kernelSize, unsigned layerDepth,
            std::vector<std::vector<Matrix>>* kernels, std::vector<Matrix>* biases);


        std::vector<Matrix> Calculate(std::vector<Matrix>& input);


        std::vector<Matrix> BackProp(std::vector<Matrix>& gradient);

    
    };

} // namespace NN

#endif // C_LAYER_H
