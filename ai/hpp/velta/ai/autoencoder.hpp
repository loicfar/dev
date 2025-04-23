#pragma once

#include "loicfar/ai/nn.hpp"

#include <torch/torch.h>

namespace loicfar::ai
{
    class autoencoder : public nn
    {
    private:
        torch::nn::Sequential encoder_;
        torch::nn::Sequential decoder_;

    public:
        autoencoder(std::string w_file, std::size_t n, int n_nodes, const options& opt);
        virtual ~autoencoder();

        virtual torch::Tensor forward(torch::Tensor& t) override;
    };
}