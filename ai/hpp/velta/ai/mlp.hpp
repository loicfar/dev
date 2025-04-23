#pragma once

#include "loicfar/ai/nn.hpp"

#include <torch/torch.h>

namespace loicfar::ai
{
    class mlp : public nn
    {
    private:
        torch::nn::Sequential layer_;

    public:
        mlp(std::string w_path, std::size_t n_obs, int n1, int n2, int n_out, const options& opt);
        virtual ~mlp();

        virtual void train(i_dataset* data, int n_epoch) override;

        virtual torch::Tensor forward(torch::Tensor& t) override;
    };
}