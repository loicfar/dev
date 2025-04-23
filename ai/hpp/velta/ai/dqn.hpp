#pragma once

#include "loicfar/ai/nn.hpp"

#include <torch/torch.h>

namespace loicfar::ai
{
    class dqn : public nn
    {
    private:
        torch::nn::Linear layer1_;
        torch::nn::Linear layer2_;
        torch::nn::Linear layer3_;

    public:
        dqn(std::string w_file, std::size_t n_obs, std::size_t n_actions, int n_nodes, const options& opt);
        virtual ~dqn();

        virtual torch::Tensor forward(torch::Tensor& t) override;
    };
}