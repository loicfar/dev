#include "loicfar/ai/dqn.hpp"

#include "loicfar/ai/utils.hpp"

namespace loicfar::ai
{
    dqn::dqn(std::string w_file, std::size_t n_obs, std::size_t n_actions, int n_nodes, const options& opt) : nn(std::move(w_file), opt), layer1_(nullptr), layer2_(nullptr), layer3_(nullptr)
    {
        layer1_ = module_->register_module("l1", make_module<torch::nn::Linear>(n_obs, n_nodes));
        layer2_ = module_->register_module("l2", make_module<torch::nn::Linear>(n_nodes, n_nodes));
        layer3_ = module_->register_module("l3", make_module<torch::nn::Linear>(n_nodes, n_actions));

        optim_ = opt.make_optim(module_->parameters());
    }

    dqn::~dqn() = default;

    torch::Tensor dqn::forward(torch::Tensor& t)
    {
        t = torch::relu(layer1_->forward(t));
        t = torch::relu(layer2_->forward(t));
        return torch::relu(layer3_->forward(t));
    }
}