#include "loicfar/ai/mlp.hpp"

#include "loicfar/ai/io_dataset.hpp"
#include "loicfar/ai/utils.hpp"

namespace loicfar::ai
{
    mlp::mlp(std::string w_file, std::size_t n_obs, int n1, int n2, int n_out, const options& opt) : nn(std::move(w_file), opt), layer_(nullptr)
    {
        layer_ = module_->register_module("layer", make_module<torch::nn::Sequential>());

        layer_->push_back(make_module<torch::nn::Linear>(n_obs, n1));
        layer_->push_back(make_module<torch::nn::ReLU>());
        layer_->push_back(make_module<torch::nn::Linear>(n1, n2));
        layer_->push_back(make_module<torch::nn::ReLU>());
        layer_->push_back(make_module<torch::nn::Linear>(n2, n_out));
        layer_->push_back(make_module<torch::nn::Sigmoid>());

        optim_ = opt.make_optim(module_->parameters());
    }

    mlp::~mlp() = default;

    torch::Tensor mlp::forward(torch::Tensor& t)
    {
        return layer_->forward(t);
    }

    void mlp::train(i_dataset* data, int n_epoch)
    {
        train_impl<io_dataset>(data, n_epoch);
    }
}