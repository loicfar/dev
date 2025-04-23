#include "loicfar/ai/autoencoder.hpp"

#include "loicfar/ai/utils.hpp"

namespace loicfar::ai
{
    autoencoder::autoencoder(std::string w_file, std::size_t n, int n_nodes, const options& opt) : nn(std::move(w_file), opt), encoder_(nullptr), decoder_(nullptr)
    {
        encoder_ = module_->register_module("encoder", make_module<torch::nn::Sequential>());
        decoder_ = module_->register_module("decoder", make_module<torch::nn::Sequential>());

        encoder_->push_back(make_module<torch::nn::Linear>(n, n_nodes));

        int n_temp = n_nodes;
        while (n_temp % 2 == 0)
        {
            encoder_->push_back(make_module<torch::nn::ReLU>());

            int n_temp2 = n_temp * 2;
            encoder_->push_back(make_module<torch::nn::Linear>(n_temp2, n_temp /= 2));
        }

        while (n_temp != n_nodes)
        {
            int n_temp2 = n_temp / 2;
            decoder_->push_back(make_module<torch::nn::Linear>(n_temp2, n_temp *= 2));
            decoder_->push_back(make_module<torch::nn::ReLU>());
        }

        decoder_->push_back(make_module<torch::nn::Linear>(n_nodes, n));
        decoder_->push_back(make_module<torch::nn::Sigmoid>());

        optim_ = opt.make_optim(module_->parameters());
    }

    autoencoder::~autoencoder() = default;

    torch::Tensor autoencoder::forward(torch::Tensor& t)
    {
        t = encoder_->forward(t);
        return decoder_->forward(t);
    }
}