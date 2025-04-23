#pragma once

#include "loicfar/ai/i_nn.hpp"
#include "loicfar/ai/dataset.hpp"
#include "loicfar/ai/options.hpp"

#include <torch/torch.h>

namespace loicfar::ai
{
    class nn : public i_nn
    {
    protected:
        std::shared_ptr<torch::nn::Module> module_;
        std::string w_file_;

        std::unique_ptr<torch::optim::Optimizer> optim_;

        std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&)> loss_;

    public:
        nn(std::string w_file, const options& opt);
        virtual ~nn();

        virtual void train(i_dataset* data, int n_epoch) override;
        
        virtual bool has_weights() const override final;
        virtual void load() override final;

    private:
        virtual double compute_impl(std::vector<double>&& p) override final;

        virtual torch::Tensor forward(torch::Tensor& t) = 0;
        torch::Tensor loss(const torch::Tensor& ref, const torch::Tensor& tgt) const;

    protected:
        virtual torch::Tensor optimize(torch::Tensor& data, torch::Tensor& tgt);
        void save(std::size_t epoch, int period = 10, bool print = false, double loss = 0.0) const;

        template<typename T>
        void train_impl(i_dataset* data, int n_epoch);

        void run();
    };
}