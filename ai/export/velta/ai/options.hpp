#pragma once

#include "loicfar/ai/ai_api.hpp"

#include <memory>

namespace torch
{
    class Tensor;
    namespace optim
    {
        class Optimizer;
    }
}

namespace loicfar::ai
{
    enum class algo
    {
        adadelta = 0,
        adafactor = 1,
        adagrad = 2,
        adam = 3,
        adamax = 4,
        adamw = 5,
        asgd = 6,
        lbfgs = 7,
        nadam = 8,
        radam = 9,
        rmsprop = 10,
        rprop = 11,
        sgd = 12,
        sparseadam = 13
    };

    enum class loss
    {
        binary_cross_entropy = 0,
        binary_cross_entropy_with_logits = 1,
        cosine_embedding_loss = 2,
        cross_entropy = 3,
        ctc_loss = 4,
        gaussian_nll_loss = 5,
        hinge_embedding_loss = 6,
        huber_loss = 7,
        kl_div = 8,
        l1_loss = 9,
        margin_ranking_loss = 10,
        mse_loss = 11,
        multi_margin_loss = 12,
        multilabel_margin_loss = 13,
        multilabel_soft_margin_loss = 14,
        nll_loss = 15,
        poisson_nll_loss = 16,
        smooth_l1_loss = 17,
        soft_margin_loss = 18,
        triplet_margin_loss = 19,
        triplet_margin_with_distance_loss = 20,

        // Customs
        mape_loss = 21
    };

    struct options
    {
        virtual std::unique_ptr<torch::optim::Optimizer> make_optim(std::vector<torch::Tensor> p) const = 0;
        virtual std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&)> make_loss() const = 0;
    };

    template<algo ealgo, loss eloss>
    class AI_API optim final : public options
    {
    private:
        double eps_;

    public:
        optim(double eps);
        virtual std::unique_ptr<torch::optim::Optimizer> make_optim(std::vector<torch::Tensor> p) const override;
        virtual std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&)> make_loss() const override;
    };
}