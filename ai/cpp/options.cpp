#include "loicfar/ai/options.hpp"

#include "loicfar/tool/exception.hpp"

namespace loicfar::ai
{
    template<algo ealgo, loss eloss>
    optim<ealgo, eloss>::optim(double eps) : eps_(eps) {}

    template<algo ealgo, loss eloss>
    std::unique_ptr<torch::optim::Optimizer> optim<ealgo, eloss>::make_optim(std::vector<torch::Tensor> p) const
    {
        switch (ealgo)
        {
        case algo::adadelta:
            break;
        case algo::adafactor:
            break;
        case algo::adagrad:
            break;
        case algo::adam:
            return std::make_unique<torch::optim::Adam>(p, eps_);
        case algo::adamax:
            break;
        case algo::adamw:
            break;
        case algo::asgd:
            break;
        case algo::lbfgs:
            break;
        case algo::nadam:
            break;
        case algo::radam:
            break;
        case algo::rmsprop:
            break;
        case algo::rprop:
            break;
        case algo::sgd:
            break;
        case algo::sparseadam:
            break;
        default:
            break;
        }
        return nullptr;
    }

    template<algo ealgo, loss eloss>
    std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&)> optim<ealgo, eloss>::make_loss() const
    {
        switch (eloss)
        {
        case loicfar::ai::loss::binary_cross_entropy: return [](const torch::Tensor& ref, const torch::Tensor& tgt) { return torch::binary_cross_entropy(ref, tgt); };
        case loicfar::ai::loss::binary_cross_entropy_with_logits: return [](const torch::Tensor& ref, const torch::Tensor& tgt) { return torch::binary_cross_entropy_with_logits(ref, tgt); };
        //case loicfar::ai::loss::cosine_embedding_loss: return [](const torch::Tensor& ref, const torch::Tensor& tgt) { return torch::cosine_embedding_loss(ref, tgt); };
        case loicfar::ai::loss::cross_entropy: return [](const torch::Tensor& ref, const torch::Tensor& tgt) { return torch::cross_entropy_loss(ref, tgt); };
        //case loicfar::ai::loss::ctc_loss: return [](const torch::Tensor& ref, const torch::Tensor& tgt) { return torch::ctc_loss(ref, tgt); };
        //case loicfar::ai::loss::gaussian_nll_loss: return [](const torch::Tensor& ref, const torch::Tensor& tgt) { return torch::gaussian_nll_loss(ref, tgt); };
        case loicfar::ai::loss::hinge_embedding_loss: return [](const torch::Tensor& ref, const torch::Tensor& tgt) { return torch::hinge_embedding_loss(ref, tgt); };
        case loicfar::ai::loss::huber_loss: return [](const torch::Tensor& ref, const torch::Tensor& tgt) { return torch::huber_loss(ref, tgt); };
        case loicfar::ai::loss::kl_div: return [](const torch::Tensor& ref, const torch::Tensor& tgt) { return torch::kl_div(ref, tgt); };
        case loicfar::ai::loss::l1_loss: return [](const torch::Tensor& ref, const torch::Tensor& tgt) { return torch::l1_loss(ref, tgt); };
        //case loicfar::ai::loss::margin_ranking_loss: return [](const torch::Tensor& ref, const torch::Tensor& tgt) { return torch::margin_ranking_loss(ref, tgt); };
        case loicfar::ai::loss::mse_loss: return [](const torch::Tensor& ref, const torch::Tensor& tgt) { return torch::mse_loss(ref, tgt); };
        case loicfar::ai::loss::multi_margin_loss: return [](const torch::Tensor& ref, const torch::Tensor& tgt) { return torch::multi_margin_loss(ref, tgt); };
        case loicfar::ai::loss::multilabel_margin_loss: return [](const torch::Tensor& ref, const torch::Tensor& tgt) { return torch::multilabel_margin_loss(ref, tgt); };
        //case loicfar::ai::loss::multilabel_soft_margin_loss: return [](const torch::Tensor& ref, const torch::Tensor& tgt) { return torch::multilabel_soft_margin_loss(ref, tgt); };
        case loicfar::ai::loss::nll_loss: return [](const torch::Tensor& ref, const torch::Tensor& tgt) { return torch::nll_loss(ref, tgt); };
        //case loicfar::ai::loss::poisson_nll_loss: return [](const torch::Tensor& ref, const torch::Tensor& tgt) { return torch::poisson_nll_loss(ref, tgt); };
        case loicfar::ai::loss::smooth_l1_loss: return [](const torch::Tensor& ref, const torch::Tensor& tgt) { return torch::smooth_l1_loss(ref, tgt); };
        case loicfar::ai::loss::soft_margin_loss: return [](const torch::Tensor& ref, const torch::Tensor& tgt) { return torch::soft_margin_loss(ref, tgt); };
        //case loicfar::ai::loss::triplet_margin_loss: return [](const torch::Tensor& ref, const torch::Tensor& tgt) { return torch::triplet_margin_loss(ref, tgt); };
        //case loicfar::ai::loss::triplet_margin_with_distance_loss: return [](const torch::Tensor& ref, const torch::Tensor& tgt) { return torch::triplet_margin_with_distance_loss(ref, tgt); };
        case loicfar::ai::loss::mape_loss: return [](const torch::Tensor& ref, const torch::Tensor& tgt) { return torch::mean(torch::abs((tgt - ref) / tgt), torch::kDouble); };
        default:
            break;
        }
        REPORT_CRITICAL("Loss function not supported.");
    }

    template struct optim<algo::adam, loss::binary_cross_entropy>;
    template struct optim<algo::adam, loss::cross_entropy>;
    template struct optim<algo::adam, loss::mse_loss>;
    template struct optim<algo::adam, loss::mape_loss>;
}