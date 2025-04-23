#pragma once

#include "loicfar/ai/ai_api.hpp"
#include "loicfar/ai/i_dataset.hpp"

#include <memory>
#include <vector>

namespace loicfar::ai
{
    class AI_API i_nn
    {
    public:
        virtual ~i_nn();

        virtual void train(i_dataset* data, int n_epoch) = 0;


        virtual bool has_weights() const = 0;
        virtual void load() = 0;

        template<typename ...Args>
        double compute(Args&& ...args)
        {
            return compute_impl({ std::forward<double>(args)... });
        }

    private:
        virtual double compute_impl(std::vector<double>&& p) = 0;
    };
}