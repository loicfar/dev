#pragma once

#include "loicfar/ai/ai_api.hpp"

namespace loicfar::ai
{
    class AI_API i_dataset
    {
    public:
        virtual ~i_dataset();

        virtual void build() = 0;
        virtual std::size_t size() const = 0;
    };
}