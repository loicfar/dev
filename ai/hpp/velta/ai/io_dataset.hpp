#pragma once

#include "loicfar/ai/dataset.hpp"

namespace loicfar::ai
{
    class io_dataset : public dataset<io_dataset>
    {
    private:
        torch::Tensor input_;
        torch::Tensor output_;

    public:
        io_dataset(const std::string& path);
        virtual ~io_dataset();

    private:
        virtual void build_tensors() override;

    public:
        virtual torch::data::Example<> get(std::size_t index) override;
        virtual std::size_t size() const override;
    };
}