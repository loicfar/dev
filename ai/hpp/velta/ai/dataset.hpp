#pragma once

#include "loicfar/ai/i_dataset.hpp"
#include "loicfar/tool/table.hpp"

#include <torch/torch.h>

namespace loicfar::ai
{
    template<typename>
    class dataset_impl;

    template<typename T>
    class dataset : public i_dataset
    {
    protected:
        using data_loader_t = torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<dataset_impl<T>, torch::data::transforms::Stack<>>, torch::data::samplers::RandomSampler>;

        tool::table table_;

    private:
        std::unique_ptr<dataset_impl<T>> dataset_;
        std::unique_ptr<data_loader_t> dataloader_;

    public:
        dataset(const std::string& path);
        virtual ~dataset();

        torch::data::datasets::Dataset<dataset_impl<T>>& get_dataset();
        data_loader_t& get_dataloader();

    private:
        virtual void build_tensors() = 0;
    
    public:
        virtual void build() override final;
        virtual torch::data::Example<> get(std::size_t index) = 0;
        virtual std::size_t size() const = 0;
    };

    template<typename T>
    class dataset_impl final : public torch::data::datasets::Dataset<dataset_impl<T>>
    {
    public:
        dataset_impl(dataset<T>* ds);

    private:
        dataset<T>* dataset_;

    public:
        virtual torch::data::Example<> get(std::size_t index) override final;
        virtual std::optional<std::size_t> size() const override final;
    };
}