#include "loicfar/ai/dataset.hpp"

#include "loicfar/ai/io_dataset.hpp"

namespace loicfar::ai
{
    template<typename T>
    dataset<T>::dataset(const std::string& path) : table_(path)
    {
        dataset_ = std::make_unique<dataset_impl<T>>(this);
    }
    
    template<typename T>
    dataset<T>::~dataset() = default;

    template<typename T>
    torch::data::datasets::Dataset<dataset_impl<T>>& dataset<T>::get_dataset()
    {
        return *this->dataset_.get();
    }

    template<typename T>
    dataset<T>::data_loader_t& dataset<T>::get_dataloader()
    {
        return *this->dataloader_.get();
    }

    template<typename T>
    void dataset<T>::build()
    {
        this->build_tensors();
        this->dataloader_ = torch::data::make_data_loader(get_dataset().map(torch::data::transforms::Stack<>()), 1024);
    }

    template<typename T>
    dataset_impl<T>::dataset_impl(dataset<T>* ds) : dataset_(ds) {}

    template<typename T>
    torch::data::Example<> dataset_impl<T>::get(std::size_t index)
    {
        return this->dataset_->get(index);
    }

    template<typename T>
    std::optional<std::size_t> dataset_impl<T>::size() const
    {
        return this->dataset_->size();
    }

    template class dataset<io_dataset>;
}