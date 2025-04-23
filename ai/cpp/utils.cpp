#include "loicfar/ai/utils.hpp"

#include "loicfar/math/matrix.hpp"
#include "loicfar/math/vector.hpp"

#include <torch/torch.h>
#include <vector>

namespace loicfar::ai
{
    template<typename T>
    torch::TensorOptions make_tensor_opt()
    {
        return torch::TensorOptions();
    }

    template<>
    torch::TensorOptions make_tensor_opt<double>()
    {
        return torch::TensorOptions().dtype(torch::kDouble);
    }

    template<typename T>
    torch::Tensor make_tensor(math::matrix<T>& m)
    {
        return torch::from_blob(m.data(), { m.rows(), m.cols()}, make_tensor_opt<T>()).cuda();
    }

    template<typename T>
    torch::Tensor make_tensor(math::vector<T>& v)
    {
        return torch::from_blob(v.data(), { v.size(), 1}, make_tensor_opt<T>()).cuda();
    }

    template<typename T>
    torch::Tensor make_tensor(std::vector<T>& v)
    {
        return torch::from_blob(v.data(), { (long long)(v.size()), 1}, make_tensor_opt<T>()).cuda();
    }

    template torch::Tensor make_tensor<double>(math::matrix<double>&);
    template torch::Tensor make_tensor<double>(math::vector<double>&);
    template torch::Tensor make_tensor<double>(std::vector<double>&);
}