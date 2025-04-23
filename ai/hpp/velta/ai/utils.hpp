#pragma once

namespace loicfar::math
{
    template<typename>
    class matrix;
    template<typename>
    class vector;
}

namespace loicfar::ai
{
    template<typename T, typename ...Args>
    T make_module(Args&& ...args)
    {
        T m(std::forward<Args>(args)...);
        m->to(torch::kDouble);
        m->to(torch::kCUDA);
        return m;
    }

    template<typename T>
    torch::Tensor make_tensor(math::matrix<T>& m);

    template<typename T>
    torch::Tensor make_tensor(math::vector<T>& m);

    template<typename T>
    torch::Tensor make_tensor(std::vector<T>& m);
}