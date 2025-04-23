#include "loicfar/ai/io_dataset.hpp"
#include "loicfar/ai/utils.hpp"
#include "loicfar/math/matrix.hpp"
#include "loicfar/math/vector.hpp"

namespace loicfar::ai
{
    io_dataset::io_dataset(const std::string& path) : dataset(path) {}

    io_dataset::~io_dataset() = default;

    void io_dataset::build_tensors()
    {
        const long long n = table_.rows();
        math::matrix<double> input(n, table_.cols() - 1);
        math::vector<double> output(n);

        std::vector<double> line;

        std::size_t i = 0;
        while (table_.read_line(line))
        {
            for (std::size_t j = 0; j < static_cast<std::size_t>(input.cols()); ++j)
            {
                input(i, j) = line.at(j);
            }
            output[i] = line.back();
            ++i;
        }

        input_ = make_tensor(input);
        output_ = make_tensor(output);
    }

    torch::data::Example<> io_dataset::get(std::size_t index)
    {
        return { input_[static_cast<int64_t>(index)], output_[static_cast<int64_t>(index)] };
    }

    std::size_t io_dataset::size() const
    {
        return input_.size(0);
    }
}