#include "loicfar/ai/factory.hpp"

#include "loicfar/ai/autoencoder.hpp"
#include "loicfar/ai/dqn.hpp"
#include "loicfar/ai/io_dataset.hpp"
#include "loicfar/ai/mlp.hpp"

namespace loicfar::ai
{
    std::unique_ptr<i_nn> make_autoencoder(std::string w_file, std::size_t n, int n_nodes, const options& opt)
    {
        return std::make_unique<autoencoder>(std::move(w_file), n, n_nodes, opt);
    }

    std::unique_ptr<i_nn> make_dqn(std::string w_file, std::size_t n_obs, std::size_t n_actions, int n_nodes, const options& opt)
    {
        return std::make_unique<dqn>(std::move(w_file), n_obs, n_actions, n_nodes, opt);
    }

    std::unique_ptr<i_nn> make_mlp(std::string w_file, std::size_t n_obs, int n1, int n2, int n_out, const options& opt)
    {
        return std::make_unique<mlp>(std::move(w_file), n_obs, n1, n2, n_out, opt);
    }

    std::unique_ptr<i_dataset> make_io_dataset(const std::string& path)
    {
        return std::make_unique<io_dataset>(path);
    }
}