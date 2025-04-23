#pragma once

#include "loicfar/ai/ai_api.hpp"
#include "loicfar/ai/i_nn.hpp"
#include "loicfar/ai/i_dataset.hpp"
#include "loicfar/ai/options.hpp"

#include <memory>

namespace loicfar::ai
{
    std::unique_ptr<i_nn> AI_API make_autoencoder(std::string w_file, std::size_t n, int n_nodes, const options& opt);
    std::unique_ptr<i_nn> AI_API make_dqn(std::string w_file, std::size_t n_obs, std::size_t n_actions, int n_nodes, const options& opt);
    std::unique_ptr<i_nn> AI_API make_mlp(std::string w_file, std::size_t n_obs, int n1, int n2, int n_out, const options& opt);

    std::unique_ptr<i_dataset> AI_API make_io_dataset(const std::string& path);
}