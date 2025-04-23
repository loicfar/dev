#include "loicfar/ai/nn.hpp"

#include "loicfar/ai/io_dataset.hpp"
#include "loicfar/ai/utils.hpp"
#include "loicfar/math/functions.hpp"
#include "loicfar/tool/exception.hpp"
#include "loicfar/tool/utils.hpp"

#include <iostream>

namespace loicfar::ai
{
    nn::nn(std::string w_file, const options& opt)
    {
        std::filesystem::path p = tool::utils::binaries_path();
        p.append("weights");
        std::filesystem::create_directory(p);
        p.append(std::move(w_file) + ".w.vlt");
        w_file_ = p.generic_string();

        module_ = std::make_shared<torch::nn::Module>();
        module_->to(torch::kDouble);
        module_->to(torch::kCUDA);

        loss_ = opt.make_loss();
    }

    nn::~nn() = default;

    void nn::train(i_dataset* /*data*/, int /*n_epoch*/)
    {
        REPORT_CRITICAL("Training not implemented for this model.");
    }

    bool nn::has_weights() const
    {
        return std::filesystem::exists(w_file_);
    }

    void nn::load()
    {
        if (!has_weights())
        {
            REPORT_CRITICAL("Can't find weights file:" + w_file_);
        }
        torch::load(module_, w_file_);
    }

    double nn::compute_impl(std::vector<double>&& p)
    {
        torch::Tensor t = make_tensor(p).transpose(0, 1);
        t = forward(t);
        return t.item<double>();
    }

    torch::Tensor nn::loss(const torch::Tensor& ref, const torch::Tensor& tgt) const
    {
        return loss_(ref, tgt);
    }

    torch::Tensor nn::optimize(torch::Tensor& data, torch::Tensor& tgt)
    {
        optim_->zero_grad();
        torch::Tensor pred = forward(data);
        torch::Tensor l = loss(pred, tgt);
        l.backward();
        optim_->step();

        return l;
    }

    void nn::save(std::size_t epoch, int period, bool print, double loss) const
    {
        if (epoch % period == 0)
        {
            if (print)
            {
                std::cout << "Epoch: " << epoch << " | Loss: " << loss << std::endl;
            }
            torch::save(module_, w_file_);
        }
    }

    template<typename T>
    void nn::train_impl(i_dataset* data, int n_epoch)
    {
        dataset<T>* ds = dynamic_cast<dataset<T>*>(data);

        if (!ds)
        {
            REPORT_CRITICAL("Wrong data set type for this model.");
        }

        std::size_t epoch = 0;
        double last_loss = 0.0;
        do
        {
            torch::Tensor loss(nullptr);

            for (auto& batch : ds->get_dataloader())
            {
                loss = optimize(batch.data, batch.target);
            }

            last_loss = loss.item<double>();
            save(epoch, n_epoch, true, last_loss);

            if (last_loss < math::epsilon<double>()) { break; }
            ++epoch;
        } while (epoch < static_cast<std::size_t>(n_epoch) || last_loss > math::epsilon<double>());
    }

    template void nn::train_impl<io_dataset>(i_dataset* data, int n_epoch);

    void nn::run()
    {
        torch::Tensor tensor = torch::rand({ 2, 3 });
        std::cout << tensor << std::endl;

        // Define a new Module.
        struct Net : torch::nn::Module
        {
            Net() {
                // Construct and register two Linear submodules.
                fc1 = register_module("fc1", torch::nn::Linear(784, 64));
                fc2 = register_module("fc2", torch::nn::Linear(64, 32));
                fc3 = register_module("fc3", torch::nn::Linear(32, 10));
            }

            // Implement the Net's algorithm.
            torch::Tensor forward(torch::Tensor x) {
                // Use one of many tensor manipulation functions.
                x = torch::relu(fc1->forward(x.reshape({ x.size(0), 784 })));
                x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
                x = torch::relu(fc2->forward(x));
                x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
                return x;
            }

            // Use one of many "standard library" modules.
            torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr };
        };

        // Create a new Net.
        auto net = std::make_shared<Net>();

        // Create a multi-threaded data loader for the MNIST dataset.
        auto data_loader = torch::data::make_data_loader(
            torch::data::datasets::MNIST("./data").map(
                torch::data::transforms::Stack<>()),
            /*batch_size=*/64);

        // Instantiate an SGD optimization algorithm to update our Net's parameters.
        torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

        for (std::size_t epoch = 1; epoch <= 10; ++epoch)
        {
            std::size_t batch_index = 0;
            // Iterate the data loader to yield batches from the dataset.
            for (auto& batch : *data_loader)
            {
                // Reset gradients.
                optimizer.zero_grad();
                // Execute the model on the input data.
                torch::Tensor prediction = net->forward(batch.data);
                // Compute a loss value to judge the prediction of our model.
                torch::Tensor loss = torch::nll_loss(prediction, batch.target);
                // Compute gradients of the loss w.r.t. the parameters of our model.
                loss.backward();
                // Update the parameters based on the calculated gradients.
                optimizer.step();
                // Output the loss and checkpoint every 100 batches.
                if (++batch_index % 100 == 0) {
                    std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                        << " | Loss: " << loss.item<float>() << std::endl;
                    // Serialize your model periodically as a checkpoint.
                    torch::save(net, "net.pt");
                }
            }
        }

    }
}