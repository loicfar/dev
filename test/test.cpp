#include "unit_tests/utils/test_data.hpp"
#include "velta/ai/factory.hpp"
#include "velta/ai/options.hpp"
#include "velta/math/functions.hpp"

namespace velta::unit_tests
{
    using namespace ai;

    VTEST(ai, pima_indians_diabetes, test_type::none, debug_mode::run)
    {
        const optim<algo::adam, loss::binary_cross_entropy> opt(0.001);

        std::unique_ptr<i_nn> mlp = ai::make_mlp(PIMA_INDIANS_DIABETES, 8, 64, 256, 1, opt);

        if (TRAINING_MODE)
        {
            std::unique_ptr<i_dataset> ds = ai::make_io_dataset(PIMA_INDIANS_DIABETES_DATASET);
            ds->build();
            mlp->train(ds.get(), 10);
        }

        if (mlp->has_weights())
        {
            mlp->load();

            EXPECT_NEAR(mlp->compute(6, 148, 72, 35, 0, 33.6, 0.627, 50), 1.0, 1e-10);
            EXPECT_NEAR(mlp->compute(1, 85, 66, 29, 0, 26.6, 0.351, 31), 0.0, 1e-10);
        }
        else
        {
            EXPECT_ANY_THROW(mlp->load());
        }
    }

    VTEST(ai, bs_option, test_type::none, debug_mode::run)
    {
        const optim<algo::adam, loss::mape_loss> opt(0.001);

        std::unique_ptr<i_nn> mlp = ai::make_mlp(OPTION_DATA, 4, 64, 256, 1, opt);

        if (TRAINING_MODE)
        {
            std::unique_ptr<i_dataset> ds = ai::make_io_dataset(OPTION_DATA_DATASET);
            ds->build();
            mlp->train(ds.get(), 1);
        }

        if (mlp->has_weights())
        {
            mlp->load();

            EXPECT_NEAR(mlp->compute(10, 14.65379877, 0.1, 0.35), 1.0, 1e-10);
        }
        else
        {
            EXPECT_ANY_THROW(mlp->load());
        }
    }
}