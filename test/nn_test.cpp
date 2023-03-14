#include <iostream>
#include <numeric>          // std::accumulate
#include <algorithm>        // std::sort
#include "gtest/gtest.h"
#include "piconet/nn.h"

using std::cout, std::endl;
using namespace ajs;


template<typename DistTy>
std::array<int, 4> random_dist_counts_helper(int num_samples=1000) {
    DistTy rng;
    int num_lt_minus_one = 0;
    int num_lt_zero = 0;
    int num_gt_zero = 0;
    int num_gt_one = 0;
    for (int i=0; i<num_samples; ++i) {
        float num = rng.get();
        num_lt_minus_one += num < -1;
        num_lt_zero += num < 0;
        num_gt_zero += num > 0;
        num_gt_one += num > 1;
    }
//    LOGVAR(num_lt_minus_one);
//    LOGVAR(num_lt_zero);
//    LOGVAR(num_gt_zero);
//    LOGVAR(num_gt_one);

    return {num_lt_minus_one, num_lt_zero, num_gt_zero, num_gt_one};
}

TEST(RandomDist, UniformFloat) {
    const int num_samples = 1000;
    auto [  num_lt_minus_one,
            num_lt_zero,
            num_gt_zero,
            num_gt_one] = random_dist_counts_helper<RandomUniformDistribution<float>>(num_samples);
    EXPECT_EQ(num_lt_minus_one, 0);
    EXPECT_GT(num_lt_zero, num_samples/2 * 0.8);
    EXPECT_GT(num_gt_zero, num_samples/2 * 0.8);
    EXPECT_EQ(num_gt_one, 0);
}

TEST(RandomDist, UniformDouble) {
    const int num_samples = 1000;
    auto [  num_lt_minus_one,
            num_lt_zero,
            num_gt_zero,
            num_gt_one] = random_dist_counts_helper<RandomUniformDistribution<double>>(num_samples);
    EXPECT_EQ(num_lt_minus_one, 0);
    EXPECT_GT(num_lt_zero, num_samples/2 * 0.8);
    EXPECT_GT(num_gt_zero, num_samples/2 * 0.8);
    EXPECT_EQ(num_gt_one, 0);
}

TEST(RandomDist, NormalDouble) {
    const int num_samples = 1000;
    auto [  num_lt_minus_one,
            num_lt_zero,
            num_gt_zero,
            num_gt_one] = random_dist_counts_helper<RandomNormalDistribution<double>>(num_samples);
    RandomNormalDistribution<double> rng;
    EXPECT_GT(num_lt_minus_one, 0.158 * 0.8);
    EXPECT_GT(num_lt_zero, num_samples/2 * 0.8);
    EXPECT_GT(num_gt_zero, num_samples/2 * 0.8);
    EXPECT_GT(num_gt_one, (1.0 - 0.158) * 0.8);
}


template<typename T, int ValueToReturn>
class MockRandomDistribution: public RandomDistribution<T> {
    // TODO: proper mocking
    // - https://stackoverflow.com/questions/5777733/mock-non-virtual-method-c-gmock
    // - https://google.github.io/googletest/gmock_for_dummies.html
    T get() {
        return ValueToReturn;
    }
};

TEST(Layer, Initialize) {
    auto input_float = std::array{Value{1.0f}};
    auto input_double = std::array{Value{1.0}};

    MockRandomDistribution<float, 1> rng1;
    Layer<float, 1, 1> init1{rng1, Activation::relu};
    auto result = init1(input_float);
    EXPECT_FLOAT_EQ(result[0].get_data(), 2.0f);
    EXPECT_EQ(init1.get_parameters().size(), 2);  // 1 input + 1 bias
    EXPECT_FLOAT_EQ(init1.get_parameters()[0].get_data(), 1.0f);
    EXPECT_FLOAT_EQ(init1.get_parameters()[1].get_data(), 1.0f);

    MockRandomDistribution<double, 0> rng0;
    Layer<double, 1, 1> init0{rng0, Activation::relu};
    EXPECT_FLOAT_EQ(init0(input_double)[0].get_data(), 0.0f);
    EXPECT_FLOAT_EQ(init0.get_parameters()[0].get_data(), 0.0f);
    EXPECT_FLOAT_EQ(init0.get_parameters()[1].get_data(), 0.0f);
}

TEST(Layer, TwoDimForward) {
    MockRandomDistribution<float, 1> rng;
    auto input = std::array{Value{0.5f}, Value{0.5f}};

    Layer<float, 2, 2> relu_layer{rng};
    auto lin_out = relu_layer(input);
    EXPECT_EQ(lin_out.size(), 2);
}

TEST(Layer, ActivationReLU) {
    MockRandomDistribution<float, 1> rng;
    auto input = std::array{Value{0.5f}, Value{0.5f}};

    Layer<float, 2, 2> relu_layer{rng, Activation::relu};
    auto lin_out = relu_layer(input);
    EXPECT_FLOAT_EQ(lin_out[0].get_data(), (input[0] * 1.0f + input[1] * 1.0f + 1.0f).get_data());
    EXPECT_FLOAT_EQ(lin_out[1].get_data(), lin_out[0].get_data());

    auto input_cutoff = std::array{Value{-1.5f}, Value{-1.5f}};
    auto lin_out_cutoff = relu_layer(input_cutoff);
    EXPECT_FLOAT_EQ(lin_out_cutoff[0].get_data(), 0.0f);
    EXPECT_FLOAT_EQ(lin_out_cutoff[1].get_data(), 0.0f);

    auto input_cutoff_edge = std::array{Value{-0.5f}, Value{-0.5f}};  // bias offsets by 1: 1.0 - 0.5 - 0.5 = 0.0
    auto lin_out_cutoff_edge = relu_layer(input_cutoff_edge);
    EXPECT_FLOAT_EQ(lin_out_cutoff_edge[0].get_data(), 0.0f);
    EXPECT_FLOAT_EQ(lin_out_cutoff_edge[1].get_data(), 0.0f);
}


TEST(Layer, Activations) {
    MockRandomDistribution<float, 1> rng;
    auto input = std::array{Value{0.5f}, Value{0.5f}};

    // Create reference values
    Layer<float, 2, 2> relu_layer{rng, Activation::relu};
    auto lin_out = relu_layer(input);
    auto tanh_out_ref = std::array{lin_out[0].tanh(), lin_out[1].tanh()};
    auto sigm_out_ref = std::array{lin_out[0].sigmoid(), lin_out[1].sigmoid()};

    Layer<float, 2, 2> tanh_layer{rng, Activation::tanh};
    auto tanh_out = tanh_layer(input);
    EXPECT_FLOAT_EQ(tanh_out[0].get_data(), tanh_out_ref[0].get_data());
    EXPECT_FLOAT_EQ(tanh_out[1].get_data(), tanh_out_ref[1].get_data());

    Layer<float, 2, 2> sigm_layer{rng, Activation::sigmoid};
    auto sigm_out = sigm_layer(input);
    EXPECT_FLOAT_EQ(sigm_out[0].get_data(), sigm_out_ref[0].get_data());
    EXPECT_FLOAT_EQ(sigm_out[1].get_data(), sigm_out_ref[1].get_data());
}


TEST(Helpers, Softmax) {
    auto input_one = std::array{Value(1.0f)};
    auto input_half = std::array{Value(0.5f)};

    // Softmax of single-element container should always be one
    EXPECT_FLOAT_EQ(softmax(input_one)[0].get_data(), 1.0f);
    EXPECT_FLOAT_EQ(softmax(input_half)[0].get_data(), 1.0f);

    auto input = std::array{Value(1.0f), Value(7.0f), Value(-1.0f)};
    std::sort(input.begin(), input.end());  // for clarity: inputs need to be sorted for a later check

    // Softmax should some up to one
    auto result = softmax(input);
    auto sum = std::accumulate(result.begin(), result.end(), Value(0.0f));
    EXPECT_FLOAT_EQ(sum.get_data(), 1.0f);

    // Results are in same order (they have only been scaled through exp() and division, but not swapped, negated, etc.)
    EXPECT_TRUE(std::is_sorted(result.begin(), result.end()));
}


TEST(Helpers, CrossEntropy) {
    auto input_one = std::array{Value(1.0f)};
    auto input_zero = std::array{Value(0.0f)};

    // CrossEntropy of single-element container (so, wrongly applied) is always zero for negative class, and log(pred) vor positive class
    EXPECT_FLOAT_EQ(cross_entropy(input_one, input_zero).get_data(), 0.0f);
    EXPECT_FLOAT_EQ(cross_entropy(input_one, input_one).get_data(), std::log(1.0f));

    // Check calculation
    // std::exp(0.5) == 1.64872
    auto some_inputs = std::array{Value(1.64872), Value(1.0), Value(1.0)};  // = log(0.5), log(0), log(0)
    auto some_ones = std::array{Value(1.0), Value(1.0), Value(1.0)};
    auto result = cross_entropy(some_inputs, some_ones);
    EXPECT_NEAR(result.get_data(), -(1.0 * 0.5 + 1.0 * 0.0 + 1.0 * 0.0)/3.0, 0.0001);
}


TEST(Helpers, CrossEntropyLogits) {
    auto pred = std::array{Value(0.8f), Value(-2.1f), Value(5.03f)};
    auto target = std::array{Value(1.0f), Value(0.0f), Value(0.0f)};
    auto pred_sigm = softmax(pred);

    auto reference = cross_entropy(pred_sigm, target);
    auto under_test = cross_entropy_with_logits(pred, target);

    EXPECT_FLOAT_EQ(under_test.get_data(), reference.get_data());
}


TEST(Helpers, BinaryCrossEntropy) {
    auto pred_bin = Value(0.8f);
    auto target_bin = Value(1.0f);
    auto pred_pair = std::array{Value(0.8f), Value(0.2f)};  // sorted desc on purpose
    auto target_pair = std::array{Value(1.0f), Value(0.0f)};  // sorted desc on purpose

    auto reference = cross_entropy(pred_pair, target_pair);
    auto under_test = binary_cross_entropy(pred_bin, target_bin);

    EXPECT_FLOAT_EQ(under_test.get_data(), reference.get_data());

    // Test it again the other way round
    std::sort(pred_pair.begin(), pred_pair.end());
    std::sort(target_pair.begin(), target_pair.end());
    pred_bin = Value(0.2f);
    target_bin = Value(0.0f);

    reference = cross_entropy(pred_pair, target_pair);
    under_test = binary_cross_entropy(pred_bin, target_bin);

    EXPECT_FLOAT_EQ(under_test.get_data(), reference.get_data());
}

