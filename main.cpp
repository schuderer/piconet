#include <iostream>
#include <array>
#include <span>
#include "piconet/nn.h"
#include "picograd/value.h"

using std::cout, std::endl;
using namespace ajs;

int main() {

    LOG(" ----------- try out NN library ----------- ");

    const size_t input_len = 2;

    RandomUniformDistribution<double> rng;
    Layer<double, input_len, 2> l1{rng};
    Layer<double, 2, 1> l2{rng, Activation::sigmoid};
    double alpha = 0.1;

    auto inputs = std::array<std::array<Value<double>, input_len + 1>, 4> {{
        {0, 0, 0},
        {1, 0, 1},
        {0, 1, 1},
        {1, 1, 0}
    }};

    for (int var = 0; var < 10; ++var) {
        double loss_sum{0.0};
        for (std::span input_and_y : inputs) {
            std::span<Value<double>, 2> input = input_and_y.subspan<0, 2>();
            auto y = input_and_y[input_len];
            // LOG("x1=" << input[0].get_data() << ", x2=" << input[1].get_data() << ", y=" << y.get_data());

            auto o1 = l1(input);
            auto o2 = l2(o1)[0];
            auto loss = binary_cross_entropy(o2, y);
//            auto o2 = l2(o1);
//            auto loss = cross_entropy_with_logits(o2, {y, -y+1});
            loss.backward();

            loss_sum += loss.get_data();
        }
        LOG(loss_sum / inputs.size());
        // learn and zero gradient
        for (auto& par : l1.get_parameters()) {
            par.set_data(par.get_data() - par.get_grad() * alpha);
            par.set_grad(0.0);
        }
        for (auto& par : l2.get_parameters()) {
            par.set_data(par.get_data() - par.get_grad() * alpha);
            par.set_grad(0.0);
        }
    }

    return 0;
}
