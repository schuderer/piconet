#include <iostream>
#include <array>
#include <span>
#include "piconet/nn.h"
#include "picograd/value.h"

using std::cout, std::endl;
using namespace ajs;

// Some temporary "manual" testing of backprop with Value<T> (no bias):

// First layer, neuron 1 of 2
auto w11_1 = Value(0.2);  // weight for incoming connection x1
auto w12_1 = Value(0.1);  // weight for incoming connection x2

// First layer, neuron 2 of 2
auto w21_1 = Value(0.01);
auto w22_1 = Value(0.4);

// Output layer
auto w11_2 = Value(0.15);  // weight for incoming connection o1_2
auto w12_2 = Value(0.8);  // weight for incoming connection o2_2


void init_net() {
    RandomUniformDistribution<double> rng;
    w11_1 = rng.get();
    w12_1 = rng.get();
    w21_1 = rng.get();
    w22_1 = rng.get();
    w11_2 = rng.get();
    w12_2 = rng.get();
}


void zero_grad() {
    w11_1.set_grad(0);
    w12_1.set_grad(0);
    w21_1.set_grad(0);
    w22_1.set_grad(0);
    w11_2.set_grad(0);
    w12_2.set_grad(0);
}

void learn(double alpha) {
    w11_1.set_data(w11_1.get_data() - w11_1.get_grad() * alpha);
    w12_1.set_data(w12_1.get_data() - w12_1.get_grad() * alpha);
    w21_1.set_data(w21_1.get_data() - w21_1.get_grad() * alpha);
    w22_1.set_data(w22_1.get_data() - w22_1.get_grad() * alpha);
    w11_2.set_data(w11_2.get_data() - w11_2.get_grad() * alpha);
    w12_2.set_data(w12_2.get_data() - w12_2.get_grad() * alpha);
}

template<typename T>
T net_forward(T x1, T x2, T y) {
//    auto x1 = 1.0;
//    auto x2 = 1.0;
//    auto y = 0.0;

    // Outputs of neurons of first layer
    auto o1_1_raw = w11_1 * x1 + w12_1 * x2;
    auto o2_1_raw = w21_1 * x1 + w22_1 * x2;
    auto o1_1 = o1_1_raw.tanh();
    auto o2_1 = o2_1_raw.tanh();

    // Outputs of neurons of second layer
    auto o3= (w11_2 * o1_1 + w12_2 * o2_1).sigmoid();

    // cross-entropy loss (negative log-likelihood)
    auto log_likelihood = (y * (o3 + 1.0E-15).log() + (-y + 1) * (-o3 + 1.0E-15 + 1).log())/2.0;
    auto loss = -log_likelihood;

//    auto loss1 = loss;
//    auto loss2 = binary_cross_entropy(o3, y);
//    LOGVAR(loss1);
//    LOGVAR(loss2);

    return loss;
}




int main() {

    LOG(" ----------- try out NN library ----------- ");

//{
//    RandomDistribution rng;
//    Layer<double, 2, 2> l1{rng};
//    Layer<double, 2, 2> l2{rng};
//    double alpha = 0.01;

//    auto input = std::array<Value<double>, 2>{1.0, 1.0};
//    auto target = std::array<Value<double>, 2>{1.0, 0.0};
//    for (int var = 0; var < 2; ++var) {
//        auto o1 = l1(input);
//        auto o2 = l2(o1);
//        auto smax = softmax(o2);
//        auto loss = cross_entropy(smax, target);
//        // or:
//        // auto loss = cross_entropy_with_logits(o2, target);
//        loss.backward();

//        for (auto& par : l1.get_parameters()) {
//            LOGVAR(par);
//            par.set_data(par.get_data() - par.get_grad() * alpha);
//            par.set_grad(0.0);
//        }
//        for (auto& par : l2.get_parameters()) {
//            LOGVAR(par);
//            par.set_data(par.get_data() - par.get_grad() * alpha);
//            par.set_grad(0.0);
//        }

//        cout << loss << endl;
//    }
//}

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

    //    auto input = std::array<Value<double>, 2>{1.0, 1.0};
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


    LOG(" ----------- try out hand-knitted NN calculations ----------- ");
    init_net();
    for (int epoch = 0; epoch < 10; ++epoch) {
        double loss_sum{0.0};
        for (auto& row : inputs) {
            auto [x1, x2, y] = row;
            // LOG("x1=" << x1.get_data() << ", x2=" << x2.get_data() << ", y=" << y.get_data());
            auto loss = net_forward(x1, x2, y);
            loss.backward();
            loss_sum += loss.get_data();
        }
        LOG(loss_sum / inputs.size());
        learn(alpha);
        zero_grad();
    }


    return 0;
}
