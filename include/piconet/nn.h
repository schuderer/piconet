#pragma once

#include <iostream>
#include <random>
#include <array>
#include <span>
#include <numeric>
#include <functional>
#include <ranges>
#include <random>

#include <picograd/value.h>

#if defined(VERBOSE_piconet) && !defined(NDEBUG)
#include <iostream>
#define FILENAME strrchr("/" __FILE__, '/') + 1
#define LOG(x) std::cout << FILENAME << " (" << __LINE__ << "): " << x << std::endl
#define LOGVAR(x) std::cout << FILENAME << " (" << __LINE__ << "): " << #x"=" << x << std::endl
#else
#define LOG(x)
#define LOGVAR(x)
#endif

namespace ajs {

// Generate uniform or normal distribution of type T
template<typename T>
class RandomDistribution {
public:
    // Create random distribution
    RandomDistribution(uint32_t seed_val=42, bool normal=false);

    // Sample a value of the random distribution
    double get();
private:
    std::mt19937 generator_;
    bool normal_;
    std::uniform_real_distribution<T> uni_dist_{-1.0, 1.0};
    std::normal_distribution<T> normal_dist_{0.0, 1.0};
};

// Options for nonlinear activations
enum class Activation {
    // I *would* put in the Layer class, but then the user would need to specify template params to use it.
    tanh, relu, sigmoid
};

// Layer of <NumOutput> neurons with <NumInputs> fully-connected inputs. Each neuron has a bias.
// Expects and uses scalar values wrapped in Value<T>.
template<typename T, size_t NumInputs, size_t NumOutputs>
class Layer {
public:
    // Create a fully connected layer using distribution `random` for initialization and the specified activation function `activation`
    Layer(RandomDistribution<T>& random, Activation activation=Activation::tanh);

    // Forward pass with `inputs`.
    std::array<Value<T>, NumOutputs> operator()(const std::span<Value<T>, NumInputs>& inputs);

    // Retrieve std::array of modifiable references to all weights and biases
    auto& get_parameters();

    // Pretty print information on the Layer weights and biases
    void print() const;
private:
    std::array<Value<T>, NumInputs * NumOutputs> weights_;
    std::array<Value<T>, NumOutputs> biases_;
    std::array<Value<T>, NumInputs * NumOutputs + NumOutputs> parameters_;  // view to values in weights_ and biases_
    Value<T> (Value<T>::*activation_func_ptr)() const;
};

// Ensure that values are probabilities (between 0 and 1, summing to 1),
// for inputs consisting of more than 1 value (for one single value, use sigmoid activation)
template<typename Container>
Container softmax(const Container& input);

// Calculate cross-entropy loss (negative log-likelihood) from probabilities
template<typename Container>  // TODO: use has_size/indexable trait?
typename Container::value_type cross_entropy(const Container& prediction, const Container& target);

// Calculate cross-entropy loss (negative log-likelihood) from logits ("log-transformed probabilities"
// - usually just raw output that can be negative; we just interpret the raw output as logits, which has
// advantages for scaling/numeric stability as well)
template<typename Container>
typename Container::value_type cross_entropy_with_logits(const Container& prediction, const Container& target);

// Calculate binary cross-entropy loss (negative log-likelihood) from probability Value<T> values.
// Use for single output values representing a binary value (passed throug sigmoid activation).
// This is an alternative to using two outputs and softmax for binary classification.
template<typename ValT>
ValT binary_cross_entropy(const ValT& prediction, const ValT& target);

//template<typename T>
//std::ostream& operator<<(std::ostream& os, const Value<T>& v);

/*
template<typename T, size_t NumInputs>
class Neuron {
public:
    Neuron() { }; // this is not good. TODO refactor so no dummy neurons need to be initialized at beginning of Layer construction.
    Neuron(RandomDistribution& random, Activation activation=Activation::tanh) {
        switch (activation) {  // maybe it would have been smarter to also make activation a template parameter?
        case Activation::tanh:
            act_func_ptr = &Value<T>::tanh;
            break;
        case Activation::relu:
            act_func_ptr = &Value<T>::relu;
            break;
        default:
            throw std::invalid_argument("Unsupported activation function");
            break;
        }
//        activation_ = activation;
        for (Value<T>& weight : weights_) {
            weight = random.get();
        }
        bias_ = Value<T>(random.get());
    };
    Value<T> operator()(const std::array<Value<T>, NumInputs>& inputs) const {
        Value<T> dot_product = std::inner_product(weights_.begin(), weights_.end(), inputs.begin(), bias_);
        LOGVAR(dot_product);
        Value<T> output = (dot_product.*act_func_ptr)();
        return output;
//        switch (activation_) {
//        case Activation::tanh:
//            return dot_product.tanh();
//            break;
//        case Activation::relu:
//            return dot_product.relu();
//            break;
//        default:
//            throw std::invalid_argument("Unsupported activation function");
//            break;
//        }
    }
    void print() const {
        std::cout << "weights:";
        for (const auto& weight : weights_) {
            std::cout << " " << weight;
        }
        std::cout << ", bias: " << bias_ << std::endl;
    }
private:
    std::array<Value<T>, NumInputs > weights_;
    Value<T> bias_;
//    Activation activation_;
    Value<T> (Value<T>::*act_func_ptr)() const;
};

template<typename T, size_t NumInputs, size_t NumNeurons>
class LayerOld {
public:
    LayerOld(RandomDistribution& random, Activation activation=Activation::tanh) {
        for (auto& neuron : neurons_) {
            neuron = Neuron<T, NumInputs>{random, activation};
        }
    }
    std::array<Value<T>, NumNeurons> operator()(const std::array<Value<T>, NumInputs>& inputs) {
        std::array<Value<T>, NumNeurons> outputs;
        for (size_t i=0; i < NumNeurons; ++i) {
            outputs[i] = neurons_[i](inputs);
        }
        return outputs;
    }
//    const std::array<Value<T>, NumNeurons>& get_outputs() const {
//        return outputs_;
//    }
    void print() {
        std::cout << "Layer (" << NumInputs << " inputs, " << NumNeurons << " neuron(s)):\n    ";
        for (auto& neuron : neurons_) {
            neuron.print();
            std::cout << "    ";
        }
        std::cout << std::endl;
    }
private:
    std::array<Neuron<T, NumInputs>, NumNeurons> neurons_;
};
*/


} // namespace ajs

// Needed because this is a template library
#include "nn.hpp"
