#pragma once

#include <iostream>
#include <random>
#include <array>
#include <span>
#include <numeric>
#include <functional>
#include <ranges>
#include <random>
#include <concepts>

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

// Base class for random distributions
template<typename T>
class RandomDistribution {
public:
    // Create random distribution
    RandomDistribution(uint32_t seed_val=42);

    // Sample a value of the random distribution
    virtual T get() = 0;
protected:
    std::mt19937 generator_;
};

// Generate uniform distribution of type T
template<std::floating_point T>
class RandomUniformDistribution: public RandomDistribution<T> {
public:
    // Inherit all constructors
    using RandomDistribution<T>::RandomDistribution;

    // Sample a value of the random distribution
    T get() override;
private:
    std::uniform_real_distribution<T> uni_dist_{-1.0, 1.0};
};

// Generate normal distribution of type T
template<typename T>
class RandomNormalDistribution: public RandomDistribution<T> {
public:
    // Inherit all constructors
    using RandomDistribution<T>::RandomDistribution;

    // Sample a value of the random distribution
    T get() override;
private:
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
    Value<T> (Value<T>::*activation_func_ptr)() const {nullptr};
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

} // namespace ajs

// Needed because this is a template library
#include "nn.hpp"
