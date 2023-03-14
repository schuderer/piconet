#include "nn.h"

namespace ajs {

//////////////////////////////////////////////////////////////////////////////
/// class RandomDistribution<T>
//////////////////////////////////////////////////////////////////////////////

// Generate uniform or normal distribution of type T
template<typename T>
RandomDistribution<T>::RandomDistribution(uint32_t seed_val) {
    generator_.seed(seed_val);
}

// Sample a value of the uniform random distribution
template<std::floating_point T>
inline
T RandomUniformDistribution<T>::get() {
    return uni_dist_(this->generator_);
}

// Sample a value of the normal random distribution
template<typename T>
inline
T RandomNormalDistribution<T>::get() {
    return normal_dist_(this->generator_);
}


//////////////////////////////////////////////////////////////////////////////
/// class Layer<T, NumInputs, NumOutputs>
//////////////////////////////////////////////////////////////////////////////

// Create a fully connected layer using distribution `random` for initialization and the specified activation function `activation`
template<typename T, size_t NumInputs, size_t NumOutputs>
Layer<T, NumInputs, NumOutputs>::Layer(RandomDistribution<T>& random, Activation activation) {
    // LOG("creating layer " << NumInputs << "x" << NumOutputs);
    // LOGVAR(biases_.size());
    switch (activation) {  // maybe it would have been smarter to also make activation a template parameter?
    case Activation::tanh:
        activation_func_ptr = &Value<T>::tanh;
        break;
    case Activation::relu:
        activation_func_ptr = &Value<T>::relu;
        break;
    case Activation::sigmoid:
        activation_func_ptr = &Value<T>::sigmoid;
        break;
    default:
        throw std::invalid_argument("Unsupported activation function");
        break;
    }
    for (auto& weight : weights_) {
        weight = random.get();
    }
    for (auto& bias : biases_) {
        bias = random.get();
    }
    // Fill redundant array `parameters_` with references to all weights and biases, for easy access
    for (size_t i=0; i<parameters_.size(); ++i) {
        if (i < weights_.size()) {
            // LOG("weight " << i << ": " << weights_[i]);
            parameters_[i] = weights_[i];  // Calls Value copy constructor, new object points to same inner Node
        }
        else {
            // LOG("bias " << i - weights_.size() << ": " << biases_[i - weights_.size()]);
            parameters_[i] = biases_[i - weights_.size()];  // Calls Value copy constructor, new object points to same inner Node
        }
    }

}

// Forward pass with `inputs`.
// TODO: Allow inputs with other container types than std::array
template<typename T, size_t NumInputs, size_t NumOutputs>
std::array<Value<T>, NumOutputs> Layer<T, NumInputs, NumOutputs>::operator()(const std::span<Value<T>, NumInputs>& inputs) {
    std::array<Value<T>, NumOutputs> outputs;
    for (size_t neuron_idx=0; neuron_idx < NumOutputs; ++neuron_idx) {
        size_t weight_offset = neuron_idx * NumInputs;
        Value<T> dot_product = std::inner_product(weights_.begin() + weight_offset,
                                                  weights_.begin() + weight_offset + NumInputs,
                                                  inputs.begin(),
                                                  biases_[neuron_idx]);
        Value<T> output = (dot_product.*activation_func_ptr)();
        outputs[neuron_idx] = output;
    }
    return outputs;
}

// Retrieve modifiable references to all weights and biases
template<typename T, size_t NumInputs, size_t NumOutputs>
inline
auto& Layer<T, NumInputs, NumOutputs>::get_parameters() {
    // TODO: if they are non-const references, i.e. accessible from outside, why not just make parameters_ public?
    return parameters_;
}

// Pretty print information on the Layer weights and biases
template<typename T, size_t NumInputs, size_t NumOutputs>
void Layer<T, NumInputs, NumOutputs>::print() const {
    std::cout << "Layer (" << NumInputs << " inputs, " << NumOutputs << " neuron(s)):\n    ";
    for (size_t neuron_idx=0; neuron_idx < NumOutputs; ++neuron_idx) {
        std::cout << "    weights:";
        for (size_t input_idx=0; input_idx < NumInputs; ++input_idx) {
            std::cout << " " << weights_[neuron_idx * NumInputs + input_idx];
        }
        std::cout << ", bias: " << biases_[neuron_idx] << std::endl;
    }
    std::cout << std::endl;
}


//////////////////////////////////////////////////////////////////////////////
/// Independent functions
//////////////////////////////////////////////////////////////////////////////
// TODO should those go into a .cpp file? Worth the hassle with template classes?

// Ensure that values are probabilities (between 0 and 1, summing to 1)
template<typename Container>  // TODO: use has_size/indexable trait instead?
Container softmax(const Container& input) {
    using ValT = typename Container::value_type;
    Container calculated{};
    ValT sum_exps{0};
    for (int i = 0; i < input.size(); ++i) {
//        LOGVAR(i);
       auto curr_input = input[i];
       auto curr_exp = curr_input.exp();
        calculated[i] = curr_exp;
       sum_exps += curr_exp;
    }
//    for (auto& c : calculated) {
//        LOGVAR(c);
//    }
    for (auto& curr_exp : calculated) {
//        LOGVAR(curr_exp);
        curr_exp /= sum_exps;
    }
//    for (auto& c : calculated) {
//        LOGVAR(c);
//    }
    return calculated;
}


// Calculate cross-entropy loss (negative log-likelihood) from probabilities.
// Scales the values with N
template<typename Container>  // TODO: use has_size/indexable trait?
typename Container::value_type cross_entropy(const Container& prediction, const Container& target) {
//    assert(prediction.size() == target.size() + 1 && "cross_entropy expects containers of equal length");  // not needed -- container type ensures this
    using ValT = typename Container::value_type;
    ValT result{0};
    for (size_t i=0; i<prediction.size(); ++i) {
        result += target[i] * prediction[i].log();
    }
    return -result/target.size();
}

// Calculate cross-entropy loss (negative log-likelihood) from logits ("log-transformed probabilities"
// - usually just raw output that can be negative; we just interpret the raw output as logits, which has
// advantages for scaling/numeric stability as well)
// Scales the values with N
template<typename Container>  // TODO: use has_size/indexable trait?
typename Container::value_type cross_entropy_with_logits(const Container& prediction, const Container& target) {
    using ValT = typename Container::value_type;
//    assert(prediction.size() == target.size() && "cross_entropy_with_logits expects containers of equal length");

    // Prepare sum of exponents
    ValT sum_exps{0};
    for (auto& curr_pred : prediction) {
        sum_exps += curr_pred.exp();
    }

    // Calculate loss (saving a some log() calculations)
    using ValT = typename Container::value_type;
    ValT result{0};
    for (size_t i=0; i<prediction.size(); ++i) {
        result += target[i] * (prediction[i] - sum_exps.log()) ;
//           result += target[i] * (prediction[i].exp() / sum_exps).log();
    }
return -result/target.size();
}

// Calculate binary cross-entropy loss (negative log-likelihood) from probability
template<typename ValT>
ValT binary_cross_entropy(const ValT& prediction, const ValT& target) {
    auto neg_pred = -prediction + 1;
    auto neg_targ = -target + 1;
//    LOGVAR(neg_pred);
//    LOGVAR(neg_targ);
    std::array<ValT, 2> pred_arr{prediction, neg_pred};
    std::array<ValT, 2> targ_arr{target, neg_targ};
    return cross_entropy(pred_arr, targ_arr);
}

//template<typename T>
//std::ostream& operator<<(std::ostream& os, const Value<T>& v) {
//    auto node = v.get_node();
//    if (node != nullptr) {
//        return (os << "Value(" << v.get_data() << ",grad=" << v.get_grad() << ",node=" << v.get_node() << ")@" << &v);
//    }
//    else {
//        return (os << "Value(?,grad=?,node=nullptr)@" << &v);
//    }
//}

} // namespace ajs
