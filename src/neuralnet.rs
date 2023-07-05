use ndarray::*;
use rand::Rng;

#[derive(Debug)]
struct Layer {
    len: usize,                  // number of neurons in layer
    incoming: usize,             // number of neurons in prev layer
    activations: Array1<f32>,    // activations of current layer * PRE SIGMOID *
    output: Array1<f32>,         // activations of current layer * POST SIGMOID *
    errors: Array1<f32>,       // error terms for each neuron in layer
    weights: Array2<f32>,      // weights for previous layer
    weights_grad: Array2<f32>, // partial derivatives for error with respect to each weight
    bias_weights: Array1<f32>, // weights for constant 1 bias
}

pub struct NeuralNet {
    size: usize,                 // number of layers
    layers: Vec<Layer>,
    learning_rate: f32
}

impl NeuralNet {
    /* randomize weights for each layer in nn */
    pub fn seed_weights(&mut self) {
        let mut rng = rand::thread_rng();
        for layer in self.layers.iter_mut() {
            for i in 0..layer.len {
                layer.bias_weights[i] = 0.0;
                for j in 0..layer.incoming {
                    layer.weights[[i, j]] = rng.gen_range(-1.0..1.0);
                }
            }
        }
    }

    /* load specified layer's outputs with values of input */
    pub fn load_neurons(&mut self, layer_idx: usize, input: Array1<f32>) {
        if layer_idx >= self.size {
            eprintln!("Invalid layer index. Exiting.");
            std::process::exit(-1);
        }
        let layer = &mut self.layers[layer_idx];
        if layer.len != input.len() {
            eprintln!("Invalid length for neuron load. Exiting.");
            std::process::exit(-1);
        }
        layer.activations = input;
        layer.output = layer.activations.clone();
    }

    /* propagate network's values forward */
    pub fn feed_forward(&mut self) {
        for layer_idx in 1..self.size {
            let len = &self.layers[layer_idx].len;
            for neuron_idx in 0..*len {
                let input = &self.layers[layer_idx-1].output;
                let weights = &self.layers[layer_idx].weights.row(neuron_idx);
                let mut prod = weights.dot(input);
                prod += 1.0 * &self.layers[layer_idx].bias_weights[neuron_idx];
                self.layers[layer_idx].activations[neuron_idx] = prod;
                prod = sigmoid(prod);
                self.layers[layer_idx].output[neuron_idx] = prod;
            }
        }
    }

    /* print layer of network */
    pub fn print_output(&self, layer_idx: usize) {
        if layer_idx >= self.size {
            eprintln!("Invalid layer index. Exiting.");
            std::process::exit(-1); 
        }
        for value in &self.layers[layer_idx].output {
            print!("{:.3}, ", value);
        }
        println!();
    }

    /* return mse given expected output values */
    pub fn mse(&self, expected: &Array1<f32>) -> f32 {
        if expected.len() != self.layers[self.size-1].output.len() {
            eprintln!("Expected vector does not match output layer size. Exiting.");
            std::process::exit(-1); 
        }
        let out = &self.layers[self.size-1].output;
        let mut err: f32 = 0.0;
        let mut cnt = 0.0;
        for i in 0..out.len() {
            cnt += 1.0;
            err += (expected[i] - out[i]).powf(2.0);
        }
        return err / cnt;
    }

    /* given expected output layer, update weights according to error function  */
    pub fn backprop(&mut self, expected: &Array1<f32>) {
        if expected.len() != self.layers[self.size-1].activations.len() {
            eprintln!("Expected vector does not match output layer size. Exiting.");
            std::process::exit(-1); 
        }

        // output layer backwards phase
        let layers = &mut self.layers;
        let output_layer = &mut layers[self.size-1];
        for idx in 0..output_layer.len {
            // calculate error term for output layer
            let mut err = output_layer.output[idx] - expected[idx];
            err *= sigmoid_prime(output_layer.activations[idx]);
            output_layer.errors[idx] = err;
        }

        // propagate error terms for previous layers
        for layer_idx in (1..self.size-1).rev() {
            for idx in 0..layers[layer_idx].len {
                let weights = &layers[layer_idx+1].weights.column(idx);
                let errors = &layers[layer_idx+1].errors;
                let err = weights.dot(errors) * sigmoid_prime(layers[layer_idx].activations[idx]);
                layers[layer_idx].errors[idx] = err;
            }
        }

        // update each weight with partial deriv
        for layer_idx in (1..self.size).rev() {
            for neuron_idx in 0..layers[layer_idx].len {
                for weight_idx in 0..layers[layer_idx].incoming {
                    let err = layers[layer_idx].errors[neuron_idx];
                    let weight = layers[layer_idx-1].output[weight_idx];
                    let grad = err * weight;
                    layers[layer_idx].weights[[neuron_idx, weight_idx]] += grad * -self.learning_rate;
                }
            }
        }
        
        // update biases for each node
        for layer_idx in (1..self.size).rev() {
            for neuron_idx in 0..layers[layer_idx].len {
                for bias_weight_idx in 0..layers[layer_idx].bias_weights.len() {
                    let err = layers[layer_idx].errors[neuron_idx];
                    let weight = layers[layer_idx].bias_weights[bias_weight_idx];
                    let grad = err * weight;
                    layers[layer_idx].bias_weights[bias_weight_idx] += grad * -self.learning_rate;
                }
            }
        }
    }

    pub fn backprop_update(&mut self, num_iter: f32) {
        let layers = &mut self.layers;
        for layer_idx in 1..self.size {
            for idx in 0..layers[layer_idx].len {
                for jdx in 0..layers[layer_idx].incoming {
                    let prod = layers[layer_idx].weights_grad[[idx, jdx]] / num_iter;
                    layers[layer_idx].weights[[idx, jdx]] = prod * self.learning_rate;
                    layers[layer_idx].weights_grad[[idx, jdx]] = 0.0;
                }
            }
        }
    }

    pub fn get_output(&self) -> &Array1<f32> {
        return &self.layers[self.size-1].output;
    }
}

pub fn init_nn(lengths: Vec<usize>) -> NeuralNet {
    if lengths.len() < 2 {
        eprintln!("Invalid neural net size. Exiting.");
        std::process::exit(-1);
    }
    let mut net = NeuralNet { size: lengths.len(), layers: std::vec!{}, learning_rate: 0.5 };
    for i in 0..lengths.len() {
        let mut prev = 0;
        if i != 0 {
            prev = lengths[i-1]; 
        }
        let layer = Layer {
            len: lengths[i],
            incoming: prev,
            activations: Array1::zeros(lengths[i]),
            output: Array1::zeros(lengths[i]),
            errors: Array1::zeros(lengths[i]),
            weights: Array2::zeros((lengths[i], prev)),
            weights_grad: Array2::zeros((lengths[i], prev)),
            bias_weights: Array1::zeros(lengths[i]),
        };
        net.layers.push(layer);
    }
    return net;
}

fn sigmoid(val: f32) -> f32 {
    return 1.0 / (1.0 + (-val).exp());
}

fn sigmoid_prime(val: f32) -> f32 {
    let sig = sigmoid(val);
    return sig * (1.0 - sig);
}