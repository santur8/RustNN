use ndarray::*;
use rand::Rng;

#[derive(Debug)]
struct Layer {
    len: usize,                  // number of neurons in layer
    incoming: usize,             // number of neurons in prev layer
    neurons: Array1<f32>,      // activations of current layer
    weights: Array2<f32>,      // weights for previous layer
    bias: Array2<f32>          // bias for previous layer
}

pub struct NeuralNet {
    size: usize,                 // number of layers
    layers: Vec<Layer> 
}

impl NeuralNet {
    /* randomize weights for each layer in nn */
    pub fn seed_weights(&mut self) {
        let mut rng = rand::thread_rng();
        for layer in self.layers.iter_mut() {
            for i in 0..layer.len {
                for j in 0..layer.incoming {
                    layer.weights[[i, j]] = rng.gen_range(0.0..1.0);
                }
            }
        }
    }

    /* load specified layer's neurons with values of input */
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
        layer.neurons = input;
    }

    /* propagate network's values forward */
    pub fn feed_forward(&mut self) {
        for layer_idx in 1..self.size {
            let len = &self.layers[layer_idx].len;
            for neuron_idx in 0..*len {
                let input = &self.layers[layer_idx-1].neurons;
                let weights = &self.layers[layer_idx].weights.row(neuron_idx);
                let prod = weights.dot(input);
                self.layers[layer_idx].neurons[neuron_idx] = prod;
            }
        }
    }

    /* print output layer of network */
    pub fn print_output(&self) {
        let layer = &self.layers[self.size-1];
        for value in &layer.neurons {
            print!("{}, ", value);
        }
        println!();
    }
}

pub fn init_nn(lengths: Vec<usize>) -> NeuralNet {
    if lengths.len() < 2 {
        eprintln!("Invalid neural net size. Exiting.");
        std::process::exit(-1);
    }

    let mut net = NeuralNet { size: lengths.len(), layers: std::vec!{} };
    for i in 0..lengths.len() {
        let mut prev = 0;
        if i != 0 {
            prev = lengths[i-1]; 
        }
        let layer = Layer {
            len: lengths[i],
            incoming: prev,
            neurons: Array1::zeros(lengths[i]),
            weights: Array2::zeros((lengths[i], prev)),
            bias: Array2::zeros((lengths[i], prev))
        };
        //println!("{:?}", layer);
        //println!();
        net.layers.push(layer);
    }
    return net;
}