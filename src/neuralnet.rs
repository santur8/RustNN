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

pub fn init_nn(lengths: &Vec<usize>) -> NeuralNet {
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


pub fn seed_nn_weights(net: &mut NeuralNet) {
    // randomize weights for each layer in nn
    let mut rng = rand::thread_rng();
    for layer in net.layers.iter_mut() {
        for i in 0..layer.len {
            for j in 0..layer.incoming {
                layer.weights[[i, j]] = rng.gen_range(0.0..1.0);
            }
        }
    }
}

pub fn load_input(net: &mut NeuralNet, input: Array1<f32>) {
    let mut layer = &mut net.layers[0];
    if layer.neurons.len() != input.len() {
        eprintln!("Invalid length for input layer. Exiting.");
        std::process::exit(-1);
    }
    layer.neurons = input;
}

pub fn feed_forward(net: &mut NeuralNet) {
    for layer_idx in 1..net.size {
        let len = net.layers[layer_idx].len;
        for neuron_idx in 0..len {
            let input = &net.layers[layer_idx-1].neurons;
            let weights = &net.layers[layer_idx].weights.row(neuron_idx);
            let weights = weights.into_shape((len,1)).unwrap();
        }
    }
}