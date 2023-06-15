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
        return NeuralNet { size: 0, layers: std::vec!{} };
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


pub fn seed_nn(net: &mut NeuralNet) {
    // randomize weights for each layer in nn
    let mut rng = rand::thread_rng();
    for layer in net.layers.iter_mut() {
        for i in 0..layer.len {
            for j in 0..layer.incoming {
                layer.weights[[i, j]] = rng.gen_range(0.0..1.0);
            }
        }
        println!("{:?}", layer);
    }
}