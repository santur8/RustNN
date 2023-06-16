// include loader for mnist training/test data
use ndarray::Array1;
use ndarray::array;
mod mnist;
use mnist::*;
use neuralnet::*;
mod neuralnet;

fn main() {
    //test_load();
    test_init_nn();
}

fn test_init_nn() {
    let v: Vec<usize> = vec![2, 2, 1];
    let mut net = init_nn(&v);
    seed_nn_weights(&mut net);
    let input: Array1<f32> = array![0.5, 0.25];
    load_input(&mut net, input);
}

fn _test_load() {
    let idx = 784;
    let input: Array1<f32> = load_train_img(&idx);
    _print_mnist_input(input);
    println!("{}", load_train_label(&idx));
}

fn _print_mnist_input(input: Array1<f32>) {
    for i in 1..785 {
        print!("{:.1}", input[i-1]);
        if i % 28 == 0 {
            println!();
        }
    }
}