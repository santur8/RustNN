// include loader for mnist training/test data
use ndarray::Array1;
use ndarray::array;
mod mnist;
use mnist::*;
use neuralnet::*;
mod neuralnet;

fn main() {
    //_test_load();
    //_test_init_nn();
    //_train_mnist();
    _test_mse();
}

fn _train_mnist() {
    let idx = 100;
    let input = load_train_img(&idx);
    let mut net = init_nn(vec![784, 16, 16, 10]);
    net.load_neurons(0, input);
    net.seed_weights();
    net.feed_forward();
    net.print_neurons(0);
    net.print_neurons(3);
}

fn _test_init_nn() {
    let mut net = init_nn(vec![2, 2, 1]);
    net.seed_weights();
    net.load_neurons(0, array![0.5, 0.25]);
    net.feed_forward();
    net.feed_forward()
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

fn _test_mse() {
    //0.25, 0.6, 0.3, 0.4, 0.1, 0.8, 0.75, 0.3, 0.4, 0.2
    let mut net = init_nn(vec![784, 16, 16, 10]);
    let input: Array1<f32> = Array1::from(array![0.25, 0.6, 0.3, 0.4, 0.1, 0.8, 0.75, 0.3, 0.4, 0.2]);
    net.load_neurons(3, input);
    for i in 0..10 {
        println!("MSE of {}: {}", i, net.mse(i));
    }
}