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
    _train_mnist();
}

fn _train_mnist() {
    let mut net = init_nn(vec![784, 16, 16, 10]);
    net.seed_weights();
    for idx in 0..60000 {
        let input = load_train_img(&idx);
        let label = load_train_label(&idx);
        net.load_neurons(0, input);
        net.feed_forward();
        let exp = gen_exp_output(label);
        println!("MSE: {}", net.mse(&exp));
        net.backprop(&exp);
        net.backprop_update(1.0);
    }
}

fn _test_init_nn() {
    let mut net = init_nn(vec![2, 2, 1]);
    net.seed_weights();
    net.load_neurons(0, array![0.5, 0.25]);
    net.feed_forward();
    net.print_output(2);
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