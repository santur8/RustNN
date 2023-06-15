// include loader for mnist training/test data
use ndarray::Array1;
mod mnist;
use mnist::*;
use neuralnet::*;
mod neuralnet;

fn main() {
    //test_load();
    test_init_nn();
}

fn test_init_nn() {
    let v: Vec<usize> = vec![2, 3, 4];
    let mut net = init_nn(&v);
    seed_nn(&mut net);
}

fn test_load() {
    let idx = 784;
    let input: Array1<f32> = load_train_img(&idx);
    print_input(input);
    println!("{}", load_train_label(&idx));
}

fn print_input(input: Array1<f32>) {
    for i in 1..785 {
        print!("{:.1}", input[i-1]);
        if i % 28 == 0 {
            println!();
        }
    }
}