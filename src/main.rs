// include loader for mnist training/test data
use ndarray::Array1;
mod mnist;
use mnist::*;

fn main() {
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