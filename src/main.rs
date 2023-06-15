//use ndarray::prelude::*;
//use ndarray::Array2;

// include loader for mnist training/test data
mod mnist;
use mnist::*;

fn main() {
    let idx = 257;
    let img = load_train_img(&idx);
    println!("{}", load_train_label(&idx));
}
