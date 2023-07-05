// include loader for mnist training/test data
use ndarray::Array1;
use ndarray::array;
use rand::Rng;
mod mnist;
use mnist::*;
use neuralnet::*;
mod neuralnet;

fn main() {
    //_test_load();
    // /_test_init_nn();
    _train_mnist();
    //_train_xor();
}

fn _train_xor() {
    let mut net = init_nn(vec![2, 2, 1]);
    let mut rng = rand::thread_rng();
    net.seed_weights();
    let mut count = 0;
    let mut avg = 0.0;
    let epoch = 10000;
    while avg < 0.85 {
        //net.seed_weights();
        count = 0;
        for i in 0..epoch {
            let i1: i32 = rng.gen_range(0..=1);
            let i2: i32 = rng.gen_range(0..=1);
            let out = i1 ^ i2;
            let input = Array1::from(array![i1 as f32, i2 as f32]);
            let exp = Array1::from(array![out as f32]);
            //println!("{}", input);
            //println!("{}", exp);
            
            net.load_neurons(0, input);
            net.feed_forward();
            if net.get_output()[0].round() as i32 == out {
                count += 1;
            }
            //println!("{} ^ {} = {}", i1, i2, net.get_output()[0].round());
            println!("MSE: {:.3} --- OUT: {:.3} --- {} ^ {} = {}", net.mse(&exp), net.get_output(), i1, i2, net.get_output()[0].round());
            net.backprop(&exp);
        }
        avg = (count as f32) / (epoch as f32);
        println!("training accuracy: {:.5}", avg);
        std::thread::sleep(std::time::Duration::from_secs(1));
    }

    // testing purposes
    // let input = Array1::from(array![1 as f32, 1 as f32]);
    // let exp = Array1::from(array![0 as f32]);
    // net.load_neurons(0, input);
    // net.feed_forward();
    // println!("{} ^ {} = {}", 1, 1, net.get_output()[0].round());
    // println!("MSE: {:.3} --- OUT: {:.3}", net.mse(&exp), net.get_output());
    // net.backprop(&exp);
    _test_xor(&mut net);
    
}

fn _test_xor(net: &mut NeuralNet) {
    let mut rng = rand::thread_rng();
    let mut count = 0;
    for i in 0..1000 {
        let i1: i32 = rng.gen_range(0..=1);
        let i2: i32 = rng.gen_range(0..=1);
        let out = i1 ^ i2;
        let input = Array1::from(array![i1 as f32, i2 as f32]);
        let exp = Array1::from(array![out as f32]);
        //println!("{}", input);
        //println!("{}", exp);
        
        net.load_neurons(0, input);
        net.feed_forward();
        if net.get_output()[0].round() as i32 == out {
            count += 1;
        }
        println!("MSE: {:.3} --- OUT: {:.3}, {} ^ {} = {}", net.mse(&exp), net.get_output(), i1, i2, net.get_output()[0].round());
    }
    println!("\ntraining accuracy: {}", (count as f32) / 1000.0);
}

fn _train_mnist() {
    let mut net = init_nn(vec![784, 16, 16, 10]);
    net.seed_weights();
    for idx in 0..60000 {
        let input = load_train_img(&idx);
        let label = load_train_label(&idx);
        //println!("{}", label);
        net.load_neurons(0, input);
        net.feed_forward();
        //net.print_output(3);
        let exp = gen_exp_output(label);
        //println!("value: {}\n{:?}", label, exp);
        println!("MSE: {:.3} --- EXP: {} --- Output: {}", net.mse(&exp), label, mnist_output(net.get_output()));
        net.backprop(&exp);
        //net.backprop_update(1.0);
        //println!("{}", idx);

    }
}

fn _test_init_nn() {
    let mut net = init_nn(vec![2, 2, 1]);
    net.seed_weights();
    net.load_neurons(0, array![0.5, 0.25]);
    net.feed_forward();
    net.feed_forward();
    net.feed_forward();
    net.feed_forward();
    net.feed_forward();
    net.feed_forward();
    net.feed_forward();
    net.feed_forward();
    net.print_output(2);
}

fn _test_load() {
    let idx = 784;
    let input: Array1<f32> = load_train_img(&idx);
    _print_mnist_input(&input);
    println!("{}", load_train_label(&idx));
}

fn _print_mnist_input(input: &Array1<f32>) {
    for i in 1..785 {
        print!("{:.1}", input[i-1]);
        if i % 28 == 0 {
            println!();
        }
    }
}