// include loader for mnist training/test data
use ndarray::Array1;
use ndarray::array;
use rand::Rng;
mod mnist;
use mnist::*;
use neuralnet::*;
mod neuralnet;

fn main() {
    let mut net = init_nn(vec![784, 16, 16, 10]);
    init_mnist_buffers();
    _train_mnist(&mut net);
    _test_mnist(&mut net);
}

fn _train_mnist(net: &mut NeuralNet) {
    net.seed_weights();
    let mut count = 0;
    let mut correct = 0;
    let mut iter = 0;
    let mut avg = 0.0;
    while avg < 0.85 {
        for idx in 0..60000 {
            count += 1;
            let input = load_train_img_mnist(&idx);
            let label = load_train_label_mnist(&idx);
            net.load_neurons(0, input);
            net.feed_forward();
            let exp = gen_exp_output(label);
            let output = mnist_output(net.get_output());
            //println!("MSE: {:.3} --- EXP: {} --- Output: {}", net.mse(&exp), label, output);
            net.backprop(&exp);
    
            if output == label {
                correct += 1;
            }
            if count % 1000 == 0 {
                iter += 1;
                avg = (correct as f32) / (count as f32);
                println!("Iteration {}: {:.3} correct", iter, avg);
            }
        }
    }
}

fn _test_mnist(net: &mut NeuralNet) {
    let mut correct = 0;
    for idx in 0..10000 {
        let input = load_test_img_mnist(&idx);
        let label = load_test_label_mnist(&idx);
        net.load_neurons(0, input);
        net.feed_forward();
        let output = mnist_output(net.get_output());
        if output == label {
            correct += 1;
        }
    }
    println!("correct count: {}", correct);
    let avg = (correct as f32) / (10000 as f32);
    println!("Testing accuracy: {:.4}", avg);
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

fn _train_xor() {
    let mut net = init_nn(vec![2, 2, 1]);
    let mut rng = rand::thread_rng();
    net.seed_weights();
    let mut count = 0;
    let mut avg = 0.0;
    let epoch = 10000;
    while avg < 0.85 {
        count = 0;
        for i in 0..epoch {
            let i1: i32 = rng.gen_range(0..=1);
            let i2: i32 = rng.gen_range(0..=1);
            let out = i1 ^ i2;
            let input = Array1::from(array![i1 as f32, i2 as f32]);
            let exp = Array1::from(array![out as f32]);
            
            net.load_neurons(0, input);
            net.feed_forward();
            if net.get_output()[0].round() as i32 == out {
                count += 1;
            }
            println!("MSE: {:.3} --- OUT: {:.3} --- {} ^ {} = {}", net.mse(&exp), net.get_output(), i1, i2, net.get_output()[0].round());
            net.backprop(&exp);
        }
        avg = (count as f32) / (epoch as f32);
        println!("training accuracy: {:.5}", avg);
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
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

fn _test_load() {
    let idx = 2;
    let input: Array1<f32> = load_train_img_mnist(&idx);
    _print_mnist_input(&input);
    println!("{}", load_train_label_mnist(&idx));
}

fn _print_mnist_input(input: &Array1<f32>) {
    for i in 1..785 {
        print!("{:.1}", input[i-1]);
        if i % 28 == 0 {
            println!();
        }
    }
}