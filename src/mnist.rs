/*
mnist.rs contains functions for retrieving data
from the MNIST database of handwritten digits at 
http://yann.lecun.com/exdb/mnist/ 
 
TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  60000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
The labels values are 0 to 9. 

TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel

TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  10000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label

The labels values are 0 to 9.
TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  10000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel
*/

use std::io::{Read, Seek, SeekFrom};
use std::fs::File;
use ndarray::Array1;

pub fn load_train_img(idx: &u32) -> Array1<f32> {
    let offset = 16 + (784 * idx);
    let mut buf = [0u8; 784];
    let mut file = File::open("dataset/digs/train-images-idx3-ubyte")
        .expect("Invalid path");
    let _ = file.seek(SeekFrom::Start(u64::from(offset)));
    file.read_exact(&mut buf).expect("Error reading file");
    let norm = norm(buf);
    let mut img: Array1<f32> = Array1::zeros(784);
    for i in 0..784 {
        img[i] = norm[i];
    }
    return img;
}

pub fn load_train_label(idx: &u32) -> usize {
    let offset = 8 + idx;
    let mut buf = [0u8; 1];
    let mut file = File::open("dataset/digs/train-labels-idx1-ubyte")
        .expect("Invalid path");
    let _ = file.seek(SeekFrom::Start(u64::from(offset)));
    file.read_exact(&mut buf).expect("Error reading file");
    let label = usize::from(buf[0]);
    return label;
}

pub fn load_test_img(idx: &u32) -> Array1<f32> {
    let offset = 16 + (784 * idx);
    let mut buf = [0u8; 784];
    let mut file = File::open("dataset/digs/t10k-images-idx3-ubyte")
        .expect("Invalid path");
    let _ = file.seek(SeekFrom::Start(u64::from(offset)));
    file.read_exact(&mut buf).expect("Error reading file");
    let norm = norm(buf);
    let mut img: Array1<f32> = Array1::zeros(784);
    for i in 0..784 {
        img[i] = norm[i];
    }
    return img;
}

pub fn load_test_label(idx: &u32) -> usize {
    let offset = 8 + idx;
    let mut buf = [0u8; 1];
    let mut file = File::open("dataset/digs/t10k-labels-idx1-ubyte")
        .expect("Invalid path");
    let _ = file.seek(SeekFrom::Start(u64::from(offset)));
    file.read_exact(&mut buf).expect("Error reading file");
    let label = usize::from(buf[0]);
    return label;
}

fn norm(buf: [u8; 784]) -> [f32; 784] {
    let mut norm_buf = [0f32; 784];
    for i in 0..784 {
        let f = f32::from(buf[i]);
        norm_buf[i] = f / 255f32;
    }
    return norm_buf;
}

pub fn gen_exp_output(label: usize) -> Array1<f32> {
    let mut arr: Array1<f32> = Array1::zeros(10);
    arr[label] = 1.0;
    return arr;
}

pub fn mnist_output(output: &Array1<f32>) -> usize {
    let mut max = -1.0;
    let mut max_idx: usize = 100;
    let i: i32 = 0;
    for i in 0..10 {
        if output[i] > max {
            max = output[i];
            max_idx = i;
        }
    }
    return max_idx;
}