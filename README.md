# neuralnet
A simple neural network framework build in rust.

To train a network on the mnist database of handwritten digits,
download the images/labels from "http://yann.lecun.com/exdb/mnist/",
edit the paths in init_mnist_buffers() in mnist.rs, and use the 
train_mnist() and test_mnist functions.

For new datasets, create functions to load inputs to the first layer
of the network and generate expected outputs.
