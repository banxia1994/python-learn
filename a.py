# This file gives the CNN model to predict eye and nose landmark in LEVEL-{level}
name: "landmark_{level}_{name}"
layer {
    name: "hdf5_train_data"
    type: "HDF5Data"
    top: "data"
    top: "landmark"
    include {
        phase: TRAIN
    }
    hdf5_data_param {
        source: "train/{level}_{name}/train.txt"
        batch_size: 64
    }
}
layer {
    name: "hdf5_test_data"
    type: "HDF5Data"
    top: "data"
    top: "landmark"
    include {
        phase: TEST
    }
    hdf5_data_param {
        source: "train/{level}_{name}/test.txt"
        batch_size: 64
    }
}
layer {
    name: "conv1"
    type: "Convolution"
    bottom: "data"
    top: "conv1"
    param {
        lr_mult: 1
    }
    param {
        lr_mult: 2
    }
    convolution_param {
        num_output: 20
        kernel_size: 4
        weight_filler {
            type: "xavier"
        }
        bias_filler {
            type: "constant"
        }
    }
}
layer {
    name: "relu1"
    type: "ReLU"
    bottom: "conv1"
    top: "conv1"
}
layer {
    name: "pool1"
    type: "Pooling"
    bottom: "conv1"
    top: "pool1"
    pooling_param {
        pool: MAX
        kernel_size: 2
        stride: 2
    }
}
layer {
    name: "conv2"
    type: "Convolution"
    bottom: "pool1"
    top: "conv2"
    param {
        lr_mult: 1
    }
    param {
        lr_mult: 2
    }
    convolution_param {
        num_output: 40
        kernel_size: 3
        weight_filler {
            type: "xavier"
        }
        bias_filler {
            type: "constant"
        }
    }
}
layer {
    name: "relu2"
    type: "ReLU"
    bottom: "conv2"
    top: "conv2"
}
layer {
    name: "pool2"
    type: "Pooling"
    bottom: "conv2"
    top: "pool2"
    pooling_param {
        pool: MAX
        kernel_size: 2
        stride: 2
    }
}
layer {
    name: "conv3"
    type: "Convolution"
    bottom: "pool2"
    top: "conv3"
    param {
        lr_mult: 1
    }
    param {
        lr_mult: 2
    }
    convolution_param {
        num_output: 60
        kernel_size: 3
        weight_filler {
            type: "xavier"
        }
        bias_filler {
            type: "constant"
        }
    }
}
layer {
    name: "relu3"
    type: "ReLU"
    bottom: "conv3"
    top: "conv3"
}
layer {
    name: "pool3"
    type: "Pooling"
    bottom: "conv3"
    top: "pool3"
    pooling_param {
        pool: MAX
        kernel_size: 2
        stride: 2
    }
}
layer {
    name: "conv4"
    type: "Convolution"
    bottom: "pool3"
    top: "conv4"
    param {
        lr_mult: 1
    }
    param {
        lr_mult: 2
    }
    convolution_param {
        num_output: 80
        kernel_size: 2
        weight_filler {
            type: "xavier"
        }
        bias_filler {
            type: "constant"
        }
    }
}
layer {
    name: "relu4"
    type: "ReLU"
    bottom: "conv4"
    top: "conv4"
}
layer {
    name: "fc1"
    type: "InnerProduct"
    bottom: "conv4"
    top: "fc1"
    param {
        lr_mult: 1
    }
    param {
        lr_mult: 2
    }
    inner_product_param {
        num_output: 100
        weight_filler {
            type: "xavier"
        }
        bias_filler {
            type: "constant"
        }
    }
}
layer {
    name: "relu_fc1"
    type: "ReLU"
    bottom: "fc1"
    top: "fc1"
}
layer {
    name: "fc2"
    type: "InnerProduct"
    bottom: "fc1"
    top: "fc2"
    param {
        lr_mult: 1
    }
    param {
        lr_mult: 2
    }
    inner_product_param {
        num_output: 6
        weight_filler {
            type: "xavier"
        }
        bias_filler {
            type: "constant"
        }
    }
}
layer {
    name: "relu_fc2"
    type: "ReLU"
    bottom: "fc2"
    top: "fc2"
}
layer {
    name: "error"
    type: "EuclideanLoss"
    bottom: "fc2"
    bottom: "landmark"
    top: "error"
    include {
        phase: TEST
    }
}
layer {
    name: "loss"
    type: "EuclideanLoss"
    bottom: "fc2"
    bottom: "landmark"
    top: "loss"
    include {
        phase: TRAIN
    }
}
