#!/bin/bash/python

import matplotlib.pyplot as plt
import numpy as np
import os
from mnist import MNIST
import cProfile

from caffe2.python import (
    brew,
    core,
    model_helper,
    net_drawer, # for drawing network diagram
    optimizer,
    visualize,
    workspace,
    utils, # for generating database from MNIST images
)

from caffe2.proto import caffe2_pb2 # for generating database from MNIST images
from IPython import display # for drawing network diagram

core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
print("Necessities imported!")

# Load the MNIST data
mndata = MNIST('./mnist')
images, labels = mndata.load_training()

# specify the input image data
images = np.array(images, dtype = np.float32)
MNIST_data = images[:600,:] # only use 1.0 percent of the 60,000 images because this calculation on the CPU will be slow

# specify the label data as integers [0, 9].
MNIST_labels = np.array(labels, dtype = np.int) # caffe2 expects class labesls as integers

# First, let's see how one can construct a TensorProtos protocol buffer from numpy arrays.
feature_and_label = caffe2_pb2.TensorProtos()
feature_and_label.protos.extend([
    utils.NumpyArrayToCaffe2Tensor(MNIST_data[0,:]),
    utils.NumpyArrayToCaffe2Tensor(MNIST_labels[0])])

print 'This is what the tensor proto looks like for a feature and its label:'
print feature_and_label

print 'This is the compact string that gets written into the db:'
print feature_and_label.SerializeToString()

# write the db.
def write_db(db_type, db_name, features, labels):
    db = core.C.create_db(db_type, db_name, core.C.Mode.write)
    transaction = db.new_transaction()
    for i in range(MNIST_data.shape[0]):
        feature_and_label = caffe2_pb2.TensorProtos()
        feature_and_label.protos.extend([
            utils.NumpyArrayToCaffe2Tensor(features[i,:]),
            utils.NumpyArrayToCaffe2Tensor(labels[i])])
        transaction.put(
            'train_%03d'.format(i),
            feature_and_label.SerializeToString())
    # Close the transaction, and then close the db.
    del transaction
    del db

write_db("minidb", "../MNIST_train.minidb", MNIST_data, MNIST_labels)

# just in case, we will reset any blobs in the current workspace
workspace.ResetWorkspace()

# Build a simple Logistic function the model workflow
# define the name of the model as 'net'
mlp_net = core.Net("MLP_the_hard_way")

dbreader = mlp_net.CreateDB([], "dbreader", db="./MNIST_train.minidb", db_type="minidb")

data_uint8, label_float = mlp_net.TensorProtosDBInput(  [dbreader], ["data_uint8", "label_float"], 
                                                    batch_size = 10,
                                                    db = "./MNIST_train.minidb", 
                                                    db_type = "minidb")

# cast the data to float
data = mlp_net.Cast(data_uint8, "data", to = core.DataType.FLOAT)

# scale data from [0,255] down to [0,1]
data = mlp_net.Scale(data, data, scale = float(1./256))

# cast the labels to int
label = mlp_net.Cast(label_float, "label", to = core.DataType.INT32)

# don't need the gradient for the backward pass
data = mlp_net.StopGradient(data, data)

# to define the first hidden layer, we need the following:
# martix of weights 'W_1', vector of biases 'b_1', neuron inputs 'in_1', and activation function output 'out_1'
W_1   = mlp_net.XavierFill(  [], ["W_1"], shape=[10, 28*28], run_once=0)
b_1   = mlp_net.ConstantFill([], ["b_1"], shape=[10,], value = 1.0, run_once=0)
in_1  = mlp_net.FC(          [data, W_1, b_1], "in_1")
out_1 = mlp_net.Relu(        [in_1], "out_1")

# to define the second hidden layer, we need the following:
W_2   = mlp_net.XavierFill(  [], ["W_2"], shape=[10, 10], run_once=0) # we only need 10 weights = out_1 neuron number
b_2   = mlp_net.ConstantFill([], ["b_2"], shape=[10,], value = 1.0, run_once=0)
in_2  = mlp_net.FC(          [out_1, W_2, b_2], "in_2")
out_2 = mlp_net.Relu(        [in_2], "out_2")

# now we need a softmax function to collate the  
softmax = mlp_net.Softmax([out_2], "softmax")

# Compute cross entropy between softmax scores and labels
xent = mlp_net.LabelCrossEntropy([softmax, label], 'xent') # LabelCrossEntropy expects class labesls as integers

# Compute the expected loss
avg_loss = mlp_net.AveragedLoss(xent, "loss")

# Track the accuracy of the model
accuracy = mlp_net.Accuracy([softmax, "label"], "accuracy")

# Use the average loss we just computed to add gradient operators to the model
gradients = mlp_net.AddGradientOperators([avg_loss])
gradients[core.BlobReference("b_1")]

# Define iteration and iter_mutex variables for the AtomicIter operator
iteration = mlp_net.ConstantFill( [], "iteration", shape=[1], value=10000, dtype=core.DataType.INT64)
iter_mutex = mlp_net.CreateMutex( [], "iteration_mutex")

# Define the iterator operator
optimizer_iteration = mlp_net.AtomicIter([iter_mutex, iteration], "optimizer_iteration")

# Define the optimization learning rate scheduler
Sgd_Optimizer =  mlp_net.LearningRate([optimizer_iteration], 'Sgd_Optimizer', base_lr=-0.1, policy="step", stepsize=20, gamma=0.9)

# Each GPU/CPU must have its own ONE blob, thus modify the name
# to include device information.
ONE = mlp_net.ConstantFill([], "ONE", shape=[1], value=1.0)

# Update the weights of each hidden layer
W_2 = mlp_net.WeightedSum([W_2, ONE, core.BlobReference("W_2_grad"), Sgd_Optimizer], W_2)
b_2 = mlp_net.WeightedSum([b_2, ONE, core.BlobReference("b_2_grad"), Sgd_Optimizer], b_2)

W_1 = mlp_net.WeightedSum([W_1, ONE, core.BlobReference("W_1_grad"), Sgd_Optimizer], W_1)
b_1 = mlp_net.WeightedSum([b_1, ONE, core.BlobReference("b_1_grad"), Sgd_Optimizer], b_1)

graph = net_drawer.GetPydotGraph(mlp_net, rankdir="LR")
display.Image(graph.create_png(), width=800)
with open("simple_mlp_1.png", "wb") as png:
    png.write(graph.create_png())

workspace.RunNetOnce(mlp_net)