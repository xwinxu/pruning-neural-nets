from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
# Good reference: https://www.tensorflow.org/beta/guide/keras/training_and_evaluation
import tensorflow as tf
from tensorflow import keras
from typing import List

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os

# Load data
# reference for loading data: https://www.tensorflow.org/beta/tutorials/load_data/numpy
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Set up model
def create_model():
    """Create a model with specified architecture."""
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(1000, activation=tf.nn.relu, use_bias=False),
        keras.layers.Dense(1000, activation=tf.nn.relu, use_bias=False),
        keras.layers.Dense(500, activation=tf.nn.relu, use_bias=False),
        keras.layers.Dense(200, activation=tf.nn.relu, use_bias=False),
        keras.layers.Dense(10, activation=tf.nn.softmax),
    ])
    # compile the model
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics= ['accuracy'],
             )
    return model

# m = create_model()
# m.fit(train_images, train_labels, epochs=10`)

m2 = create_model()
m2.fit(train_images, train_labels, epochs=5)
m2.save_weights('/content/drive/My Drive/Pruning_Model/epoch5.2/run1')
tf.keras.models.save_model(m2, "/content/drive/My Drive/Pruning_Model/epoch5.2/run1")
print("model saved successfully")

# Prune the Model
k_pcent = [0, 25, 50, 60, 70, 80, 90, 95, 97, 99]

def concat_layer_weights(model) -> List[np.ndarray]:
    """Return a list of arrays for each layer."""
    result = []
    for i in range(1, 5):
      # index 0 for the weights, 1 for the biases
      result.append(np.copy(model.layers[i].get_weights()[0]))
    return result

# Weight Pruning
def prune_weights(model, k: int):
    """Return a model with k% of smallest weights pruned."""
    weights = concat_layer_weights(model)
    # master holds all of the model's nodes in flattened 1D shape.
    master = np.array([])
    # store info of layers as: (layer, row, col).
    info = []
    # the input layer doesn't have weights, i.e. layer = 1 to start.
    for i, w in enumerate(weights):
        info.append((i+1, np.shape(w)[0], np.shape(w)[1]))
        master = np.concatenate([master, w.flatten()])

    # ensure no negative values.
    master = master.flatten()
    master_abs = np.abs(master)
    # floors, might be insignificant difference.
    num_to_prune = int(k * len(master) / 100)
    
    # get the ranks
    all_idx = np.argsort(master_abs)
    all_ranks = np.empty_like(all_idx)
    all_ranks[all_idx] = np.arange(len(master_abs))
    master[np.where(all_ranks <= num_to_prune)] = 0

    prev = 0
    for i in info:
        num = i[1] * i[2]
        rank = all_ranks[prev: (num + prev)]
        # set weights using np array:
        # https://github.com/tensorflow/tensorflow/issues/19108
        # https://github.com/tensorflow/tensorflow/issues/29663
        model.layers[i[0]].set_weights([master[prev:(num + prev)].reshape((i[1], i[2]))])
        prev += num

# Unit Pruning
def prune_units(model, k: int):
    """Return a model with k% of smallest units deleted from the network."""
    weights = concat_layer_weights(model)
    # calculate L2-norm for each weight, axis = 0 == column
    unit_norms = np.concatenate([np.linalg.norm(w, axis=0) for w in weights])
    num_units_to_prune = int(k * len(unit_norms) / 100)
    # get the ranking for the array
    # adapted from: https://stackoverflow.com/questions/5284646/rank-items-in-an-array-using-python-numpy-without-sorting-array-twice
    all_idx = np.argsort(unit_norms)
    all_ranks = np.empty_like(all_idx)
    all_ranks[all_idx] = np.arange(len(unit_norms))
    start_idx = 0
    for i, w in enumerate(weights):
        # the rank of the value at each index of the layer matrix.
        rank = all_ranks[start_idx: (start_idx + np.shape(w)[1])]
        # null out all the rank values as calculated above.
        w[:, np.where(rank <= num_units_to_prune)] = 0
        model.layers[i + 1].set_weights([w])
        start_idx += np.shape(w)[1]

# Percent sparsity
k_pcent = [0, 25, 50, 60, 70, 80, 90, 95, 97, 99]

# Weight pruned accuracy
weight_accuracies = []
loss = []
for k in k_pcent:
    print("the value of k is: {}".format(k))
    model = load_model(model_path=model_path)
    prune_weights(model, k)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    weight_accuracies.append(test_acc)
    loss.append(test_loss)
    
    
# matplotlib tutorial: https://matplotlib.org/users/pyplot_tutorial.html
plt.plot(k_pcent, weight_accuracies)
plt.xlabel('Percent Weights Pruned')
plt.ylabel('Test Accuracy')
plt.title('Result of Weight Pruning')

# Unit pruned accuracy
unit_accuracies = []
model_loss = []
for k in k_pcent:
    print(k)
    model = load_model(model_path=model_path)
    prune_units(model, k)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    unit_accuracies.append(test_acc)
    model_loss.append(test_loss)
    
plt.plot(k_pcent, unit_accuracies)
plt.xlabel('Percent Units Pruned')
plt.ylabel('Test Accuracy')
plt.title('Result of Unit Pruning')


