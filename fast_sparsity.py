'''
Attempts to speed up execution by finding the correct sparsity.
'''

# pip install -q tensorflow-model-optimization
from tensorflow_model_optimization.sparsity import keras as sparsity

# Create a pruned model
epochs = 5
num_samples = len(train_images)
batch_size = 32 # default size
end_step = np.ceil(1.0 * num_samples / batch_size).astype(np.int32) * epochs
print(end_step)

pruning_params = {
#     'pruning_schedule': ConstantSparsity(0.7, 0),
    'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.00,
                                                final_sparsity=0.70,
                                                begin_step=0,
                                                end_step=end_step,
                                                frequency=100)
}

model1 = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(1000, activation=tf.nn.relu, use_bias=False),
        keras.layers.Dense(1000, activation=tf.nn.relu, use_bias=False),
        keras.layers.Dense(500, activation=tf.nn.relu, use_bias=False),
        keras.layers.Dense(200, activation=tf.nn.relu, use_bias=False),
        keras.layers.Dense(10, activation=tf.nn.softmax),
])

pruned_model = sparsity.prune_low_magnitude(model1, **pruning_params)
pruned_model.summary()
pruned_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

# Train the pruned model
# can be useful if using tensorboard
training_log_dir = "/content/drive/My Drive/Pruning_Model/pruned_epoch5_log"
model1_pruned_path = "/content/drive/My Drive/Pruning_Model/Pruned_k70/pruned_k70"

callbacks = [
    # update the pruning step.
    sparsity.UpdatePruningStep(),
    # Add summaries to keep track of the sparsity in different layers during training.
    sparsity.PruningSummaries(log_dir=training_log_dir)
]

pruned_model.fit(
    train_images,
    train_labels,
    epochs=epochs,
    verbose=1,
    callbacks=callbacks,
    # to validate loss and metrics at end of each epoch
    validation_data=(test_images, test_labels)
)

# save the pruned model
pruned_model.save_weights(model1_pruned_path)

# valuate the model on our test data
score = pruned_model.evaluate(test_images, test_labels, verbose=0)
print("Test loss: {}".format(score[0]))
print("Test accuracy: {}".format(score[1]))

# Comparison of the runtimes of pruned vs non-pruned models
import time

start = time.time()
pruned_model.evaluate(test_images, test_labels, verbose=0)
print("The pruned model took {} seconds".format(time.time() - start))

model = load_model('/content/drive/My Drive/Pruning_Model/epoch5/run1')
start = time.time()
model.evaluate(test_images, test_labels, verbose=0)
print("The un-pruned model takes {} seconds".format(time.time() - start))
