
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import nengo
import nengo_dl
from tensorflow.keras import layers, models

# data
segmented_df = pd.read_csv("segmented_geophone_data.csv")

time_steps = 2000  # Define how many past readings to use per sample
X =  segmented_df.loc[:, segmented_df.columns != "label"].values
y = segmented_df["label"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")


# Define the ANN model with named layers
keras_model = models.Sequential([
    layers.Input(shape=(time_steps,), name="inp"),  # Named input layer
    layers.Dense(16, activation=tf.nn.relu, name="lay0"),
    layers.Dense(8, activation=tf.nn.relu, name="lay1"),
    layers.Dense(3, activation=tf.nn.relu, name="out")  # Named output layer
])

# keras_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# keras_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))




# weights = keras_model.get_weights()
# np.savez("keras_model_weights.npz", *weights)
# loss, accuracy = keras_model.evaluate(X_test, y_test)
# print(f"Test Loss: {loss:.4f}")
# print(f"Test Accuracy: {accuracy:.4f}")

################
converter = nengo_dl.Converter(keras_model)
# Use the functional model from the converter
functional_model = converter.model
# Reshape training and validation data to include the n_steps dimension
n_steps = 1  # Define the number of timesteps (can be adjusted as needed)

# Ensure the batch size matches or exceeds the minibatch_size
minibatch_size = 10

X_train_batches = np.repeat(X_train[:X_train.shape[0]//10*10, :].reshape(minibatch_size, -1, 2000), 20, axis=1)
X_val_batches = np.repeat(X_val[:X_val.shape[0]//10*10, :].reshape(minibatch_size, -1, 2000), 20, axis=1)
X_test_batches = X_test[:X_test.shape[0]//10*10, :].reshape(minibatch_size, -1, 2000)

y_train_batches = np.repeat(y_train[:y_train.shape[0]//10*10].reshape(minibatch_size, -1, 1), 20, axis=1)
y_val_batches = np.repeat(y_val[:y_val.shape[0]//10*10].reshape(minibatch_size, -1, 1), 20, axis=1)
y_test_batches = y_test[:y_test.shape[0]//10*10].reshape(minibatch_size, -1, 1)

print(X_train_batches.shape, y_train_batches.shape)

do_training = True
if do_training:
    with nengo_dl.Simulator(converter.net) as sim:
        # run training
        sim.compile(
            optimizer=tf.optimizers.Adam(0.001),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.sparse_categorical_accuracy],
        )
        sim.fit(
            {converter.inputs[functional_model.input]: X_train_batches},
            {converter.outputs[functional_model.output]: y_train_batches},
                validation_data=(
                {converter.inputs[functional_model.input]: X_val_batches},
                {converter.outputs[functional_model.output]: y_val_batches},
            ),
            epochs=20,
            steps_per_epoch=len(X_train) // minibatch_size
        )
        # save the parameters to file
        sim.save_params("./keras_snn_trained_params")
        print("steps_per_epoch", len(X_train) // minibatch_size)



presentation_time=0.25
n_out = keras_model.output_shape[-1]  # Number of output classes




###############
activation = nengo.SpikingRectifiedLinear()
params_file = "keras_snn_trained_params"
n_steps = 30
scale_firing_rates = 1
synapse = 0.01
n_test = 400



# convert the keras model to a nengo network
nengo_converter = nengo_dl.Converter(
    keras_model,
    swap_activations={tf.nn.relu: activation},
    scale_firing_rates=scale_firing_rates,
    synapse=synapse,
)

# Use the functional model from the converter
functional_model = nengo_converter.model

# Access input and output layers using the functional model
nengo_input = nengo_converter.inputs[functional_model.input]
nengo_output = nengo_converter.outputs[functional_model.output]

# add a probe to the first convolutional layer to record activity.
# we'll only record from a subset of neurons, to save memory.
sample_neurons = np.linspace(
    0,
    np.prod(functional_model.get_layer("lay0").output_shape[1:]),
    1000,
    endpoint=False,
    dtype=np.int32,
)
# Use the output tensor as the key to access the layer in nengo_converter
with nengo_converter.net:
    lay0_probe = nengo.Probe(list(nengo_converter.layers.values())[1][sample_neurons])

# repeat inputs for some number of timesteps
tiled_test_images = np.tile(X_test_batches[:n_test], (1, n_steps, 1))

# set some options to speed up simulation
with nengo_converter.net:
    nengo_dl.configure_settings(stateful=False)

# build network, load in trained weights, run inference on test images
with nengo_dl.Simulator(
    nengo_converter.net, minibatch_size=10, progress_bar=False
) as nengo_sim:
    nengo_sim.load_params(params_file)
    data = nengo_sim.predict({nengo_input: tiled_test_images})

# compute accuracy on test data, using output of network on
# last timestep
predictions = np.argmax(data[nengo_output][:, -1], axis=-1)
accuracy = (predictions == y_test_batches[:n_test, 0, 0]).mean()
print(f"Test accuracy: {100 * accuracy:.2f}%")

# # plot the results
# for ii in range(3):
#     plt.figure(figsize=(12, 4))
#
#     plt.subplot(1, 3, 1)
#     plt.title("Input image")
#     plt.imshow(X_test[ii, 0].reshape((28, 28)), cmap="gray")
#     plt.axis("off")
#
#     plt.subplot(1, 3, 2)
#     scaled_data = data[lay0_probe][ii] * scale_firing_rates
#     if isinstance(activation, nengo.SpikingRectifiedLinear):
#         scaled_data *= 0.001
#         rates = np.sum(scaled_data, axis=0) / (n_steps * nengo_sim.dt)
#         plt.ylabel("Number of spikes")
#     else:
#         rates = scaled_data
#         plt.ylabel("Firing rates (Hz)")
#     plt.xlabel("Timestep")
#     plt.title(
#         f"Neural activities (conv0 mean={rates.mean():.1f} Hz, "
#         f"max={rates.max():.1f} Hz)"
#     )
#     plt.plot(scaled_data)
#
#     plt.subplot(1, 3, 3)
#     plt.title("Output predictions")
#     plt.plot(tf.nn.softmax(data[nengo_output][ii]))
#     plt.legend([str(j) for j in range(10)], loc="upper left")
#     plt.xlabel("Timestep")
#     plt.ylabel("Probability")
#
#     plt.tight_layout()

##############
#
# converter = nengo_dl.Converter(keras_model)
# model = converter.net  # Use this functional model # The converted Nengo model
# # Convert the model to NengoDL format
# with model:
#     input_node = nengo.Node(
#         nengo.processes.PresentInput(X_test, presentation_time), label="input"
#     )
#     # Output display node
#     output_node = nengo.Node(size_in=n_out, label="output class")
#     # print(converter.layer.values(), converter.layer.keys())
#     # # Projections to and from the fpga ensemble
#     # nengo.Connection(input_node,converter.layer.values()[0] , synapse=None)
#     # nengo.Connection(converter.output.values()[0], output_node, synapse=None)
#
#     # Output SPA display (for nengo_gui)
#     vocab_names = [
#         "Nothing",
#         "Man",
#         "Car",
#     ]
#     vocab_vectors = np.eye(len(vocab_names))
#
#     vocab = nengo.spa.Vocabulary(len(vocab_names))
#     for name, vector in zip(vocab_names, vocab_vectors):
#         vocab.add(name, vector)
#
#     config = nengo.Config(nengo.Ensemble)
#     config[nengo.Ensemble].neuron_type = nengo.Direct()
#     with config:
#         output_spa = nengo.spa.State(len(vocab_names), subdimensions=n_out, vocab=vocab)
#     nengo.Connection(output_node, output_spa.input)
#



