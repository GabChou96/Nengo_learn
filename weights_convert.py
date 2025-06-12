from tensorflow import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import nengo_dl
import nengo

from ann2snn2 import synapse

# data
segmented_df = pd.read_csv(r"segmented_geophone_data.csv")

time_steps = 2000  # Define how many past readings to use per sample
X =  segmented_df.loc[:, segmented_df.columns != "label"].values
y = segmented_df["label"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

# for i in range(10):
#     plt.plot(X_train[i])
#     plt.title(f"data for {y_train[i]}")
#     plt.show()
#
#
#
#
do_training= False
if do_training:
    # Define the ANN model with named layers
    keras_model = models.Sequential([
        layers.Input(shape=(time_steps,), name="inp"),  # Named input layer
        layers.Dense(256, activation=tf.nn.relu, name="lay0"),
        layers.Dense(128, activation=tf.nn.relu, name="lay1"),
        layers.Dense(3, activation="softmax", name="out")  # Named output layer
    ])

    keras_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    keras_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

    weights = keras_model.get_weights()
    np.savez("keras_model1_weights.npz", *weights)
    loss, accuracy = keras_model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")


    keras_model.save("keras_model1.h5")


keras_model = keras.models.load_model("keras_model1.h5")

# converter = nengo_dl.Converter(model)
activation = nengo.SpikingRectifiedLinear()
scale_firing_rates = 1
# synapse = None #0.001
converter = nengo_dl.Converter(
    keras_model,
    swap_activations={tf.nn.relu: activation},
    scale_firing_rates=scale_firing_rates,
)
# with nengo_dl.Simulator(converter.net) as sim:
#     # Load Keras-trained weights into the simulator
#     sim.load_params(converter)
params_file = "converter_keras_model1"
with nengo_dl.Simulator(converter.net) as sim:
    sim.run(1.0)
    sim.save_params(params_file)

functional_model = converter.model

# Access input and output layers using the functional model
nengo_input = converter.inputs[functional_model.input]
nengo_output = converter.outputs[functional_model.output]

# build network, load in trained weights, run inference on test images
with nengo_dl.Simulator(
    converter.net, minibatch_size=1, progress_bar=False
) as nengo_sim:
    nengo_sim.load_params(params_file)
    print(np.expand_dims(X_test, axis = 1).shape)
    data = nengo_sim.predict({nengo_input: np.expand_dims(X_test, axis = 1)})
np.save("pred_100.npy", data[nengo_output])

pred = np.argmax(data[nengo_output], axis =-1)
accuracy = (y_test == pred[:, 0]).mean()
print(accuracy)

for node in converter.net.nodes:
    print(node.label, node.size_in, node.size_out)


for node in converter.net.all_ensembles:
    print(node, node.neuron_type, node.n_neurons, node.dimensions)