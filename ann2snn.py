
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


# Define the ANN model
keras_model = models.Sequential([
    layers.Input(shape=(time_steps,)),  # Single feature input
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(3, activation="softmax")
])
keras_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
keras_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))



loss, accuracy = keras_model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

presentation_time = 0.25
n_out = 3

converter = nengo_dl.Converter(keras_model)
model = converter.net  # Use this functional model # The converted Nengo model
# Convert the model to NengoDL format
with model:
    input_node = nengo.Node(
        nengo.processes.PresentInput(X_test, presentation_time), label="input"
    )
    # Output display node
    output_node = nengo.Node(size_in=n_out, label="output class")
    print(converter.layers.values(), converter.layers.keys())
    values_view = converter.layers.items()  # Get the ValuesView
    values_list = list(values_view)  # Convert to a list

    for i in range(len(values_list)):
        print(i, values_list[i])  # Access each element
    # Projections to and from the fpga ensemble
    print(list(converter.outputs.items()))
    print(keras_model.output)
    print(list(converter.outputs.values())[0])
    # nengo.Connection(input_node,values_list[0] , synapse=None)
    # nengo.Connection(list(converter.outputs.values())[0], output_node, synapse=None)

    # Output SPA display (for nengo_gui)
    vocab_names = [
        "Nothing",
        "Man",
        "Car",
    ]
    vocab_vectors = np.eye(len(vocab_names))

    vocab = nengo.spa.Vocabulary(len(vocab_names))
    for name, vector in zip(vocab_names, vocab_vectors):
        vocab.add(name, vector)

    config = nengo.Config(nengo.Ensemble)
    config[nengo.Ensemble].neuron_type = nengo.Direct()
    with config:
        output_spa = nengo.spa.State(len(vocab_names), subdimensions=n_out, vocab=vocab)
    nengo.Connection(list(converter.outputs.values())[0], output_spa.input)
 






