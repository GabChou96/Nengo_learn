import nengo
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28) / 255.0  # Normalize
x_test = x_test.reshape(-1, 28 * 28) / 255.0

# One-hot encoding for labels
encoder = OneHotEncoder(sparse_output=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))

# Define Nengo model
model = nengo.Network()
with model:
    input_layer = nengo.Node(lambda t: x_train[int(t * 100) % len(x_train)])  # Feed MNIST images

    hidden_layer = nengo.Ensemble(1000, 28 * 28, neuron_type=nengo.LIF())  # Spiking neurons
    output_layer = nengo.Ensemble(10, 10, neuron_type=nengo.LIF())  # Output classification

    # Connect layers
    nengo.Connection(input_layer, hidden_layer)
    nengo.Connection(hidden_layer, output_layer, transform=np.random.randn(10, 784))

    # Probes to record output
    output_probe = nengo.Probe(output_layer.neurons)

# Run simulation
sim = nengo.Simulator(model)
sim.run(1.0)  # Run for 1 second

# Get classification results
output_data = sim.data[output_probe]
print("Final Output:", output_data[-1])