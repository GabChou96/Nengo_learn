import nengo
import numpy as np
import matplotlib.pyplot as plt
from keras.src.applications.efficientnet import block

import windows_features_generator as wg

# --- Parameters ---
duration = 30  # seconds
sampling_rate = 1000  # Hz
samples = duration * sampling_rate


neuron_count = 50
neuron_params = dict(
    neuron_type=nengo.LIF(),
    intercepts=nengo.dists.Uniform(-1.0, 1.0),      # Wider range for input values
    max_rates=nengo.dists.Uniform(100, 300)         # Diverse firing rates
)

# --- Load Signals ---
nothing_signal = wg.extract_data_from_csv('/home/david/PycharmProjects/neengomorf/guisnn/nothing.csv', 60, 90)
man_signal = wg.extract_data_from_csv('/home/david/PycharmProjects/neengomorf/guisnn/man.csv', 60, 90)
car_signal = wg.extract_data_from_csv('/home/david/PycharmProjects/neengomorf/guisnn/car.csv', 60, 90)
test_signal = wg.extract_data_from_csv('/home/david/PycharmProjects/neengomorf/guisnn/nothing.csv', 60, 90)  # Testing with "man" signal

# Normalize all signals

# --- Nengo Model ---
model = nengo.Network()
with model:
    # Input nodes (1D signals over time)
    test_input = nengo.Node(lambda t: test_signal[int(t * sampling_rate)] if t < duration else 0)

    # Memory vectors (constant nodes with stored values)
    nothing_vector = nengo.Node(lambda t: nothing_signal[int(t * sampling_rate)] if t < duration else 0)
    man_vector = nengo.Node(lambda t: man_signal[int(t * sampling_rate)] if t < duration else 0)
    car_vector = nengo.Node(lambda t: car_signal[int(t * sampling_rate)] if t < duration else 0)

    # Input ensemble
    test_ens = nengo.Ensemble(neuron_count, 1, **neuron_params)
    nengo.Connection(test_input, test_ens)

    # Class signal ensembles (these are like template matchers)
    nothing_ens = nengo.Ensemble(neuron_count, 1, **neuron_params)
    man_ens = nengo.Ensemble(neuron_count, 1, **neuron_params)
    car_ens = nengo.Ensemble(neuron_count, 1, **neuron_params)

    nengo.Connection(nothing_vector, nothing_ens)
    nengo.Connection(man_vector, man_ens)
    nengo.Connection(car_vector, car_ens)

    # Similarity computation (dot product approximation)
    # We multiply matching signals (should be close if similar)
    nothing_sim = nengo.Ensemble(neuron_count, 1)
    man_sim = nengo.Ensemble(neuron_count, 1)
    car_sim = nengo.Ensemble(neuron_count, 1)

    nengo.Connection(test_ens, nothing_sim, transform=1)
    nengo.Connection(nothing_ens, nothing_sim, transform=1, function=lambda x: x)

    nengo.Connection(test_ens, man_sim, transform=1)
    nengo.Connection(man_ens, man_sim, transform=1, function=lambda x: x)

    nengo.Connection(test_ens, car_sim, transform=1)
    nengo.Connection(car_ens, car_sim, transform=1, function=lambda x: x)

    # Probes
    nothing_probe = nengo.Probe(nothing_sim, synapse=0.05)
    man_probe = nengo.Probe(man_sim, synapse=0.05)
    car_probe = nengo.Probe(car_sim, synapse=0.05)

    # Class output ensemble (receives similarity scores)
    class_output = nengo.Ensemble(neuron_count, 3)

    # Connect similarity distances into class output
    nengo.Connection(nothing_sim, class_output[0])
    nengo.Connection(man_sim, class_output[1])
    nengo.Connection(car_sim, class_output[2])

    # Probe the class output ensemble
    class_output_probe = nengo.Probe(class_output, synapse=0.05)


# --- Run the simulation ---
with nengo.Simulator(model) as sim:
    sim.run(duration)

# --- Plot Results ---
plt.figure(figsize=(10, 6))
plt.plot(sim.trange(), sim.data[nothing_probe], label='Similarity to Nothing')
plt.plot(sim.trange(), sim.data[man_probe], label='Similarity to Man')
plt.plot(sim.trange(), sim.data[car_probe], label='Similarity to Car')
plt.title("Similarity Scores Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Similarity (approximate)")
plt.legend()
plt.grid()
plt.show(block=False)

# Predicted class from class_output ensemble
class_output_data = sim.data[class_output_probe]
predicted_classes_from_ensemble = np.argmax(class_output_data, axis=1)

plt.figure(figsize=(10, 4))
plt.plot(sim.trange(), predicted_classes_from_ensemble, drawstyle='steps-post')
plt.yticks([0, 1, 2], ['Nothing', 'Man', 'Car'])
plt.xlabel("Time (s)")
plt.ylabel("Predicted Class")
plt.title("Classification from class_output Ensemble Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()


