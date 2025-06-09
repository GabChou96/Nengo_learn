import nengo
import numpy as np
import matplotlib.pyplot as plt
import windows_features_generator as wg

# Set random seed for reproducibility
np.random.seed(42)

# Generate 60 seconds of data for each class (1000 samples per second)
duration = 30  # seconds
sampling_rate = 1000  # Hz
samples = duration * sampling_rate

nothing_signal = wg.extract_data_from_csv('/home/david/PycharmProjects/neengomorf/guisnn/nothing_old.csv', 60, 90)
man_signal = wg.extract_data_from_csv('/home/david/PycharmProjects/neengomorf/guisnn/man_old.csv', 60, 90)
car_signal = wg.extract_data_from_csv('/home/david/PycharmProjects/neengomorf/guisnn/car_old.csv', 60, 90)

test_signal = wg.extract_data_from_csv('/home/david/PycharmProjects/neengomorf/guisnn/nothing_old.csv', 90, 120)

# Create a Nengo model
model = nengo.Network()
with model:
    # Input nodes for each class
    nothing_input = nengo.Node(lambda t: nothing_signal[int(t * sampling_rate)] if t < duration else 0)
    man_input = nengo.Node(lambda t: man_signal[int(t * sampling_rate)] if t < duration else 0)
    car_input = nengo.Node(lambda t: car_signal[int(t * sampling_rate)] if t < duration else 0)

    # Test input node (for classification)
    test_input = nengo.Node(lambda t: test_signal[int(t * sampling_rate)] if t < duration else 0)

    #neurons ensemble params
    neuron_count = 10
    neuron_params = dict(
        neuron_type=nengo.LIF(),
        #intercepts=nengo.dists.Uniform(-0.3, 0.3),
        intercepts=nengo.dists.Uniform(-0.9, 0.9),
        max_rates=nengo.dists.Uniform(200, 300)
    )
    nothing_ens = nengo.Ensemble(neuron_count, 1, **neuron_params)
    man_ens = nengo.Ensemble(neuron_count, 1, **neuron_params)
    car_ens = nengo.Ensemble(neuron_count, 1, **neuron_params)
    test_ens = nengo.Ensemble(neuron_count, 1, **neuron_params)  # New test ensemble
    # Connect inputs to ensembles
    nengo.Connection(nothing_input, nothing_ens)
    nengo.Connection(man_input, man_ens)
    nengo.Connection(car_input, car_ens)
    nengo.Connection(test_input, test_ens)  # New test input connection
    # Classifier ensemble (receives all neural activities)
    classifier_ens = nengo.Ensemble(neuron_count, 4, **neuron_params)
    nengo.Connection(test_ens, classifier_ens[0])  # Test activity
    nengo.Connection(nothing_ens, classifier_ens[1])
    nengo.Connection(man_ens, classifier_ens[2])
    nengo.Connection(car_ens, classifier_ens[3])
    # Probes to record activity
    nothing_probe = nengo.Probe(nothing_ens, synapse=0.01)
    man_probe = nengo.Probe(man_ens, synapse=0.01)
    car_probe = nengo.Probe(car_ens, synapse=0.01)
    test_probe = nengo.Probe(test_ens, synapse=0.01)
    classifier_probe = nengo.Probe(classifier_ens, synapse=0.01)  # Moved inside with model
# Run the simulation
with nengo.Simulator(model) as sim:
    sim.run(10)


# Function to extract number representating each class
def weighted_average(data, alpha=0.9):
    avg = np.zeros_like(data[0])
    for i in range(len(data)):
        avg = alpha * avg + (1 - alpha) * data[i]
    return avg


# Compute stable class representations
last_seconds = 2 * sampling_rate  # Last 2 seconds for stability
class_representations = {
    "Nothing": weighted_average(sim.data[nothing_probe][-last_seconds:]),
    "Man": weighted_average(sim.data[man_probe][-last_seconds:]),
    "Car": weighted_average(sim.data[car_probe][-last_seconds:])
}
print(class_representations)

def classify_test_signal_raw(sim):
    # Extract the last 2 seconds of raw neural activity for each class and test signal
    last_seconds = 4 * sampling_rate
    test_activity = sim.data[test_probe][-last_seconds:]
    nothing_activity = sim.data[nothing_probe][-last_seconds:]
    man_activity = sim.data[man_probe][-last_seconds:]
    car_activity = sim.data[car_probe][-last_seconds:]

    # Compute Euclidean distances based on raw neural activity (no normalization)
    distances = {
        "Nothing": np.linalg.norm(test_activity - nothing_activity),
        "Man": np.linalg.norm(test_activity - man_activity),
        "Car": np.linalg.norm(test_activity - car_activity),
    }

    print("Raw Euclidean Distances:", distances)  # Debugging
    return min(distances, key=distances.get)  # Return class with the smallest distance


# Run the classification
predicted_class = classify_test_signal_raw(sim)
print(f"Predicted class for test signal: {predicted_class}")

# Checking the last 2 seconds of neural activity for each class and test signal
plt.figure(figsize=(10, 5))
plt.plot(sim.trange()[-last_seconds:], sim.data[nothing_probe][-last_seconds:], label="Nothing Activity")
plt.plot(sim.trange()[-last_seconds:], sim.data[man_probe][-last_seconds:], label="Man Activity")
plt.plot(sim.trange()[-last_seconds:], sim.data[car_probe][-last_seconds:], label="Car Activity")
plt.plot(sim.trange()[-last_seconds:], sim.data[test_probe][-last_seconds:], label="Test Signal Activity", linestyle="dashed")
plt.xlabel("Time (s)")
plt.ylabel("Neural Activity")
plt.legend()
plt.title("Neural Activity for Class Comparison (Last 5 Seconds)")
plt.show(block=False)

# Plot classifier ensemble activity
plt.figure(figsize=(10, 5))
plt.plot(sim.trange(), sim.data[classifier_probe][:, 1], label="Nothing")
plt.plot(sim.trange(), sim.data[classifier_probe][:, 2], label="Man")
plt.plot(sim.trange(), sim.data[classifier_probe][:, 3], label="Car")
plt.plot(sim.trange(), sim.data[classifier_probe][:, 0], label="Test Input")
plt.xlabel("Time (s)")
plt.ylabel("Classifier Activity")
plt.title("Classifier Ensemble Activity (Last 5 Seconds)")
plt.legend()
plt.show()
