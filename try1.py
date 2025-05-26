import nengo
import numpy as np
import matplotlib.pyplot as plt


def product(x):
    return x[0] * x[1]


model = nengo.Network()
with model:
    my_ensemble = nengo.Ensemble(n_neurons=40, dimensions=1)
    sin_node = nengo.Node(output=np.sin)
    cos_node = nengo.Node(output=np.cos)
    two_d_ensemble = nengo.Ensemble(n_neurons=80, dimensions=2)
    nengo.Connection(sin_node, two_d_ensemble[0])
    nengo.Connection(my_ensemble, two_d_ensemble[1])
    nengo.Connection(cos_node, my_ensemble)
    square = nengo.Ensemble(n_neurons=40, dimensions=1)
    nengo.Connection(my_ensemble, square, function=np.square)

    product_ensemble = nengo.Ensemble(n_neurons=40, dimensions=1)
    nengo.Connection(two_d_ensemble, product_ensemble, function=product)

    two_d_probe = nengo.Probe(two_d_ensemble, synapse=0.01)
    product_probe = nengo.Probe(product_ensemble, synapse=0.01)


sim = nengo.Simulator(model)
sim.run(5.0)

print(sim.data[product_probe][-10:])