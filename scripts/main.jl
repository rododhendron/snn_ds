using DrWatson # have to install it globally
quickactivate(@__DIR__) # needed to load project env

include("../src/neuron.jl")

using Symbolics

using .Neuron

tspan = (0, 1)

neuron = Neuron.make_neuron(nothing, Neuron.Soma, nothing, tspan)
params = Neuron.AdExNeuronParams()

Neuron.get_synapse(Neuron.SIMPLE_E(), params)
