using DrWatson # have to install it globally
quickactivate(@__DIR__) # needed to load project env

include("../src/neuron.jl")

using Symbolics, ModelingToolkit

using .Neuron

tspan = (0, 1)
n_neurons = 3
i_neurons = 2
e_neurons = 8

params = Neuron.AdExNeuronParams()
neurons = [neuron = Neuron.make_neuron(params, Neuron.Soma, tspan, Symbol("neuron_$(i)")) for i in 1:n_neurons]

network = Neuron.make_network(neurons)

simplified_model = Neuron.structural_simplify(network)

# infere params
uparams = Neuron.AdExNeuronUParams()

iparams, iuparams = Neuron.map_params(simplified_model, params, uparams)

# resolve
prob = ODEProblem(simplified_model, iuparams, tspan, iparams)
sol = solve(prob, Vern6(); abstol=1e-9, reltol=1e-9)
