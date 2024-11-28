using DrWatson # have to install it globally
quickactivate(@__DIR__) # needed to load project env

include("../src/neuron.jl")

using Symbolics, ModelingToolkit, DifferentialEquations

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

a = [iparams[1:end-1]; simplified_model.neuron_3.soma.input_value => 1e-9]
# resolve
prob = ODEProblem(simplified_model, iuparams, tspan, a)
sol = solve(prob, Vern6(); abstol=1e-7, reltol=1e-7)
