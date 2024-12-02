using DrWatson # have to install it globally
quickactivate(@__DIR__) # needed to load project env

include("../src/neuron.jl")

using Symbolics, ModelingToolkit, DifferentialEquations

using .Neuron

tspan = (0, 1)
n_neurons = 10
i_neurons = 2
e_neurons = 8

@time params = Neuron.AdExNeuronParams(; input_value=1e-9)
@time i_neurons = [Neuron.make_neuron(params, Neuron.Soma, tspan, Symbol("i_neuron_$(i)")) for i in 1:i_neurons]
@time e_neurons = [Neuron.make_neuron(params, Neuron.Soma, tspan, Symbol("e_neuron_$(i)")) for i in 1:e_neurons]

ee_connections = Neuron.connect_many_neurons(e_neurons, e_neurons, Neuron.AMPA(), 0.5)
ei_connections = Neuron.connect_many_neurons(e_neurons, i_neurons, Neuron.AMPA(), 0.5)
ie_connections = Neuron.connect_many_neurons(e_neurons, i_neurons, Neuron.GABAa(), 1.0)

@time network = Neuron.make_network(vcat(e_neurons, i_neurons), vcat(ee_connections, ei_connections, ie_connections))

simplified_model = network
# @time simplified_model = Neuron.structural_simplify(network)

# infere params
@time uparams = Neuron.AdExNeuronUParams()

@time iparams, iuparams = Neuron.map_params(simplified_model, params, uparams)

# a = [iparams[1:end-1]; simplified_model.neuron_3.soma.input_value => 1e-9]
# resolve
@time prob = ODEProblem(simplified_model, iuparams, tspan, iparams)

sol = solve(prob, Vern6(); abstol=1e-4, reltol=1e-4)
