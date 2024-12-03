using DrWatson # have to install it globally
quickactivate(@__DIR__) # needed to load project env

include("../src/neuron.jl")
include("../src/utils.jl")

using Symbolics, ModelingToolkit, DifferentialEquations

using .Neuron
using .Utils

tspan = (0, 1)
n_neurons = 10
i_neurons = 3
e_neurons = 3

@time params = Neuron.AdExNeuronParams(; input_value=1e-9)
@time i_neurons = [Neuron.make_neuron(params, Neuron.Soma, tspan, Symbol("i_neuron_$(i)")) for i in 1:i_neurons]
@time e_neurons = [Neuron.make_neuron(params, Neuron.Soma, tspan, Symbol("e_neuron_$(i)")) for i in 1:e_neurons]

ee_rule = Neuron.ConnectionRule(Neuron.excitator, Neuron.excitator, Neuron.AMPA(), 1.0)
ei_rule = Neuron.ConnectionRule(Neuron.excitator, Neuron.inhibitor, Neuron.AMPA(), 1.0)
ie_rule = Neuron.ConnectionRule(Neuron.inhibitor, Neuron.excitator, Neuron.GABAa(), 1.0)


(id_map, map_connect) = Neuron.init_connection_map(e_neurons, i_neurons, vcat(ee_rule, ei_rule, ie_rule))
connections = Neuron.instantiate_connections(id_map, map_connect, vcat(e_neurons, i_neurons))

@time network = Neuron.make_network(vcat(e_neurons, i_neurons), connections)

simplified_model = network
# @time simplified_model = Neuron.structural_simplify(network)

# infere params
@time uparams = Neuron.AdExNeuronUParams()

@time iparams, iuparams = Neuron.map_params(simplified_model, params, uparams)

@show iparams
# a = [iparams[1:end-1]; simplified_model.neuron_3.soma.input_value => 1e-9]
# resolve
@time prob = ODEProblem(simplified_model, iuparams, tspan, iparams)

sol = solve(prob, Vern6(); abstol=1e-6, reltol=1e-6)

tree = Utils.make_param_tree(simplified_model)

path = ["e_neuron_1", "R"]
fetched = Utils.fetch_tree(path, tree)
