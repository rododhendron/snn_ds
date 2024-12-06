using DrWatson # have to install it globally
quickactivate(@__DIR__) # needed to load project env

include("../src/neuron.jl")
include("../src/utils.jl")
include("../src/plots.jl")

using Symbolics, ModelingToolkit, DifferentialEquations
using CUDA

using .Neuron
using .Utils, .Plots

tspan = (0, 1)
n_neurons = 10
i_neurons_n = 20
e_neurons_n = 80

# @time params = Neuron.AdExNeuronParams()
@time params = Neuron.get_adex_neuron_params_skeleton(Float64)
@time i_neurons = [Neuron.make_neuron(params, Neuron.Soma, tspan, Symbol("i_neuron_$(i)")) for i in 1:i_neurons_n]
@time e_neurons = [Neuron.make_neuron(params, Neuron.Soma, tspan, Symbol("e_neuron_$(i)")) for i in 1:e_neurons_n]

ee_rule = Neuron.ConnectionRule(Neuron.excitator, Neuron.excitator, Neuron.AMPA(), 0.05)
ei_rule = Neuron.ConnectionRule(Neuron.excitator, Neuron.inhibitor, Neuron.AMPA(), 0.05)
ie_rule = Neuron.ConnectionRule(Neuron.inhibitor, Neuron.excitator, Neuron.GABAa(), 0.05)
ii_rule = Neuron.ConnectionRule(Neuron.inhibitor, Neuron.inhibitor, Neuron.AMPA(), 0.05)


(id_map, map_connect) = Neuron.init_connection_map(e_neurons, i_neurons, vcat(ee_rule, ei_rule, ie_rule))
connections = Neuron.instantiate_connections(id_map, map_connect, vcat(e_neurons, i_neurons))

@time network = Neuron.make_network(vcat(e_neurons, i_neurons), connections)

simplified_model = network
# @time simplified_model = Neuron.structural_simplify(network)

# infere params
@time uparams = Neuron.get_adex_neuron_uparams_skeleton(Float64)

@time iparams, iuparams = Neuron.map_params(simplified_model, params, uparams; match_nums=true)

# a = [iparams[1:end-1]; simplified_model.neuron_3.soma.input_value => 1e-9]
# resolve
@time prob1 = ODEProblem(simplified_model, iuparams |> CuArray, tspan, iparams |> CuArray)
prob = Neuron.make_gpu_compatible(prob1)

sol = solve(prob, Vern6(); abstol=1e-6, reltol=1e-6)

tree = Utils.make_param_tree(network)
res = Utils.fetch_tree(["e_neuron", "R"], tree)
ris = Utils.fetch_tree(["i_neuron", "R"], tree)

mes = reduce(hcat, sol[res])
mis = reduce(hcat, sol[ris])

spikes_e = Utils.get_spike_timings(mes, sol)
spikes_i = Utils.get_spike_timings(mis, sol)

(start, stop) = tspan

name_prefix = "network_AdEx_neurons__e=$e_neurons_n,i=$i_neurons_n,ampa_gamaa"
name_interpol(name) = name_prefix*name

Plots.plot_spikes(spikes_e; start=start, stop=stop, color=:red, title="excitator spikes", name=name_interpol("excitators.png"))
Plots.plot_spikes(spikes_i; start=start, stop=stop, color=:red, title="inhibitor spikes", name=name_interpol("inhibitors.png"))
