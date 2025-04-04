using DrWatson # have to install it globally
quickactivate(@__DIR__) # needed to load project env

include("../src/neuron.jl")
include("../src/utils.jl")
include("../src/plots.jl")
include("../src/device.jl")

using Symbolics, ModelingToolkit, DifferentialEquations, ComponentArrays, LinearAlgebra, LinearSolve

using SNN

# gpu = to_device_fn()
gpu = x -> x

tspan = (0, 1)
n_neurons = 10
i_neurons_n = floor(n_neurons * 0.8)
e_neurons_n = n_neurons - i_neurons_n


# @time params = Neuron.AdExNeuronParams()
@time params = SNN.Neuron.get_adex_neuron_params_skeleton(Float64)
# make schedule
stim_params = SNN.Params.get_stim_params_skeleton()
schedule = SNN.Params.generate_schedule()
@time i_neurons = [Neuron.make_neuron(params, Neuron.Soma, tspan, Symbol("i_neuron_$(i)"), schedule) for i in 1:i_neurons_n]
@time e_neurons = [Neuron.make_neuron(params, Neuron.Soma, tspan, Symbol("e_neuron_$(i)"), schedule) for i in 1:e_neurons_n]

ee_rule = Neuron.ConnectionRule(Neuron.excitator, Neuron.excitator, Neuron.AMPA(), 0.05)
ei_rule = Neuron.ConnectionRule(Neuron.excitator, Neuron.inhibitor, Neuron.AMPA(), 0.05)
ie_rule = Neuron.ConnectionRule(Neuron.inhibitor, Neuron.excitator, Neuron.GABAa(), 0.05)
ii_rule = Neuron.ConnectionRule(Neuron.inhibitor, Neuron.inhibitor, Neuron.AMPA(), 0.05)


(id_map, map_connect) = Neuron.init_connection_map(e_neurons, i_neurons, vcat(ee_rule, ei_rule, ie_rule))
connections = Neuron.instantiate_connections(id_map, map_connect, vcat(e_neurons, i_neurons))

@time network = Neuron.make_network(vcat(e_neurons, i_neurons), connections)

@time simplified_model = structural_simplify(network; split=false)
# @time simplified_model = Neuron.structural_simplify(network)

# infere params
@time uparams = Neuron.get_adex_neuron_uparams_skeleton(Float64)

@time iparams, iuparams = Neuron.map_params(simplified_model, params, uparams; match_nums=false)

# a = [iparams[1:end-1]; simplified_model.neuron_3.soma.input_value => 1e-9]
# resolve
# ups = ModelingToolkit.varmap_to_vars(iuparams, unknowns(simplified_model))
# pps = ModelingToolkit.varmap_to_vars(iparams, parameters(simplified_model))
@time prob = ODEProblem{true}(simplified_model, iuparams, tspan, iparams, jac=true, sparse=true)

println("Solving...")
# @time sol = solve(prob, ImplicitDeuflhardExtrapolation(threading=true); abstol=1e-3, reltol=1e-3)
@time sol = solve(prob, KenCarp47(linsolve=KrylovJL_GMRES()); abstol=1e-4, reltol=1e-4, dtmax=1e-3)
# @time sol = solve(prob, Rodas5P(); abstol=1e-6, reltol=1e-6)

println("tree fetch...")
@time tree::ParamTree = Utils.make_param_tree(simplified_model)
@time res = Utils.fetch_tree(["e_neuron", "R"], tree)
@time ris = Utils.fetch_tree(["i_neuron", "R"], tree)

println("hcat solutions")
@time mes = reduce(hcat, sol[res])
@time mis = reduce(hcat, sol[ris])

println("get spikes timings")
@time spikes_e = Utils.get_spike_timings(mes, sol)
@time spikes_i = Utils.get_spike_timings(mis, sol)

(start, stop) = tspan

name_prefix = "results/network_AdEx_neurons__e=$e_neurons_n,i=$i_neurons_n,ampa_gabaa"
name_interpol(name) = name_prefix * name

println("plotting")
@time Plots.plot_spikes(spikes_e; start=start, stop=stop, color=:red, title="excitator spikes", name=name_interpol("excitators.png"))
@time Plots.plot_spikes(spikes_i; start=start, stop=stop, color=:blue, title="inhibitor spikes", name=name_interpol("inhibitors.png"))


for i in 1:maximum([div(e_neurons_n, 5), 10])
    @time Plots.plot_excitator_value(i, sol, start, stop, name_interpol, tree)
end
