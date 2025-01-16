using DrWatson # have to install it globally
quickactivate(@__DIR__) # needed to load project env

include("../src/SNN.jl")

using Symbolics, ModelingToolkit, DifferentialEquations, ComponentArrays, LinearAlgebra, LinearSolve


# gpu = to_device_fn()
gpu = x -> x

tspan = (0, 5)

e_neurons_n = 3

# @time params = Neuron.AdExNeuronParams()
@time params = SNN.Neuron.get_adex_neuron_params_skeleton(Float64)
params.inc_gsyn = 60e-9
params.a = 120.0e-9          # Subthreshold adaptation (A)
params.b = 10e-9          # Spiking adaptation (A)
params.TauW = 644.0e-3      # Adaptation time constant (s)
# make schedule
stim_params = SNN.Params.get_stim_params_skeleton()
# stim_params.n_trials = 20
stim_params.amplitude = 0.50e-9
stim_params.duration = 50.0e-3
schedule = SNN.Params.generate_schedule(stim_params)
# @time i_neurons = [SNN.Neuron.make_neuron(params, Neuron.Soma, tspan, Symbol("i_neuron_$(i)"), schedule) for i in 1:i_neurons_n]
@time e_neurons = [SNN.Neuron.make_neuron(params, SNN.Neuron.Soma, tspan, Symbol("e_neuron_$(i)"), schedule) for i in 1:e_neurons_n]

rules = SNN.Params.update_neurons_rules_from_sequence(e_neurons, stim_params, params)
overriden_params = SNN.Params.override_params(params, rules)

# ee_rule = SNN.Neuron.ConnectionRule(Neuron.excitator, Neuron.excitator, Neuron.AMPA(), 0.05)
# ei_rule = SNN.Neuron.ConnectionRule(Neuron.excitator, Neuron.inhibitor, Neuron.AMPA(), 0.05)
# ie_rule = SNN.Neuron.ConnectionRule(Neuron.inhibitor, Neuron.excitator, Neuron.GABAa(), 0.05)
# ii_rule = SNN.Neuron.ConnectionRule(Neuron.inhibitor, Neuron.inhibitor, Neuron.AMPA(), 0.05)

# id_map struct = (id of neuron to map in connections, neuron)
# connections shape = (n_neurons, n_neurons)
con_mapping = [
    (1, 3, SNN.Neuron.AMPA()),
    (2, 3, SNN.Neuron.AMPA())
]

(id_map, map_connect) = SNN.Neuron.infer_connection_from_map(e_neurons, con_mapping)
# (id_map, map_connect) = SNN.Neuron.init_connection_map(e_neurons, i_neurons, vcat(ee_rule, ei_rule, ie_rule))
connections = SNN.Neuron.instantiate_connections(id_map, map_connect, e_neurons)

@time network = SNN.Neuron.make_network(e_neurons, connections)

@time simplified_model = structural_simplify(network; split=false)
# @time simplified_model = Neuron.structural_simplify(network)
@time tree::SNN.Utils.ParamTree = SNN.Utils.make_param_tree(simplified_model)

# infere params
@time uparams = SNN.Neuron.get_adex_neuron_uparams_skeleton(Float64)

@time iparams, iuparams = SNN.Neuron.map_params(simplified_model, overriden_params, uparams; match_nums=true)

# a = [iparams[1:end-1]; simplified_model.neuron_3.soma.input_value => 1e-9]
# resolve
# ups = ModelingToolkit.varmap_to_vars(iuparams, unknowns(simplified_model))
# pps = ModelingToolkit.varmap_to_vars(iparams, parameters(simplified_model))
@time prob = ODEProblem{true}(simplified_model, iuparams, tspan, iparams, sparse=true)

println("Solving...")
# @time sol = solve(prob, ImplicitDeuflhardExtrapolation(threading=true); abstol=1e-3, reltol=1e-3)
@time sol = solve(prob, KenCarp47(linsolve=KrylovJL_GMRES()); abstol=1e-4, reltol=1e-4, dtmax=1e-3)
# @time sol = solve(prob, Rodas5P(); abstol=1e-6, reltol=1e-6)

@show sol.stats

println("tree fetch...")
@time res = SNN.Utils.fetch_tree(["e_neuron", "R"], tree)
# @time ris = SNN.Utils.fetch_tree(["i_neuron", "R"], tree)
@show res

println("hcat solutions")
@time mes = reduce(hcat, sol[res])
# @time mis = reduce(hcat, sol[ris])

println("get spikes timings")
@time spikes_e = SNN.Utils.get_spike_timings(mes, sol)
@show spikes_e
# @time spikes_i = SNN.Utils.get_spike_timings(mis, sol)

(start, stop) = tspan

name_prefix = "results/network_AdEx_neurons__e=$e_neurons_n,ampa__"
name_interpol(name) = name_prefix * name

println("plotting")


for i in 1:e_neurons_n
    @time SNN.Plots.plot_excitator_value(i, sol, start, stop, name_interpol, tree, stim_params.start_offset)
end

res = SNN.Utils.fetch_tree(["e_neuron", "R"], tree)
@show res
ma = SNN.Utils.hcat_sol_matrix(res, sol)
spikes_times = SNN.Utils.get_spike_timings(ma, sol)
SNN.Plots.plot_spikes((spikes_times, []); start=start, stop=stop, color=(:red, :blue), height=1500, title="Network activity", xlabel="time (in s)", ylabel="neuron index", name=name_interpol("rs.png"))
