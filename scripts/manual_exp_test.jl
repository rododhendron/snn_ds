using DrWatson # have to install it globally
quickactivate(@__DIR__) # needed to load project env

include("../src/SNN.jl")

using Symbolics, ModelingToolkit, DifferentialEquations, ComponentArrays, LinearAlgebra, LinearSolve, Random


# gpu = to_device_fn()
gpu = x -> x

tspan = (0, 3)

# e_neurons_n = 5

# make schedule
stim_params = SNN.Params.get_stim_params_skeleton()
# stim_params.n_trials = 20
stim_params.amplitude = 0.3e-9
stim_params.duration = 50.0e-3
stim_params.deviant_idx = 0
stim_params.standard_idx = 1
stim_params.p_deviant = 0.1

stim_schedule = SNN.Params.generate_schedule(stim_params, tspan)

sch_t = deepcopy(stim_schedule[1, :])
sch_onset = deepcopy(stim_schedule[2, :])
sch_group = deepcopy(stim_schedule[3, :])

# @time params = Neuron.AdExNeuronParams()
@time params = SNN.Neuron.get_adex_neuron_params_skeleton(Float64)#, sch_t, sch_onset, sch_group)
params.inc_gsyn = 40e-9
params.a = 5.0e-8          # Subthreshold adaptation (A)
params.b = 0.2e-9          # Spiking adaptation (A)
params.TauW = 500.0e-3      # Adaptation time constant (s)
params.Cm = 4.5e-10

params.Ibase = 4.7e-10

# params.sch_t = sch_t
# params.sch_onset = sch_onset
# params.sch_group = sch_group
# id_map struct = (id of neuron to map in connections, neuron)
# connections shape = (n_neurons, n_neurons)
# con_mapping_nested = [
#     (SNN.Params.@connect_neurons [1, 2] SNN.Neuron.AMPA() 3),
# ]
# con_mapping = reduce(vcat, con_mapping_nested)
# con_mapping_5_neurons = [
#     (1, 2, SNN.Neuron.AMPA()),
#     (2, 3, SNN.Neuron.AMPA()),
#     (5, 4, SNN.Neuron.AMPA()),
#     (4, 3, SNN.Neuron.AMPA()),
# ]
con_mapping = [
    (1, 3, SNN.Neuron.AMPA()),
    (2, 3, SNN.Neuron.AMPA())
]
#
# pre_neurons = [row[1] for row in con_mapping]
# post_neurons = [row[2] for row in con_mapping]
# e_neurons_n = size(unique([pre_neurons; post_neurons]), 1)
e_neurons_n = 2

name = "jitter_network_noise"
path_prefix = "results/"

Random.seed!(1234)
path = path_prefix * name * "/"
mkpath(path)
exp_name = path * name

@time e_neurons = [SNN.Neuron.make_neuron(params, SNN.Neuron.Soma, tspan, Symbol("e_neuron_$(i)"), stim_schedule) for i in 1:e_neurons_n]

(input_neurons, rules) = SNN.Params.update_neurons_rules_from_sequence(e_neurons, stim_params, params)
(input_grp, input_neuron_vec) = zip(input_neurons...)
input_neurons_name = [neuron.name for neuron in [input_neuron_vec...;]]
overriden_params = SNN.Params.override_params(params, rules)
(id_map, map_connect) = SNN.Neuron.infer_connection_from_map(e_neurons, [])
connections = SNN.Neuron.instantiate_connections(id_map, map_connect, e_neurons)

@time network = SNN.Neuron.make_network(e_neurons, connections)

# @named noise_network = System([network; noise_eqs])
# @time simplified_ode = structural_simplify(network; split=true, conservative=true)

# add noise
noise_eqs = SNN.Neuron.instantiate_noise(network, e_neurons, 0.001)

@named noise_network = SDESystem(network, noise_eqs, continuous_events=continuous_events(network), observed=observed(network))


# @time simplified_model = structural_simplify(noise_network; split=true)
@time simplified_model = structural_simplify(noise_network)

# ModelingToolkit.continuous_events(simplified_model) = simplified_model.continuous_callback
# infere params
@time uparams = SNN.Neuron.get_adex_neuron_uparams_skeleton(Float64)

@time iparams, iuparams = SNN.Neuron.map_params(simplified_model, overriden_params, uparams; match_nums=true)


ModelingToolkit.get_continuous_events(sys::SDESystem) = [sys.continuous_events...]

contin_cb = ModelingToolkit.generate_rootfinding_callback([simplified_model.continuous_events...], simplified_model, unknowns(simplified_model), parameters(simplified_model))
cb = ModelingToolkit.merge_cb(contin_cb, nothing) # 2nd arg is placeholder for discrete callback

@time prob = SDEProblem(simplified_model, iuparams, tspan, iparams, cb=cb)#, sparse=true)

println("Solving...")
# @time sol = solve(prob, KenCarp47(linsolve=KrylovJL_GMRES()); abstol=1e-4, reltol=1e-4, dtmax=1e-3)
# @time sol = solve(prob, SOSRA(); abstol=1e-3, reltol=1e-3, dtmax=1e-3)
# @time sol = solve(prob, SKenCarp(); abstol=1e-3, reltol=1e-3, dtmax=1e-3)
# @time sol = solve(prob, ImplicitRKMil(), abstol=1e-2, reltol=-1e-2, dtmax=1e-3)
@time sol = solve(prob, SKenCarp(), abstol=1e-4, reltol=1e-4, dtmax=1e-3)

@time tree::SNN.Utils.ParamTree = SNN.Utils.make_param_tree(simplified_model)

(start, stop) = tspan

name_prefix = exp_name * ""
name_interpol(name) = name_prefix * "_" * name

@time SNN.Plots.plot_excitator_value(1, sol, start, stop, name_interpol, tree, stim_params.start_offset, stim_schedule)
