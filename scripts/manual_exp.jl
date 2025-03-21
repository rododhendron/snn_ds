using DrWatson # have to install it globally
quickactivate(@__DIR__) # needed to load project env

include("../src/SNN.jl")

using Symbolics, ModelingToolkit, DifferentialEquations, ComponentArrays, LinearAlgebra, LinearSolve


# gpu = to_device_fn()
gpu = x -> x

tspan = (0, 30)

# e_neurons_n = 5

# make schedule
stim_params = SNN.Params.get_stim_params_skeleton()
# stim_params.n_trials = 20
stim_params.amplitude = 0.8e-9
stim_params.duration = 100.0e-3
stim_params.deviant_idx = 2
stim_params.standard_idx = 1
stim_params.p_deviant = 0.1

stim_schedule = SNN.Params.generate_schedule(stim_params, tspan)

sch_t = deepcopy(stim_schedule[1, :])
sch_onset = deepcopy(stim_schedule[2, :])
sch_group = deepcopy(stim_schedule[3, :])

# @time params = Neuron.AdExNeuronParams()
@time params = SNN.Neuron.get_adex_neuron_params_skeleton(Float64)#, sch_t, sch_onset, sch_group)
params.inc_gsyn_ampa = 6e-9
params.inc_gsyn_gabaa = 6e-9
params.inc_gsyn_gabab = 6e-9
params.a = 8.0e-9          # Subthreshold adaptation (A)
params.b = 3e-10          # Spiking adaptation (A)
params.TauW = 500.0e-3      # Adaptation time constant (s)
params.Cm = 4.5e-10

params.Ibase = 4e-10

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
pre_neurons = [row[1] for row in con_mapping]
post_neurons = [row[2] for row in con_mapping]
e_neurons_n = size(unique([pre_neurons; post_neurons]), 1)
# e_neurons_n = 2

name = "jitter_network_noise"
out_path_prefix = "results/"
(sol, simplified_model, prob) = SNN.Pipeline.run_exp(
    out_path_prefix, name;
    e_neurons_n=e_neurons_n,
    params=params,
    stim_params=stim_params,
    tspan=tspan,
    con_mapping=con_mapping,
    stim_schedule=stim_schedule,
    solver=ISSEulerHeun(),
    save_plots=false
)
