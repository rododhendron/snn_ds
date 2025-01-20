using DrWatson # have to install it globally
quickactivate(@__DIR__) # needed to load project env

include("../src/SNN.jl")

using Symbolics, ModelingToolkit, DifferentialEquations, ComponentArrays, LinearAlgebra, LinearSolve


# gpu = to_device_fn()
gpu = x -> x

tspan = (0, 10)

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

# id_map struct = (id of neuron to map in connections, neuron)
# connections shape = (n_neurons, n_neurons)
con_mapping = [
    (1, 3, SNN.Neuron.AMPA()),
    (2, 3, SNN.Neuron.AMPA())
]
out_path = "results/3neurons_ssa/"
mkpath(out_path)
name = "3neurons_ssa"
(sol, simplified_model, prob) = SNN.Pipeline.run_exp(
    out_path * name;
    e_neurons_n=e_neurons_n,
    params=params,
    stim_params=stim_params,
    tspan=tspan,
    con_mapping=con_mapping,
)
