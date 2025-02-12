using DrWatson # have to install it globally
quickactivate(@__DIR__) # needed to load project env

include("../src/SNN.jl")

using Symbolics, ModelingToolkit, DifferentialEquations, ComponentArrays, LinearAlgebra, LinearSolve


# gpu = to_device_fn()
gpu = x -> x

tspan = (0, 10)

# e_neurons_n = 5

# param_range = 2.0e-10:0.1e-10:2.6e-10
param_range = 1.4e-10:0.01e-10:1.5e-10
param_to_change = :a

for param_i in param_range
    # @time params = Neuron.AdExNeuronParams()
    @time params = SNN.Neuron.get_adex_neuron_params_skeleton(Float64)
    params.inc_gsyn = 7e-9
    params.a = 0.15e-9          # Subthreshold adaptation (A)
    params.b = 9e-12          # Spiking adaptation (A)
    params.TauW = 300.0e-3      # Adaptation time constant (s)
    params.Cm = 4.5e-10

    params.Ibase = 2.4e-10
    # params.Ibase = 0
    params.sigma = 2.5
    # make schedule
    stim_params = SNN.Params.get_stim_params_skeleton()
    # stim_params.n_trials = 20
    stim_params.amplitude = 2.2e-9
    stim_params.duration = 50.0e-3
    stim_params.deviant_idx = 0
    stim_params.standard_idx = 1
    stim_params.p_deviant = 0.01
    stim_params.start_offset = 2.5
    stim_params.isi = 300e-3

    params[param_to_change] = param_i

    stim_schedule = SNN.Params.generate_schedule(stim_params, tspan)
    # id_map struct = (id of neuron to map in connections, neuron)
    # connections shape = (n_neurons, n_neurons)
    con_mapping_nested = [
        (SNN.Params.@connect_neurons [1, 2] SNN.Neuron.AMPA() 3),
    ]
    con_mapping = reduce(vcat, con_mapping_nested)
    con_mapping_5_neurons = [
    # (1, 2, SNN.Neuron.AMPA()),
    # (2, 3, SNN.Neuron.AMPA()),
    # (5, 4, SNN.Neuron.AMPA()),
    # (4, 3, SNN.Neuron.AMPA()),
    ]
    # con_mapping = [
    #     (1, 3, SNN.Neuron.AMPA()),
    #     (2, 3, SNN.Neuron.AMPA())
    # ]
    #
    # pre_neurons = [row[1] for row in con_mapping]
    # post_neurons = [row[2] for row in con_mapping]
    # e_neurons_n = size(unique([pre_neurons; post_neurons]), 1)
    e_neurons_n = 1

    name = "base_1_adaptation_" * string(param_to_change) * "=" * string(param_i)
    out_path_prefix = "results/"
    (sol, simplified_model, prob) = SNN.Pipeline.run_exp(
        out_path_prefix, name;
        e_neurons_n=e_neurons_n,
        params=params,
        stim_params=stim_params,
        stim_schedule=stim_schedule,
        tspan=tspan,
        con_mapping=[],
        solver=DRI1(),
        tols=(1e-3, 1e-3)
    )
end
