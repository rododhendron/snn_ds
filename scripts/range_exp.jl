using DrWatson # have to install it globally
quickactivate(@__DIR__) # needed to load project env

@everywhere include("../src/SNN.jl")

using Symbolics, ModelingToolkit, ComponentArrays, LinearAlgebra, LinearSolve, Random
using ProgressMeter
using Base.Threads, Distributed, SharedArrays
@everywhere using DifferentialEquations

UID_g = randstring(6)


# gpu = SNN.Device.to_device_fn(; backend="amd")
gpu = x -> x
# gpu = ROCArray

@everywhere const tspan = (0, 100)

# e_neurons_n = 5

# a & b
@everywhere const param_a_range = 1.0e-10:2.0e-10:6.0e-9
@everywhere const param_b_range = 2.0e-12:2e-12:9.0e-11

# param_to_change_a = :a
@everywhere const param_to_change_b = :b

# @everywhere const param_a_range = 0.01:0.05:2.0
# param_b_range = 0.1:0.1:3.0
@everywhere const param_to_change_a = :a
# param_to_change_b = :sigma

@show exp_size = length(param_a_range) * length(param_b_range)

@everywhere begin
    # make schedule
    UID = $UID_g
    stim_params = SNN.Params.get_stim_params_skeleton()
    # stim_params.n_trials = 20
    stim_params.amplitude = 1.6e-9
    stim_params.duration = 50.0e-3
    stim_params.deviant_idx = 2
    stim_params.standard_idx = 1
    stim_params.p_deviant = 0.15
    stim_params.start_offset = 2.5
    stim_params.isi = 300e-3
    stim_schedule = SNN.Params.generate_schedule(stim_params, tspan)

    @time params = SNN.Neuron.get_adex_neuron_params_skeleton(Float64)
    params.inc_gsyn = 8e-9
    params.a = 1.5e-9          # Subthreshold adaptation (A)
    params.b = 4.4e-12          # Spiking adaptation (A)
    params.TauW = 600.0e-3      # Adaptation time constant (s)
    params.Cm = 4.5e-10

    params.Ibase = 2.4e-10
    # params.Ibase = 0
    params.sigma = 1.2

    con_mapping_nested = [
        (SNN.Params.@connect_neurons [1, 2] SNN.Neuron.AMPA() 3),
    ]
    con_mapping = reduce(vcat, con_mapping_nested)

    pre_neurons = [row[1] for row in con_mapping]
    post_neurons = [row[2] for row in con_mapping]
    e_neurons_n = size(unique([pre_neurons; post_neurons]), 1)
    # e_neurons_n = 1
    #
    l = ReentrantLock()
    tols = (1e-3, 1e-3)

    solver = DRI1()
end
# solver = ISSEulerHeun()
# solver = SKenCarp()
# solver = WangLi3SMil_B()

(sol, simplified_model, prob, csi, neurons) = SNN.Pipeline.run_exp(
    "tmp/", "tmp";
    e_neurons_n=e_neurons_n,
    params=params,
    stim_params=stim_params,
    stim_schedule=stim_schedule,
    tspan=tspan,
    con_mapping=con_mapping,
    solver=solver,
    tols=tols,
    fetch_csi=true,
    to_device=gpu,
    # l=l
)
pb_len = length(param_b_range)
pa_len = length(param_a_range)

Random.seed!(1235)
indices = [(i, j) for i in 1:pa_len, j in 1:pb_len]

@everywhere function run_model!(
    params,
    e_neurons_n,
    indices,
    param_a_range,
    param_b_range,
    stim_params,
    stim_schedule,
    tspan,
    con_mapping,
    tols
)
    (i, j) = indices
    new_pars = deepcopy(params)
    new_pars[param_to_change_a] = param_i = param_a_range[i]
    new_pars[param_to_change_b] = param_j = param_b_range[j]

    name = "base_3_adaptation_" * string(param_to_change_a) * "=" * string(param_i) * "_" * string(param_to_change_b) * "=" * string(param_j)
    out_path_prefix = "results/$(UID)/"
    csi_i = SNN.Pipeline.run_exp(
        out_path_prefix, name;
        e_neurons_n=e_neurons_n,
        params=new_pars,
        stim_params=stim_params,
        stim_schedule=stim_schedule,
        tspan=tspan,
        con_mapping=con_mapping,
        solver=solver,
        tols=tols,
        fetch_csi=true,
        nout=true
        # to_device=gpu,
        # remake_prob=new_prob,
        # model=simplified_model,
        # l=l,
        # neurons=neurons
    )
    # @show csi_i
    csi_i
end

@everywhere run_model_for_ij!(indices) = run_model!(
    params,
    e_neurons_n,
    indices,
    param_a_range,
    param_b_range,
    stim_params,
    stim_schedule,
    tspan,
    con_mapping,
    tols
)

returns = @showprogress pmap(row -> run_model_for_ij!(row), indices; retry_delays=ones(4))
@show returns

heatmap_values = (collect(param_a_range), collect(param_b_range), returns)
@show heatmap_values
SNN.Plots.plot_heatmap(
    heatmap_values,
    title="csi over params search", name="results/$(UID)/base_3_adaptation_scan_$(string(param_to_change_a))_$(string(param_to_change_b)).png", tofile=true, xlabel=String(param_to_change_a), ylabel=String(param_to_change_b)
)
