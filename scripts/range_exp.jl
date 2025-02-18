using DrWatson # have to install it globally
quickactivate(@__DIR__) # needed to load project env

include("../src/SNN.jl")

using Symbolics, ModelingToolkit, DifferentialEquations, ComponentArrays, LinearAlgebra, LinearSolve, Random
using AMDGPU
using Base.Threads

UID = randstring(6)


# gpu = SNN.Device.to_device_fn(; backend="amd")
gpu = x -> x
# gpu = ROCArray

tspan = (0, 200)

# e_neurons_n = 5

# param_range = 2.0e-10:0.1e-10:2.6e-10
param_a_range = 0.0:0.1e-10:3.0e-10
param_b_range = 0.0:0.2:3.0
param_to_change_a = :TauW
param_to_change_b = :sigma

@show exp_size = length(param_a_range) * length(param_b_range)

csis = zeros(Float64, (length(param_a_range), length(param_b_range)))

# make schedule
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
params.inc_gsyn = 7e-9
params.a = 1.5e-9          # Subthreshold adaptation (A)
params.b = 4.4e-12          # Spiking adaptation (A)
params.TauW = 600.0e-3      # Adaptation time constant (s)
params.Cm = 4.5e-10

params.Ibase = 2.4e-10
# params.Ibase = 0
params.sigma = 2.0

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

(sol, simplified_model, prob, csi) = SNN.Pipeline.run_exp(
    "tmp/", "tmp";
    e_neurons_n=e_neurons_n,
    params=params,
    stim_params=stim_params,
    stim_schedule=stim_schedule,
    tspan=tspan,
    con_mapping=con_mapping,
    solver=DRI1(),
    tols=(1e-3, 1e-3),
    fetch_csi=true,
    to_device=gpu,
    l=l
)
pb_len = length(param_b_range)
pa_len = length(param_a_range)

for i in 1:pa_len
    Threads.@threads :greedy for j in 1:pb_len
        new_pars = deepcopy(params)
        new_pars[param_to_change_a] = param_i = param_a_range[i]
        new_pars[param_to_change_b] = param_j = param_b_range[j]

        name = "base_3_adaptation_" * string(param_to_change_a) * "=" * string(param_i) * "_" * string(param_to_change_b) * "=" * string(param_j)
        out_path_prefix = "results/$(UID)/"
        (sol_i, simplified_model_i, prob_i, csi_i) = SNN.Pipeline.run_exp(
            out_path_prefix, name;
            e_neurons_n=e_neurons_n,
            params=new_pars,
            stim_params=stim_params,
            stim_schedule=stim_schedule,
            tspan=tspan,
            con_mapping=con_mapping,
            solver=DRI1(),
            tols=(1e-3, 1e-3),
            fetch_csi=true,
            to_device=gpu,
            remake_prob=prob,
            model=simplified_model,
            l=l
        )
        csis[i, j] = csi_i
        @show param_i
        @show param_j
        @show csi_i
    end
end
heatmap_values = (collect(param_a_range), collect(param_b_range), csis)
@show heatmap_values
SNN.Plots.plot_heatmap(
    heatmap_values,
    title="csi over params search", name="results/$(UID)/base_3_adaptation_scan_$(string(param_to_change_a))_$(string(param_to_change_b)).png", tofile=true, xlabel=String(param_to_change_a), ylabel=String(param_to_change_b)
)
