using Base: SpawnIO
using DrWatson # have to install it globally
quickactivate(@__DIR__) # needed to load project env

using ParallelProcessingTools
include("../src/SNN.jl")

using Symbolics, ModelingToolkit, ComponentArrays, LinearAlgebra, LinearSolve, Random
using Base.Threads
using Distributed
using SharedArrays
using DifferentialEquations, ProgressMeter
using JLD2
using CairoMakie

# Add some workers and initialize with all `@always_everywhere` code:
old_nprocs = nprocs()
const N = 10
const max_proc = old_nprocs + N
# _, n = runworkers(OnLocalhost(; n=N))
cluster = ppt_cluster_manager()

pool = FlexWorkerPool(; withmyid=true, init_workers=true)

addprocs(N; exeflags="-L src/SNN.jl")
@onprocs workers() include("src/SNN.jl")

push!.(Ref(pool), workers())
import ThreadPinning

# force pseudorandom reviewing protocols OK
# ressortir les plots avec negatifs OK
# checker si kern and chao mentionnent brette OK et oui
# prendre les standards précédents au deviant OK normalement, rechecker pour toutes les fonctions mais CSI ok
# taille csi ? -> pas encore dynamic bin

display(pool)
display(worker_resources())
ThreadPinning.distributed_pinthreads(:numa)
# @show ThreadPinning.distributed_getcpuids()

UID_g = randstring(6)

# Function to monitor and kill processes

# gpu = SNN.Device.to_device_fn(; backend="amd")
gpu = x -> x
# gpu = ROCArray
tspan = (0, 100)
stim_params = SNN.Params.get_stim_params_skeleton()
# stim_params.n_trials = 20
stim_params.amplitude = 1.6e-9
stim_params.duration = 50.0e-3
stim_params.deviant_idx = 2
stim_params.standard_idx = 1
stim_params.p_deviant = 0.15
stim_params.start_offset = 2.5
stim_params.isi = 300e-3
stim_schedule = SNN.Params.generate_schedule(stim_params, tspan; is_pseudo_random=true)

@always_everywhere begin #let SNN = SNN
    using ParallelProcessingTools, Distributed
    using DifferentialEquations, ProgressMeter
    using Statistics

    tspan = $tspan
    stim_params = $stim_params
    stim_schedule = $stim_schedule


    function monitor_and_kill(memory_limit)
        pid = Sys.getpid()  # Get the current process ID
        # Use a system command to check memory usage (example for Linux)
        mem_usage = parse(Int, read(`ps -o rss= -p $pid`, String))  # Memory in KB
        # println("mem usage: $(mem_usage) / $(memory_limit) on worker $pid")
        # println(mem_usage)
        if mem_usage > memory_limit
            println("Killing worker (PID: $pid) for exceeding memory limit")
            @fetchfrom 1 rmprocs(pid)  # Remove the worker from the pool
            exit()
            println("killed $pid")
        end
    end

    memlimit = 7 * 1000^2
    # Optional: Set a custom memory limit for worker processes:
    Distributed.myid() != 1 && memlimit

    # e_neurons_n = 5

    # a & b
    param_b_range = 1.0e-10:4.0e-10:6.0e-9
    # param_b_range = 2.0e-12:4e-12:9.0e-11

    # param_a_range = 1.0e-9:0.5e-9:18.0e-9
    # param_b_range = 50:50:1200

    # param_to_change_a = :a
    # param_to_change_b = :b
    param_a_range = 0.001:0.01:0.5
    param_to_change_a = :sigma

    # param_a_range = 0.01:0.05:3.0
    # param_b_range = 0.1:0.1:3.0
    # param_to_change_a = :a
    param_to_change_b = :a
    # param_b_range = 0.1:0.1:3.0
    # param_to_change_a = :a
    # param_b_range = 1.0e-12:0.1e-11:10.0e-11
    # param_to_change_b = :b
    # param_to_change_b = :sigma

    # make schedule
    UID = $UID_g

    @time params = SNN.Neuron.get_adex_neuron_params_skeleton(Float64)
    params.inc_gsyn = 8e-9
    params.a = 1.5e-9          # Subthreshold adaptation (A)
    params.b = 4.4e-12          # Spiking adaptation (A)
    params.TauW = 600.0e-3      # Adaptation time constant (s)
    params.Cm = 4.5e-10

    params.Ibase = 1.0e-10
    params.sigma = 1.0

    con_mapping_nested = [
        (SNN.Params.@connect_neurons [1, 2] SNN.Neuron.AMPA() 3),
    ]
    con_mapping::Array{Tuple{Int64,Int64,SNN.Neuron.SynapseType},1} = reduce(vcat, con_mapping_nested)

    pre_neurons = [row[1] for row in con_mapping]
    post_neurons = [row[2] for row in con_mapping]
    e_neurons_n = size(unique([pre_neurons; post_neurons]), 1)
    # e_neurons_n = 1
    #
    l = ReentrantLock()
    # tols = (1e-2, 1e-2)
    tols = (2e-3, 2e-3)

    solver = DRI1()
    # solver = EulerHeun()

    run_model_for_ij!(indices) = run_model!(
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

    function run_model!(
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
        monitor_and_kill(memlimit)
        (i, j) = indices
        new_pars = deepcopy(params)
        if isnothing(j)
            new_pars[param_to_change_a] = param_i = param_a_range[i]
            name = "base_3_adaptation_" * string(param_to_change_a) * "=" * string(param_i)
        else
            new_pars[param_to_change_a] = param_i = param_a_range[i]
            new_pars[param_to_change_b] = param_j = param_b_range[j]
            name = "base_3_adaptation_" * string(param_to_change_a) * "=" * string(param_i) * "_" * string(param_to_change_b) * "=" * string(param_j)
        end

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
end
# solver = ISSEulerHeun()
# solver = SKenCarp()
# solver = WangLi3SMil_B()

ensure_procinit()
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
    # to_device=gpu,
    # l=l
)
pb_len = length(param_b_range)
pa_len = length(param_a_range)

Random.seed!(1234)
#indices = [(i, j) for i in 1:pa_len, j in 1:pb_len]
indices = [(i, nothing) for i in 1:pa_len]

function spawn_workers()
    println("try spawning...")
    workers_to_launch = max_proc - nprocs()
    if workers_to_launch > 0
        ps = addprocs(workers_to_launch)
        @onprocs ps include("src/SNN.jl")
        ensure_procinit(ps)
        push!.(Ref(pool), ps)
        println("added workers $(ps)")
    end
    sleep(5)
    spawn_workers()
end
Threads.@spawn spawn_workers()


@show pool
returns = @showprogress pmap(row -> run_model_for_ij!(row), pool, indices; retry_delays=ones(4))
# returns = @showprogress @onprocs(row -> run_model_for_ij!(row), indices; retry_delays=ones(4))
@show returns

@save "csis_results.jld2" returns


for k in keys(returns[1])
    if occursin("csi", k)
        values = get.(returns, k, nothing)
        if length(size(values)) == 2
            heatmap_values = (collect(param_a_range), collect(param_b_range), values)
            SNN.Plots.plot_heatmap(
                heatmap_values,
                title="csi over params search", name="results/$(UID)/base_3_adaptation_scan_$(string(param_to_change_a))_$(string(param_to_change_b))_$k.png", tofile=true, xlabel=String(param_to_change_a), ylabel=String(param_to_change_b)
            )
        elseif length(size(values)) == 1
            SNN.Plots.plot_xy(
                collect(param_a_range),
                values,
                title="csi over params search", name="results/$(UID)/base_3_adaptation_scan_$(string(param_to_change_a))_$k.png", tofile=true, xlabel=String(param_to_change_a), ylabel=k
            )
        end
    end
end
