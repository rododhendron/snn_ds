using DrWatson, Test
@quickactivate "snn_ds"

# Here you include files using `srcdir`
# include(srcdir("file.jl"))
include("../src/SNN.jl")
# using SNN
using Random
using ModelingToolkit, LinearSolve, DifferentialEquations

using AMDGPU

# Run test suite
println("Starting tests")
ti = time()

@testset "SNN.jl" begin
    Random.seed!(1234)
    tspan = (0, 100)

    @time params = SNN.Neuron.get_adex_neuron_params_skeleton(Float64)
    # params.Ibase = 2.4e-10
    params.Ibase = 1.0e-10
    params.sigma = 0.5

    params.inc_gsyn = 8e-9
    params.a = 1.5e-9          # Subthreshold adaptation (A)
    params.b = 4.4e-12          # Spiking adaptation (A)
    params.TauW = 600.0e-3      # Adaptation time constant (s)
    params.Cm = 4.5e-10

    @time stim_params = SNN.Params.get_stim_params_skeleton()
    stim_params.p_deviant = 0.3
    stim_params.amplitude = 1.6e-9
    stim_params.duration = 50.0e-3
    stim_params.deviant_idx = 2
    stim_params.standard_idx = 1
    stim_params.p_deviant = 0.15
    stim_params.start_offset = 2.5
    stim_params.isi = 300e-3

    @time stim_schedule = SNN.Params.generate_schedule(stim_params, tspan)

    # @time i_neurons = [SNN.Neuron.make_neuron(params, SNN.Neuron.Soma, tspan, Symbol("i_neuron_$(i)"), schedule) for i in 1:i_neurons_n]
    # @time e_neurons = [SNN.Neuron.make_neuron(params, SNN.Neuron.Soma, tspan, Symbol("e_neuron_$(i)"), schedule) for i in 1:e_neurons_n]

    # ee_rule = SNN.Neuron.ConnectionRule(SNN.Neuron.excitator, SNN.Neuron.excitator, SNN.Neuron.AMPA(), 0.05)
    # ei_rule = SNN.Neuron.ConnectionRule(SNN.Neuron.excitator, SNN.Neuron.inhibitor, SNN.Neuron.AMPA(), 0.05)
    # ie_rule = SNN.Neuron.ConnectionRule(SNN.Neuron.inhibitor, SNN.Neuron.excitator, SNN.Neuron.GABAa(), 0.05)
    # ii_rule = SNN.Neuron.ConnectionRule(SNN.Neuron.inhibitor, SNN.Neuron.inhibitor, SNN.Neuron.AMPA(), 0.05)

    con_mapping_nested = [
        (SNN.Params.@connect_neurons [1, 2] SNN.Neuron.AMPA() 3),
    ]
    con_mapping = reduce(vcat, con_mapping_nested)

    pre_neurons = [row[1] for row in con_mapping]
    post_neurons = [row[2] for row in con_mapping]
    e_neurons_n = size(unique([pre_neurons; post_neurons]), 1)

    # solver = ISSEM(theta=1 / 2)
    solver = EulerHeun()
    tols = (5e-2, 5e-2)

    (sol, simplified_model, prob, results, neurons) = SNN.Pipeline.run_exp(
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
    )
    @show results

    # test gpu

    (sol, simplified_model, prob, results, neurons) = SNN.Pipeline.run_exp(
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
        # to_device=x -> ROCArray(x),
        remake_prob=prob,
        model=simplified_model,
        neurons=neurons
    )
    @show results

end

ti = time() - ti
println("\nTest took total time of:")
println(round(ti / 60, digits=3), " minutes")
