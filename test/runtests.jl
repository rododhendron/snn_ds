using DrWatson, Test
@quickactivate "snn_ds"

# Here you include files using `srcdir`
# include(srcdir("file.jl"))
# include("../src/SNN.jl")
using SNN
using ModelingToolkit, LinearSolve, DifferentialEquations

# Run test suite
println("Starting tests")
ti = time()

@testset "SNN.jl" begin
    tspan = (0, 1)
    i_neurons_n = 2
    e_neurons_n = 2

    @time params = SNN.Neuron.get_adex_neuron_params_skeleton(Float64)

    @time stim_params = SNN.Params.get_stim_params_skeleton()
    @time schedule = SNN.Params.generate_schedule(stim_params)

    @time i_neurons = [SNN.Neuron.make_neuron(params, SNN.Neuron.Soma, tspan, Symbol("i_neuron_$(i)"), schedule) for i in 1:i_neurons_n]
    @time e_neurons = [SNN.Neuron.make_neuron(params, SNN.Neuron.Soma, tspan, Symbol("e_neuron_$(i)"), schedule) for i in 1:e_neurons_n]

    ee_rule = SNN.Neuron.ConnectionRule(SNN.Neuron.excitator, SNN.Neuron.excitator, SNN.Neuron.AMPA(), 0.05)
    ei_rule = SNN.Neuron.ConnectionRule(SNN.Neuron.excitator, SNN.Neuron.inhibitor, SNN.Neuron.AMPA(), 0.05)
    ie_rule = SNN.Neuron.ConnectionRule(SNN.Neuron.inhibitor, SNN.Neuron.excitator, SNN.Neuron.GABAa(), 0.05)
    ii_rule = SNN.Neuron.ConnectionRule(SNN.Neuron.inhibitor, SNN.Neuron.inhibitor, SNN.Neuron.AMPA(), 0.05)

    (id_map, map_connect) = SNN.Neuron.init_connection_map(e_neurons, i_neurons, vcat(ee_rule, ei_rule, ie_rule))
    @show id_map
    @show map_connect
    connections = SNN.Neuron.instantiate_connections(id_map, map_connect, vcat(e_neurons, i_neurons))
    @show connections

    @time network = SNN.Neuron.make_network(vcat(e_neurons, i_neurons), connections)

    @time simplified_model = SNN.structural_simplify(network; split=false)

    @testset "Params" begin

        rules = SNN.Params.update_neurons_rules_from_sequence(e_neurons, stim_params, params)

        uparams = SNN.Neuron.get_adex_neuron_uparams_skeleton(Float64)

        overriden_params = SNN.Params.override_params(params, rules)
        iparams, iuparams = SNN.Neuron.map_params(simplified_model, overriden_params, uparams; match_nums=false)
        prob = ODEProblem{true}(simplified_model, iuparams, tspan, iparams)
        sol = solve(prob, KenCarp47(linsolve=KrylovJL_GMRES()); abstol=1e-4, reltol=1e-4, dtmax=1e-3)
    end
end

ti = time() - ti
println("\nTest took total time of:")
println(round(ti / 60, digits=3), " minutes")
