module Pipeline

using Symbolics, ModelingToolkit, DifferentialEquations, ComponentArrays, LinearAlgebra, LinearSolve

using ..Params
using ..Neuron
using ..Utils
using ..Plots

function run_exp(path_prefix, name; e_neurons_n=0, i_neurons_n=0, params, stim_params, tspan, con_mapping=nothing, prob_con=(0.05, 0.05, 0.05, 0.05), remake_prob=nothing)
    path = path_prefix * name * "/"
    mkpath(path)
    exp_name = path * name
    schedule = Params.generate_schedule(stim_params, tspan)
    @time e_neurons = [Neuron.make_neuron(params, Neuron.Soma, tspan, Symbol("e_neuron_$(i)"), schedule) for i in 1:e_neurons_n]
    @time i_neurons = [Neuron.make_neuron(params, Neuron.Soma, tspan, Symbol("i_neuron_$(i)"), schedule) for i in 1:i_neurons_n]

    rules = Params.update_neurons_rules_from_sequence(e_neurons, stim_params, params)
    overriden_params = Params.override_params(params, rules)
    if !isnothing(con_mapping)
        (id_map, map_connect) = Neuron.infer_connection_from_map(e_neurons, con_mapping)
    else
        (ee_prob, ei_prob, ie_prob, ii_prob) = prob_con
        ee_rule = Neuron.ConnectionRule(Neuron.excitator, Neuron.excitator, Neuron.AMPA(), ee_prob)
        ei_rule = Neuron.ConnectionRule(Neuron.excitator, Neuron.inhibitor, Neuron.AMPA(), ei_prob)
        ie_rule = Neuron.ConnectionRule(Neuron.inhibitor, Neuron.excitator, Neuron.GABAa(), ie_prob)
        ii_rule = Neuron.ConnectionRule(Neuron.inhibitor, Neuron.inhibitor, Neuron.AMPA(), ii_prob)
        (id_map, map_connect) = Neuron.init_connection_map(e_neurons, i_neurons, vcat(ee_rule, ei_rule, ie_rule, ii_rule))
    end
    connections = Neuron.instantiate_connections(id_map, map_connect, e_neurons)

    @time network = Neuron.make_network(e_neurons, connections)

    @time simplified_model = structural_simplify(network; split=false)

    # infere params
    @time uparams = Neuron.get_adex_neuron_uparams_skeleton(Float64)

    @time iparams, iuparams = Neuron.map_params(simplified_model, overriden_params, uparams; match_nums=true)

    if isnothing(remake_prob)
        @time prob = ODEProblem{true}(simplified_model, iuparams, tspan, iparams, sparse=true)
    else
        @time prob = remake(remake_prob; u0=iuparams, p=iparams)
    end

    println("Solving...")
    # @time sol = solve(prob, ImplicitDeuflhardExtrapolation(threading=true); abstol=1e-3, reltol=1e-3)
    @time sol = solve(prob, KenCarp47(linsolve=KrylovJL_GMRES()); abstol=1e-4, reltol=1e-4, dtmax=1e-3)
    # @time sol = solve(prob, Rodas5P(); abstol=1e-6, reltol=1e-6)

    @show sol.stats

    @time tree::Utils.ParamTree = Utils.make_param_tree(simplified_model)

    println("tree fetch...")
    @time res = Utils.fetch_tree(["e_neuron", "R"], tree)
    # @time ris = Utils.fetch_tree(["i_neuron", "R"], tree)

    println("hcat solutions")
    @time mes = reduce(hcat, sol[res])
    # @time mis = reduce(hcat, sol[ris])

    println("get spikes timings")
    @time spikes_e = Utils.get_spike_timings(mes, sol)
    # @time spikes_i = Utils.get_spike_timings(mis, sol)

    (start, stop) = tspan

    name_prefix = exp_name * ""
    name_interpol(name) = name_prefix * "_" * name

    println("plotting")


    for i in 1:e_neurons_n
        @time Plots.plot_excitator_value(i, sol, start, stop, name_interpol, tree, stim_params.start_offset, schedule)
        @time Plots.plot_adaptation_value(i, sol, start, stop, name_interpol, tree, stim_params.start_offset, schedule)
    end

    res = Utils.fetch_tree(["e_neuron", "R"], tree)
    ma = Utils.hcat_sol_matrix(res, sol)
    spikes_times = Utils.get_spike_timings(ma, sol)
    Utils.write_params(params; name=name_interpol("params.yaml"))
    Utils.write_params(stim_params; name=name_interpol("stim_params.yaml"))
    Utils.write_params(iparams; name=name_interpol("iparams.yaml"))
    Utils.write_params(iuparams; name=name_interpol("iuparams.yaml"))
    Utils.write_sol(sol; name=name_interpol("sol.jld2"))
    Plots.plot_spikes((spikes_times, []); start=start, stop=stop, color=(:red, :blue), height=400, title="Network activity", xlabel="time (in s)", ylabel="neuron index", name=name_interpol("rs.png"), schedule=schedule)
    (sol, simplified_model, prob)
end



end
