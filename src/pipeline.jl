module Pipeline

using Symbolics, ModelingToolkit, DifferentialEquations, ComponentArrays, LinearAlgebra, LinearSolve, SciMLBase
using Random

using ..Params
using ..Neuron
using ..Utils
using ..Plots

ModelingToolkit.get_continuous_events(sys::SDESystem) = [sys.continuous_events...]

function run_exp(path_prefix, name; e_neurons_n=0, i_neurons_n=0, solver, params, stim_params, stim_schedule, tspan, con_mapping=nothing, prob_con=(0.05, 0.05, 0.05, 0.05), remake_prob=nothing)
    Random.seed!(1234)
    path = path_prefix * name * "/"
    mkpath(path)
    exp_name = path * name

    @time e_neurons = [Neuron.make_neuron(params, Neuron.Soma, tspan, Symbol("e_neuron_$(i)"), stim_schedule) for i in 1:e_neurons_n]
    @time i_neurons = [Neuron.make_neuron(params, Neuron.Soma, tspan, Symbol("i_neuron_$(i)"), stim_schedule) for i in 1:i_neurons_n]

    (input_neurons, rules) = Params.update_neurons_rules_from_sequence(e_neurons, stim_params, params)
    (input_grp, input_neuron_vec) = zip(input_neurons...)
    input_neurons_name = [neuron.name for neuron in [input_neuron_vec...;]]
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

    # add noise
    noise_eqs = Neuron.instantiate_noise(network, e_neurons, 0.001)

    @named noise_network = SDESystem(network, noise_eqs, continuous_events=continuous_events(network), observed=observed(network))

    @time simplified_model = structural_simplify(noise_network; split=false)
    @time tree::Utils.ParamTree = Utils.make_param_tree(simplified_model)
    @time res = Utils.fetch_tree(["e_neuron", "R"], tree)

    # infere params
    @time uparams = Neuron.get_adex_neuron_uparams_skeleton(Float64)

    @time iparams, iuparams = Neuron.map_params(simplified_model, overriden_params, uparams; match_nums=true)


    if isnothing(remake_prob)
        println("Make problem...")
        contin_cb = ModelingToolkit.generate_rootfinding_callback([simplified_model.continuous_events...], simplified_model, unknowns(simplified_model), parameters(simplified_model))
        cb = ModelingToolkit.merge_cb(contin_cb, nothing) # 2nd arg is placeholder for discrete callback

        @time prob = SDEProblem(simplified_model, iuparams, tspan, iparams, cb=cb)#, sparse=true)
        # @time prob = SDEProblem{true}(simplified_model, iuparams, tspan, iparams)#, sparse=true)
    else
        println("Remake problem...")
        @time prob = remake(remake_prob; u0=iuparams, p=iparams)
    end

    println("Solving...")
    # @time sol = solve(prob, KenCarp47(linsolve=KrylovJL_GMRES()); abstol=1e-4, reltol=1e-4, dtmax=1e-3)
    # @time sol = solve(prob, SOSRA(); abstol=1e-3, reltol=1e-3, dtmax=1e-3)
    # @time sol = solve(prob, SKenCarp(); abstol=1e-3, reltol=1e-3, dtmax=1e-3)
    # @time sol = solve(prob, ImplicitRKMil(), abstol=1e-2, reltol=-1e-2, dtmax=1e-3)
    @time sol = solve(prob, solver, abstol=1e-4, reltol=1e-4, dtmax=1e-3)

    println("tree fetch...")
    # @time res = Utils.fetch_tree(["e_neuron", "R"], tree)
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
        @time Plots.plot_excitator_value(i, sol, start, stop, name_interpol, tree, stim_params.start_offset, stim_schedule)
        @time Plots.plot_adaptation_value(i, sol, start, stop, name_interpol, tree, stim_params.start_offset, stim_schedule)
    end

    res = Utils.fetch_tree(["e_neuron", "R"], tree)
    ma = Utils.hcat_sol_matrix(res, sol)
    spikes_times = Utils.get_spike_timings(ma, sol)
    Utils.write_params(params; name=name_interpol("params.yaml"))
    Utils.write_params(stim_params; name=name_interpol("stim_params.yaml"))
    Utils.write_params(iparams; name=name_interpol("iparams.yaml"))
    Utils.write_params(iuparams; name=name_interpol("iuparams.yaml"))
    # sol_stripped = SciMLBase.strip_solution(sol)
    Utils.write_sol(sol; name=name_interpol("sol.jld2"))
    Plots.plot_spikes((spikes_times, []); start=start, stop=stop, color=(:red, :blue), height=400, title="Network activity", xlabel="time (in s)", ylabel="neuron index", name=name_interpol("rs.png"), schedule=stim_schedule)

    @time groups = Utils.fetch_tree(["e_neuron", "group"], tree)
    stim_groups = !isnothing(stim_params.deviant_idx) ? [stim_params.standard_idx; stim_params.deviant_idx] : stim_params.standard_idx

    results = Dict()

    if !isnothing(stim_params.deviant_idx) && stim_params.deviant_idx > 0
        standards = [stim[1] for stim in eachcol(stim_schedule) if stim[3] in stim_params.standard_idx]
        deviants = [stim[1] for stim in eachcol(stim_schedule) if stim[3] in stim_params.deviant_idx]
        #readout |> count dev on standard |> ok

        readouts = [neuron for neuron in e_neurons if neuron.name ∉ input_neurons_name]
        # I have one for now so take first

        readout = first(readouts) |> x -> Utils.fetch_tree([String(x.name), "R"], tree)
        mr = Utils.hcat_sol_matrix(readout, sol)
        spikes_readout = Utils.get_spike_timings(mr, sol) |> first # take first as I have one readout

        window = 70e-3 # time in ms
        dev_match = Utils.get_matching_timings(deviants, spikes_readout, window)
        standard_match = Utils.get_matching_timings(standards, spikes_readout, window)

        dev_count = count(i -> i > 0, dev_match)
        standard_count = count(i -> i > 0, standard_match)

        results["deviant_proportion"] = dev_count / size(deviants, 1)
        results["standard_proportion"] = standard_count / size(standards, 1)
        tpnp = dev_count + size(standards, 1) - standard_count
        accuracy = tpnp / size(stim_schedule, 2)
        f1_score = 2 * dev_count / (2 * dev_count + standard_count + size(deviants, 1) - dev_count)
        results["f1_score"] = f1_score
        results["accuracy"] = accuracy
    end

    @show results
    Utils.write_params(results; name=name_interpol("result_metrics.yaml"))

    (sol, simplified_model, prob)
end


end
