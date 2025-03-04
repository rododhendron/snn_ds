module Pipeline

using SciMLBase: AbstractODEProblem
using Symbolics, ModelingToolkit, DifferentialEquations, ComponentArrays, LinearAlgebra, LinearSolve, SciMLBase, DataInterpolations
using Random
using DataInterpolations

using ..Params
using ..Neuron
using ..Utils
using ..Plots

ModelingToolkit.get_continuous_events(sys::SDESystem) = [sys.continuous_events...]

function run_exp(path_prefix::String, name::String;
    tols::Tuple{Float64,Float64}=(1e-5, 1e-5),
    e_neurons_n::Int=0,
    i_neurons_n::Int=0,
    solver,
    params::ComponentVector,
    stim_params::ComponentVector,
    stim_schedule::Array{Float64,2},
    tspan::Tuple{Int,Int},
    con_mapping::Union{Nothing,Vector{Any}}=nothing,
    prob_con::Tuple{Float64,Float64,Float64,Float64}=(0.05, 0.05, 0.05, 0.05),
    remake_prob::Any=nothing,
    model::Any=nothing,
    save_plots::Bool=false,
    fetch_csi::Bool=false,
    to_device::Function=x -> x,
    l::Union{Nothing,Base.AbstractLock}=nothing,
    neurons::Tuple{Union{Nothing,Vector{ODESystem}},Union{Nothing,Vector{ODESystem}}}=(nothing, nothing),
    nout::Bool=false
)# where {T<:Neuron.SynapseType}
    path = path_prefix * name * "/"
    mkpath(path)
    exp_name = path * name

    uparams = Neuron.get_adex_neuron_uparams_skeleton(Float64)
    if isnothing(remake_prob)
        e_neurons = [Neuron.make_neuron(params, Neuron.Soma, tspan, Symbol("e_neuron_$(i)"), stim_schedule) for i in 1:e_neurons_n]
        i_neurons = [Neuron.make_neuron(params, Neuron.Soma, tspan, Symbol("i_neuron_$(i)"), stim_schedule) for i in 1:i_neurons_n]
    else
        (e_neurons, i_neurons) = (neurons[1], neurons[2])
    end

    (input_neurons, rules) = Params.update_neurons_rules_from_sequence(e_neurons, stim_params, params)
    (input_grp, input_neuron_vec) = zip(input_neurons...)
    input_neurons_name = [neuron.name for neuron in [input_neuron_vec...;]]
    overriden_params = Params.override_params(params, rules)
    if isnothing(remake_prob)
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

        # [println(e) for e in equations(noise_network)]
        # [println(e) for e in parameters(noise_network)]
        # [println(e) for e in unknowns(noise_network)]

        network = Neuron.make_network(e_neurons, connections)

        # add noise
        noise_eqs = Neuron.instantiate_noise(network, e_neurons, params.sigma)

        @named sde_network = SDESystem(
            equations(network),
            collect(noise_eqs),
            ModelingToolkit.get_iv(network),
            unknowns(network),
            parameters(network),
            continuous_events=continuous_events(network),
            observed=observed(network);
        )
        # @named noise_network = System([equations(network)...; noise_eqs...], ModelingToolkit.get_iv(network))
        noise_network = ODESystem(sde_network)
        simplified_model = structural_simplify(noise_network; split=true, jac=true)
        tree::Utils.ParamTree = Utils.make_param_tree(simplified_model)
        res = Utils.fetch_tree(["e_neuron", "R"], tree)
        iparams, iuparams = Neuron.map_params(simplified_model, overriden_params, uparams; match_nums=true)
        contin_cb = ModelingToolkit.generate_rootfinding_callback([simplified_model.continuous_events...], simplified_model, unknowns(simplified_model), parameters(simplified_model))
        cb = ModelingToolkit.merge_cb(contin_cb, nothing) # 2nd arg is placeholder for discrete callback

        prob = SDEProblem{true,SciMLBase.FullSpecialize}(simplified_model, iuparams, tspan, iparams, cb=cb, maxiters=1e7)#, sparse=true)
    # prob = SymbolicsAMDGPU.SDEProblem(simplified_model, iuparams, tspan, iparams, cb=cb, maxiters=1e7; use_gpu=true)#, sparse=true)
    else
        simplified_model = model
        tree = Utils.make_param_tree(model)
        iparams, iuparams = Neuron.map_params(model, overriden_params, uparams; match_nums=true)
        prob = remake(remake_prob; u0=iuparams, p=iparams)
    end

    # @time sol = solve(prob, KenCarp47(linsolve=KrylovJL_GMRES()); abstol=1e-4, reltol=1e-4, dtmax=1e-3)
    # @time sol = solve(prob, SOSRA(); abstol=1e-3, reltol=1e-3, dtmax=1e-3)
    # @time sol = solve(prob, SKenCarp(); abstol=1e-3, reltol=1e-3, dtmax=1e-3)
    # @time sol = solve(prob, ImplicitRKMil(), abstol=1e-2, reltol=-1e-2, dtmax=1e-3)
    # AMDGPU.allowscalar(true)
    sol = solve(prob, solver, abstol=tols[1], reltol=tols[2], dt=1e-4)

    (start, stop) = tspan

    name_prefix = exp_name * ""
    name_interpol(name) = name_prefix * "_" * name

    results = Dict()
    try
        Utils.write_params(params; name=name_interpol("params.yaml"))
        Utils.write_params(stim_params; name=name_interpol("stim_params.yaml"))
        Utils.write_params(iparams; name=name_interpol("iparams.yaml"))
        Utils.write_params(iuparams; name=name_interpol("iuparams.yaml"))
        for i in 1:e_neurons_n
            @time Plots.plot_excitator_value(i, sol, start, stop, name_interpol, tree, stim_params.start_offset, stim_schedule, true)
            @time Plots.plot_adaptation_value(i, sol, start, stop, name_interpol, tree, stim_params.start_offset, stim_schedule, true)
            @time Plots.plot_spike_rate(i, sol, start, stop, name_interpol, tree, stim_params.start_offset, stim_schedule, true)
            @time Plots.plot_spike_rate(i, sol, start, stop, name_interpol, tree, stim_params.start_offset, stim_schedule, true; time_window=0.1)
            @time Plots.plot_spike_rate(i, sol, start, stop, name_interpol, tree, stim_params.start_offset, stim_schedule, true; time_window=0.05)
            @time Plots.plot_spike_rate(i, sol, start, stop, name_interpol, tree, stim_params.start_offset, stim_schedule, true; time_window=0.01)
            (grouped_trials, offsetted_times) = Plots.compute_grand_average(sol, Plots.fetch_tree_neuron_value("e_neuron", i, "v", tree) |> first, stim_schedule, :value; interpol_fn=LinearInterpolation, sampling_rate=2000)
            Plots.plot_neuron_value(offsetted_times, grouped_trials, nothing, nothing, [0.0, 0.05]; start=-0.1, stop=maximum(offsetted_times), title="gdavg voltage neuron $(i)", name=name_interpol("gdavg_e_3_voltage.png"), schedule=stim_schedule, tofile=true, ylabel="voltage (in V)", xlabel="Time (in s)", multi=true, plot_stims=false)
            (agg_rate, ot) = Plots.compute_grand_average(sol, Plots.fetch_tree_neuron_value("e_neuron", i, "R", tree) |> first, stim_schedule, :spikes; interpol_fn=LinearInterpolation, time_window=0.01, sampling_rate=2000)
            Plots.plot_neuron_value(ot, agg_rate, nothing, nothing, [0.0, 0.05]; start=-0.1, stop=maximum(offsetted_times), title="gdavg spike rate neuron $(i)", name=name_interpol("gdavg_e_3_rate.png"), schedule=stim_schedule, tofile=true, ylabel="spike rate (in Hz)", xlabel="Time (in s)", multi=true, plot_stims=false)
            # @time Plots.plot_aggregated_rate(i, sol, name_interpol, tree, stim_schedule, true)
        end

        res = Utils.fetch_tree(["e_neuron", "R"], tree)
        ma = Utils.hcat_sol_matrix(res, sol)
        spikes_times = Utils.get_spike_timings(ma, sol)
        # sol_stripped = SciMLBase.strip_solution(sol)
        # Utils.write_sol(sol; name=name_interpol("sol.jld2"))
        Plots.plot_spikes((spikes_times, []); start=start, stop=stop, color=(:red, :blue), height=400, title="Network activity", xlabel="time (in s)", ylabel="neuron index", name=name_interpol("rs.png"), schedule=stim_schedule)

        groups = Utils.fetch_tree(["e_neuron", "group"], tree)
        stim_groups = !isnothing(stim_params.deviant_idx) ? [stim_params.standard_idx; stim_params.deviant_idx] : stim_params.standard_idx

        csi_returned = nothing

        if !isnothing(stim_params.deviant_idx) && stim_params.deviant_idx > 0
            standards = [stim[1] for stim in eachcol(stim_schedule) if stim[3] in stim_params.standard_idx]
            deviants = [stim[1] for stim in eachcol(stim_schedule) if stim[3] in stim_params.deviant_idx]
            #readout |> count dev on standard |> ok

            readouts = [neuron for neuron in e_neurons if neuron.name ∉ input_neurons_name]
            # I have one for now so take first

            if !isempty(readouts)
                readout = first(readouts) |> x -> Utils.fetch_tree([String(x.name), "R"], tree)
                mr = Utils.hcat_sol_matrix(readout, sol)
                spikes_readout = Utils.get_spike_timings(mr, sol) |> first # take first as I have one readout
                trials = Plots.get_trials_from_schedule(stim_schedule)
                trials_response = [count(x -> trial_t[1] < x < trial_t[2], spikes_readout) for trial_t in trials]
                groups = unique(stim_schedule[3, :]) .|> Int
                # @show getindex.(trials, 1)
                filtered_stim_schedule = [sched for sched in eachcol(stim_schedule) if sched[1] in getindex.(trials, 1)]
                # @show filtered_stim_schedule
                groups_stim_idxs = [findall(row -> row == group, filtered_stim_schedule[3, :]) for group in groups]
                groups_spikes = [sum(trials_response[gsi]) / length(trials_response[gsi]) for gsi in groups_stim_idxs]
                # @show groups_spikes

                window = 70e-3 # time in ms
                dev_match = Utils.get_matching_timings(deviants, spikes_readout, window)
                standard_match = Utils.get_matching_timings(standards, spikes_readout, window)

                dev_count = count(i -> i > 0, dev_match)
                standard_count = count(i -> i > 0, standard_match)

                results["deviant_proportion"] = dev_count / size(deviants, 1)
                results["standard_proportion"] = standard_count / size(standards, 1)
                tpnp = dev_count + size(standards, 1) - standard_count
                accuracy = tpnp / size(filtered_stim_schedule, 2)
                f1_score = 2 * dev_count / (2 * dev_count + standard_count + size(deviants, 1) - dev_count)
                results["f1_score"] = f1_score
                results["accuracy"] = accuracy

                (agg_rate_1, ot) = Plots.compute_grand_average(sol, first(readout), stim_schedule, :spikes; interpol_fn=LinearInterpolation, time_window=0.01, sampling_rate=20000)
                (agg_rate_2, ot) = Plots.compute_grand_average(sol, first(readout), stim_schedule, :spikes; interpol_fn=LinearInterpolation, time_window=0.1, sampling_rate=20000)
                # results["csi_returned_max"] = Plots.csi(agg_rate, ot, 0.0, 0.3; is_adaptative=true)
                results["csi_returned_50"] = Plots.csi(agg_rate_1, ot, 0.0, 0.05)
                results["csi_returned_100"] = Plots.csi(agg_rate_1, ot, 0.0, 0.1)
                results["csi_returned_300"] = Plots.csi(agg_rate_1, ot, 0.0, 0.3)

                results["csi_returned_50_01"] = Plots.csi(agg_rate_2, ot, 0.0, 0.05)
                results["csi_returned_100_01"] = Plots.csi(agg_rate_2, ot, 0.0, 0.1)
                results["csi_returned_300_01"] = Plots.csi(agg_rate_2, ot, 0.0, 0.3)
                # @show results["csi_returned"]
            end
        end

        # @show results

        Utils.write_params(results; name=name_interpol("result_metrics.yaml"))
    catch e
        println("catched $e")
        results["csi_returned_50"] = NaN
        results["csi_returned_100"] = NaN
        results["csi_returned_300"] = NaN
        # results["csi_returned_max"] = nothing
        if isa(e, LoadError) ||
           isa(e, DataInterpolations.RightExtrapolationError) ||
           isa(e, ArgumentError)
            println("LoadError")
            return results
        else
            rethrow()
        end
    end

    if nout
        results
    else
        (sol, simplified_model, prob, results, (e_neurons, i_neurons))
    end
end


end
