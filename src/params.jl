module Params
using Base.Cartesian
using Symbolics
using ComponentArrays
using Distributions
using Transducers
using ModelingToolkit

function get_model_params(Je=10, delta=2, vthr=-50, Cm=200, vrest=-65, TauW=500, w0=0, a=6)# model params
    mparams = SLVector((
        Je=10,
        delta=2,
        vthr=-50,
        Cm=200,
        vrest=-65,
        TauW=500,
        w0=0,
        a=6
    ))
    uparams = SLVector((
        v=mparams.vrest,
        w=mparams.w0
    ))
    (mparams, uparams)
end

function instantiate_params_as_symbols(params)
    # Create expression for @parameters macro
    expr = Expr(:macrocall, Symbol("@parameters"))
    push!(expr.args, nothing)  # Required for macro calls

    # Add each parameter with its default value
    for name in LabelledArrays.symbols(mparams)
        push!(expr.args, :($name))
    end

    # Remember to eval expr on main scope
    expr
end

function get_stim_params_skeleton()
    ComponentVector(
        standard_idx=1,
        deviant_idx=2,
        p_deviant=0.2,
        n_trials=10,
        isi=350.0,
        duration=50.0,
        amplitude=5.0,
        onset_ramp=0.0,
        offset_ramp=0.0,
        select_size=0,
        start_offset=200.0,
    )
end

make_rule(prefix, range::Union{Vector{Int},UnitRange}, suffix, value) = Symbol.(prefix .* "_" .* string.(range) .* "__" .* suffix) .=> value
function override_params(params, rules)
    params_dict = Dict()
    for rule in rules
        params_dict[rule[1]] = rule[2]
    end
    ComponentArray(params; params_dict...
    )
end

function generate_schedule(params::ComponentVector)::Array{Float64, 2}
    # generate sequence detecting paradigm case of deviant or many standard
    # output of shape : (t_start, onset_duration, group)
    if isnothing(params.deviant_idx)
        n_standard = len(params.standard_idx)
        target_prob = 1 / n_standard
        prob_vector = fill(target_prop, n_stim)
    else
        n_stim = 2
        target_prob = params.p_deviant
        prob_vector = [target_prob, 1 - target_prob]
    end

    stim_distribution = Categorical(prob_vector)
    stims = rand(stim_distribution, Int(params.n_trials))

    schedule = Array{Float64, 2}(undef, 3, size(stims, 1))

    for i in 1:size(stims, 1)
        @inbounds schedule[1, i] = (i-1)*(params.isi + params.duration) + params.start_offset
        @inbounds schedule[2, i] = params.duration
        @inbounds schedule[3, i] = stims[i]
    end

    @show stims
    @show schedule
    return schedule
end

function get_input_neuron_index(idx_loc::NTuple, neurons, select_size)
    # !!! do not specify idx_loc
    selections = [(i - select_size, i + select_size) for i in idx_loc]
    # adjust selection with dim limits
    dims = size(neurons)
    select_query::Vector{UnitRange} = []
    for (sel, dim) in zip(selections, dims)
        min_sel::Int = max(sel[1], 1)
        max_sel::Int = min(sel[2], dim)
        push!(select_query, min_sel:max_sel)
    end
    return neurons[CartesianIndices(tuple(select_query...),)]
end

capture_neuron_id(neuron::ModelingToolkit.ODESystem)::Int = match(r"[ei]_neuron_(\d+)", neuron.name |> String) |> m -> parse(Int, m.captures[1])
@generated function fetch_neuron_ids(neurons::AbstractArray{T,N})::Vector{Int} where {T,N}
    quote
        ids = Array{Int}(undef, size(neurons)[1:N]...)
        @nloops $N i neurons begin
            obj = @nref $N neurons i
            ids[(@ntuple $N i)...] = capture_neuron_id(obj)
        end
        ids
    end
end

function make_input_rule_neurons(neuron_groups, input_value)
    rules_target = 1:size(neuron_groups, 1) |> Map(i -> make_rule("e_neuron", fetch_neuron_ids(neuron_groups[i]), "soma__input_value", input_value)) |> collect
    rules_group = 1:size(neuron_groups, 1) |> Map(i -> make_rule("e_neuron", fetch_neuron_ids(neuron_groups[i]), "soma__group", i)) |> collect
    return [rules_target...; rules_group...]
end

function update_neurons_rules_from_sequence(neurons, stims_params, network_params)
    # update iv value in neurons
    stims_loc = isnothing(stims_params.deviant_idx) ? stims_params.standard_idx : [stims_params.standard_idx, stims_params.deviant_idx]
    input_neurons_groups = [get_input_neuron_index((idx_loc,), neurons, stims_params.select_size) for idx_loc in stims_loc]

    rules = make_input_rule_neurons(input_neurons_groups, stims_params.amplitude)
end
end
