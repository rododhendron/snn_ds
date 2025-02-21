module Params
using Base.Cartesian
using Symbolics
using ComponentArrays
using Distributions
using Transducers
using ModelingToolkit
using Random

export connect_neurons

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
        deviant_idx=5,
        p_deviant=0.2,
        n_trials=10,
        isi=350.0e-3,
        duration=50.0e-3,
        amplitude=5.0,
        onset_ramp=0.0,
        offset_ramp=0.0,
        select_size=0,
        start_offset=200.0e-3,
    )
end

make_rule(prefix, range::Union{Vector{Int},UnitRange}, suffix, value) = Symbol.(prefix .* "_" .* string.(range) .* "__" .* suffix) .=> value
make_rule(prefix, id::Int, suffix, value) = Symbol(prefix * "_" * string(id) * "__" * suffix) .=> value
function override_params(params, rules)
    params_dict = Dict()
    for rule in rules
        params_dict[rule[1]] = rule[2]
    end
    ComponentArray(params; params_dict...
    )
end

function generate_schedule(params::ComponentVector, tspan::Tuple{Int,Int})::Array{Float64,2}
    # Random.seed!(1234)
    # generate sequence detecting paradigm case of deviant or many standard
    # output of shape : (t_start, onset_duration, group)
    n_trials = div((tspan[2] - params.start_offset), (params.isi + params.duration))
    if isnothing(params.deviant_idx)
        n_standard = len(params.standard_idx)
        target_prob = 1 / n_standard
        prob_vector = fill(target_prop, n_stim)
    else
        n_stim = 2
        target_prob = params.p_deviant
        prob_vector = [1 - target_prob, target_prob]
    end

    stim_distribution = Categorical(prob_vector)
    stims = rand(stim_distribution, Int(n_trials))

    schedule = Array{Float64,2}(undef, 3, size(stims, 1))

    for i in 1:size(stims, 1)
        @inbounds schedule[1, i] = (i - 1) * (params.isi + params.duration) + params.start_offset
        @inbounds schedule[2, i] = params.duration
        @inbounds schedule[3, i] = stims[i]
    end

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
    rules = []
    for (group, (idx_loc, neuron_group)) in enumerate(neuron_groups)
        append!(rules, make_rule("e_neuron", fetch_neuron_ids(neuron_group), "soma__input_value", input_value))
        append!(rules, make_rule("e_neuron", fetch_neuron_ids(neuron_group), "soma__group", group))
    end
    # @show rules
    return rules
end

function update_neurons_rules_from_sequence(neurons, stims_params, network_params)
    # update iv value in neurons
    stims_loc = isnothing(stims_params.deviant_idx) ? stims_params.standard_idx : [stims_params.standard_idx, stims_params.deviant_idx]
    input_neurons_groups = [(idx_loc, get_input_neuron_index((idx_loc,), neurons, stims_params.select_size)) for idx_loc in stims_loc]

    rules = make_input_rule_neurons(input_neurons_groups, stims_params.amplitude)
    (input_neurons_groups, rules)
end

macro connect_neurons(pre_neurons, prob, synapse, post_neurons)
    # Generate the code
    quote
        local connections_set = []
        for pre_neuron in $pre_neurons
            for post_neuron in $post_neurons
                if rand() < $prob
                    push!(connections_set, (pre_neuron, post_neuron, $synapse))
                end
            end
        end
        connections_set
    end |> esc
end
macro connect_neurons(pre_neurons, synapse, post_neurons)
    # Generate the code
    quote
        local connections_set = []
        for pre_neuron in $pre_neurons
            for post_neuron in $post_neurons
                push!(connections_set, (pre_neuron, post_neuron, $synapse))
            end
        end
        connections_set
    end |> esc
end
end
