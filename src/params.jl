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

function deviant_stim_rule(stims, i)
    return i > 1 && stims[i-1] == 2 && stims[i] == 2
end

function standard_stim_rule(stims)
    n_consecutive = 5
    standards_idxs = findall(stim -> stim == 1, stims)
    standards_idxs_consecutive_mask = standards_idxs |>
                                      Consecutive(2; step=1) |>
                                      Map(x -> x[2] - x[1]) |>
                                      Consecutive(n_consecutive; step=1) |>
                                      Map(stim_group -> all(stim -> stim == 1, stim_group)) |>
                                      collect
    return standards_idxs[1:length(standards_idxs)-n_consecutive][standards_idxs_consecutive_mask] .+ n_consecutive
end

function filter_stims(stims::Vector)
    new_stims = stims
    stims_dev_to_replace = findall(deviant_stim_rule.(Ref(stims), 1:length(stims))) |> x -> randsubseq(x, 0.5)
    new_stims[stims_dev_to_replace] .= 1
    # for N consecutive standards, change the last one as a deviant

    # for each standard with N consecutive following values, replace by deviant
    standards_idxs_consecutive_mask = standard_stim_rule(new_stims)
    new_stims[standards_idxs_consecutive_mask] .= 2
    new_stims[[1]] .= 1
    new_stims
end

function check_stims(stims)
    # return true if wrong schedule
    return !isempty(standard_stim_rule(stims)) || any(deviant_stim_rule.(Ref(stims), 1:length(stims)))
end

function generate_schedule(params::ComponentVector, tspan::Tuple{Int,Int}; is_pseudo_random::Bool=true)::Array{Float64,2}
    # Implementation for Int, Int tspan
    # Random.seed!(1234)
    # generate sequence detecting paradigm case of deviant or many standard
    # output of shape : (t_start, onset_duration, group)
    # Handle the case where tspan[2] <= params.start_offset to avoid negative n_trials
    duration = tspan[2] - params.start_offset
    if duration <= 0
        n_trials = 0
    else
        n_trials = max(1, div(duration, (params.isi + params.duration)))
    end
    if isnothing(params.deviant_idx)
        n_standard = len(params.standard_idx)
        target_prob = 1 / n_standard
        prob_vector = fill(target_prop, n_stim)
    else
        n_stim = 2
        target_prob = params.p_deviant
        prob_vector = [1 - target_prob, target_prob]
    end

    # Handle case where n_trials is 0 - create empty schedule
    if n_trials == 0
        return Array{Float64,2}(undef, 3, 0)
    end

    stim_distribution = Categorical(prob_vector)
    stims = rand(stim_distribution, Int(n_trials))

    if is_pseudo_random && !isnothing(params.deviant_idx)
        prev_stims = zeros(eltype(stims), size(stims))
        filtered_stims = stims
        while check_stims(filtered_stims)
            @show filtered_stims
            filtered_stims = filter_stims(filtered_stims)
            @show filtered_stims
        end
    else
        filtered_stims = stims
    end

    @show filtered_stims

    schedule = Array{Float64,2}(undef, 3, size(filtered_stims, 1))

    for i in 1:size(stims, 1)
        @inbounds schedule[1, i] = (i - 1) * (params.isi + params.duration) + params.start_offset
        @inbounds schedule[2, i] = params.duration
        @inbounds schedule[3, i] = filtered_stims[i]
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

# Add overload for Tuple{Int,Float64} to handle the type in tests
function generate_schedule(params::ComponentVector, tspan::Tuple{Int,Float64}; is_pseudo_random::Bool=true)::Array{Float64,2}
    # Convert the Float64 to Int for compatibility - use ceil to ensure we don't truncate time
    int_tspan = (tspan[1], Int(ceil(tspan[2])))
    return generate_schedule(params, int_tspan; is_pseudo_random=is_pseudo_random)
end

end
