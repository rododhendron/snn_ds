module Params
using Base.Cartesian
using Symbolics
using ComponentArrays
using Distributions
using Transducers

function get_model_params(Je=10, delta=2, vthr=-50, Cm=200, vrest=-65, TauW=500, w0=0, a=6)	# model params
	mparams = SLVector((
		Je=10,
		delta = 2,
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
        p_deviant=.0,
        n_trials=10,
        isi=350.0,
        duration=50.0,
        amplitude=5.0,
        onset_ramp=0.0,
        offset_ramp=0.0,
        select_size=2
    )
end

make_rule(prefix, range, suffix, value) = Symbol.(prefix .* "_" .* string.(range) .* "__" .* suffix) .=> value
function override_params(params, rules)
	params_dict = Dict()
	for rule in rules
		params_dict[rule[1]] = rule[2]
	end
	ComponentArray(params; params_dict...
	)
end

function generate_sequence(params::ComponentVector)
    # generate sequence detecting paradigm case of deviant or many standard
    if isnothing(deviant_idx)
        n_stim = len(params.standard_idx)
        target_prob = 1/n_standards
        prob_vector = fill(target_prop, n_stim)
    else
        n_stim = 2
        target_prob = p_deviant
        prob_vector = [target_prob, 1-target_prob]
    end

    stim_distribution = Multinomial(n_stim, prob_vector)
    stims = rand(stim_distribution, params.n_trials)

    if !isnothing(deviant_idx)
        scan_stims = stims[2:end] |> ScanEmit(
            (prev, x) -> (prev == x == deviant_idx ? 1 : 0, x),
            first(stims)
        ) |> collect |> out_vec -> vcat(0, out_vec)
        idx_to_replace = findall(scan_stims)
        stims[idx_to_replace] .= standard_idx
        return stims
    end
    return stims
end

function get_input_neuron_index(idx_loc::NTuple, neurons, select_size)
    # !!! do not specify idx_loc
    selections = [(i-select_size, i+select_size) for i in idx_loc]
    # adjust selection with dim limits
    dims = size(neurons)
    select_query = Vector{UnitRange{Int}}(undef, size(dims, 1))
    for (sel, dim) in zip(selections, dims)
        min_sel::Int = min(sel[1], 1)
        max_sel::Int = max(sel[2], dim)
        push!(select_query, min_sel:max_sel)
    end
    return neurons[CartesianIndices(select_query)]
end

function fetch_neuron_ids(neurons::AbstractArrays{T, N}) where {T, N}
    pattern = r"[ei]_neuron_(\d+)"
    capture_neuron_id(neuron)::Int = match(pattern, neuron.name) |> m -> parse(Int, m.captures[1])
    @show size(neurons)
    ids = Array{Int}(undef, size(neurons)[1:N-1]...)

    @nloops (N-1) i neurons begin
        obj = neurons[(@ntuple (N*1) i)...]
        ids[(@ntuple (N-1) i)...] = capture_neuron_id(obj)
    end
    ids
end

function make_input_rule_neurons(neuron_groups, input_value)
    rules_target = 1:size(neuron_groups, 1) |> Map(i -> make_rule("e_neuron", fetch_neuron_ids([neuron_groups[i]]), "soma__input_value", x)) |> collect
end

function update_neurons_rules_from_sequence(neurons, stims_params, network_params)
    # update iv value in neurons
    stims_loc = isnothing(stims_params.deviant_idx) ? stims_params.standard_idx : [stims_params.standard_idx, stims_params.deviant_idx]
    input_neurons_groups = [get_input_neuron_index((idx_loc,), neurons, stims_params.select_size) for idx_loc in stims_loc]

    rules = [make_input_rule_neurons(input_neurons_groups, stims_params.amplitude)]
end
end
