module Neuron
using Random
using ModelingToolkit, Parameters, StringDistances, Transducers, BangBang, StaticArrays, Symbolics, DifferentialEquations, ComponentArrays
using ModelingToolkit: t_nounits as t, D_nounits as D, Model, AbstractODESystem
using PrecompileTools: @setup_workload, @compile_workload

const start_input::Float64 = 200e-3
# const input_duration::Float64 = 200e-3
const dt_clamp = 5 # ms

function step_fn(t, iv, sch_onset, sch_group, sch_t, neuron_group)
    # Vector of (t_start, duration, target)
    t_range = searchsorted(sch_t[:], t)
    t_min = min(t_range.start, t_range.stop) |> x -> x == 0 ? 1 : x
    stim = (sch_t[t_min], sch_onset[t_min], sch_group[t_min])
    return ifelse(
        (neuron_group == stim[3]) & (stim[1] <= t <= stim[1] + stim[2]),
        iv,
        0
    )
end
@register_symbolic step_fn(t, iv, sch_onset::Vector{Float64}, sch_group::Vector{Float64}, sch_t::Vector{Float64}, neuron_group)

function step_input(t, iv)
    return ifelse(
        t <= start_input,
        iv,
        1
    )
end

@mtkmodel Soma begin # AdEx neuron from https://journals.physiology.org/doi/pdf/10.1152/jn.00686.2005
    @variables begin
        v(t) = -65e-3
        w(t) = 0
        Ie(t), [input = true]
        Ii(t), [input = true]
        Ib(t), [input = true]
        R(t), [output = true]
    end
    @structural_parameters begin
        n_stims
        # step
        sch_onset
        sch_group
        sch_t
    end
    @parameters begin
        Je
        vrest
        delta
        vthr
        vspike
        Cm
        a
        b
        TauW
        input_value
        duration
        group
        Ibase
        # noise
        # Ie
        # Ii
    end
    @equations begin
        D(v) ~ (-Je * (v - vrest) + Je * delta * exp((v - vthr) / delta) - w + Ii + Ie + Ib + Ibase) / Cm # voltage dynamic
        D(w) ~ (-w + a * (v - vrest)) / TauW # adaptation dynamic
        Ib ~ step_fn(t, input_value, sch_onset, sch_group, sch_t, group)
        # Ib ~ step_input(t, input_value)
        D(R) ~ 0
    end
    # @continuous_events begin
    # [Rp(k) != Rp(k-1)] => [v ~ vrest]
    # end
end

@mtkmodel SynapseAMPA begin # AMPA synapse https://neuronaldynamics.epfl.ch/online/Ch3.S1.html#Ch3.E2
    @parameters begin
        tau_ampa
        vtarget_exc
        inc_gsyn
    end
    @variables begin
        g_syn(t) = 0, [input = true]
    end
    # have to bind I on connect
    @equations begin
        D(g_syn) ~ -g_syn / tau_ampa
    end
end

@mtkmodel SynapseGABAa begin # GABAa synapse https://neuronaldynamics.epfl.ch/online/Ch3.S1.html#Ch3.E2
    @parameters begin
        tau_GABAa_fast
        vtarget_inh
        inc_gsyn
    end
    @variables begin
        g_syn(t) = 0, [input = true]
    end
    # have to bind I on connect
    @equations begin
        D(g_syn) ~ -g_syn / tau_GABAa_fast
    end
end

function get_adex_neuron_params_skeleton(type::DataType)#, sch_t, sch_group, sch_onset)
    ComponentArray{type}(
        vrest=-65.0e-3,     # Resting membrane potential (V)
        vthr=-50.0e-3,      # Spike threshold (V)
        Je=30.0e-9,         # Membrane time constant (Siemens S)
        delta=2.0e-3,       # Spike slope factor (V)
        Cm=281e-12,         # Membrane capacitance (Farad F)
        TauW=144.0e-3,      # Adaptation time constant (s)
        a=40.0e-9,          # Subthreshold adaptation (A)
        gl=20e-9,           # leak conductance
        b=0.16e-9,          # Spiking adaptation (A)
        input_value=0,      # Ib = Basal input current (A)
        tau_ampa=10.0e-3,   # AMPA decrease time constant (s)
        tau_GABAa_rise=1e-3,# GABAa increase time constant (s)
        tau_GABAa_fast=6e-3,# GABAa decrease time constant (s)
        vtarget_exc=0,      # Voltage value that drive excitation (V)
        vtarget_inh=-75e-3, # Voltage value that drive inhibition (V)
        vspike=0,           # Voltage value defining resetting voltage at resting value (V)
        inc_gsyn=0.3e-9,    # Conductance value by which to increment when a synapse receive a spike (S)
        # e_neuron_1__soma__input_value=0.80e-9, # input value specific to excitatory neuron 1
        duration=400e-3,    # Duration of the stimulus onset (s)
        group=0,
        sigma=0,
        Ibase=0,
        # sch_t=sch_t,
        # sch_group=sch_group,
        # sch_onset=sch_onset,
    )
end

function get_adex_neuron_uparams_skeleton(type::DataType)
    ComponentVector{type}(
        v=-65.0e-3,
        w=0.0e-9,
        Ie=0,
        Ii=0,
        Ib=0e-9,
        R=0,
        g_syn=0
    )
end

abstract type SynapseType end
struct SIMPLE_E <: SynapseType end
struct AMPA <: SynapseType end
struct GABAa <: SynapseType end

function get_synapse_eq(_synapse_type::AMPA, post_neuron::AbstractODESystem)::Equation # _ before variable is a convention for a unused variable in function body ; just used for type dispatch
    Equation(post_neuron.ampa_syn.g_syn, post_neuron.ampa_syn.g_syn + post_neuron.ampa_syn.inc_gsyn)
end

function get_synapse_eq(_synapse_type::GABAa, post_neuron::AbstractODESystem)::Equation # _ before variable is a convention for a unused variable in function body ; just used for type dispatch
    Equation(post_neuron.gabaa_syn.g_syn, post_neuron.gabaa_syn.g_syn + post_neuron.gabaa_syn.inc_gsyn)
end

function get_synapse_eq(_synapse_type::Nothing, post_neuron::AbstractODESystem)::Nothing # _ before variable is a convention for a unused variable in function body ; just used for type dispatch
    nothing
end

function get_noise_eq(neuron, sigma::Float64)
    # ModelingToolkit.tobrownian(neuron.soma.noise)
    # neuron.soma.noise ~ sigma * b
    neuron.soma.v * sigma
    # sigma
end

function make_neuron(params::ComponentArray, soma_model::Model, tspan::Tuple{Int,Int}, name::Symbol, schedule_p)::ODESystem
    @named soma = soma_model(; name=Symbol("soma"), n_stims=size(schedule_p, 2), sch_t=deepcopy(schedule_p[1, :]), sch_group=deepcopy(schedule_p[3, :]), sch_onset=deepcopy(schedule_p[2, :]))
    @named ampa_syn = SynapseAMPA(; name=Symbol("ampa_syn"))
    @named gabaa_syn = SynapseGABAa(; name=Symbol("gabaa_syn"))

    neuron = ModelingToolkit.compose(
        ODESystem(
            [
                soma.Ie ~ ampa_syn.g_syn * -(soma.v - ampa_syn.vtarget_exc)
                soma.Ii ~ gabaa_syn.g_syn * -(soma.v - gabaa_syn.vtarget_inh)
            ], t; name=name
        ), soma, ampa_syn, gabaa_syn
    )
    neuron
end
make_events(premises::Vector{Equation}, pre::AbstractODESystem)::Pair{Vector{Equation},Vector{Equation}} = [pre.soma.v ~ pre.soma.vspike] => vcat(
    [
        pre.soma.v ~ pre.soma.vrest,
        pre.soma.R ~ pre.soma.R + 1,
        pre.soma.w ~ pre.soma.w + pre.soma.b
    ], premises
)

@enum NeuronType begin
    excitator
    inhibitor
end

struct ConnectionRule
    pre_neurons_type::NeuronType
    post_neurons_type::NeuronType
    synapse_type::SynapseType
    prob::Float64
end

function instantiate_connections(id_map, map_connect, post_neurons)::Vector{Pair{Vector{Equation},Vector{Equation}}}
    all_callbacks::Vector{Pair{Vector{Equation},Vector{Equation}}} = []
    for (i, neuron) in id_map
        post_neurons_syn = map_connect[i, :]
        post_neurons_callbacks::Vector{Equation} = get_synapse_eq.(post_neurons_syn, post_neurons) |> x -> filter(!isnothing, x)

        current_neuron_callbacks = make_events(post_neurons_callbacks, neuron)
        push!(all_callbacks, current_neuron_callbacks)
    end
    all_callbacks
end

function instantiate_noise(network, neurons, sigma)
    noise_eqs = get_noise_eq.(neurons, Ref(sigma)) |> Iterators.Stateful
    eqs = equations(network)
    # eqs_placeholder = Vector{Any}(undef, size(equations(network), 1), size(equations(network), 1))
    eqs_placeholder::Matrix{Any} = zeros(Float64, size(equations(network), 1), size(equations(network), 1))
    re_differential = r"Differential\(t\)\(((?:e_|i_))neuron_(\d+)₊soma₊v\(t\)\)"
    for i in 1:size(eqs_placeholder, 1)
        eq = eqs[i]
        match_eq = match(re_differential, string(eq.lhs))
        if !isnothing(match_eq)
            (neuron_type, neuron_id) = match_eq.captures
            # noise_eq_idx = findfirst(x -> occursin("$(neuron_type)neuron_$(neuron_id)", string(x)), noise_eqs)
            # eqs_placeholder[i, i] = noise_eqs[noise_eq_idx]
            eqs_placeholder[i, i] = popfirst!(noise_eqs)
        else
            eqs_placeholder[i, i] = 0.0
        end
    end
    eqs_placeholder
end

function infer_connection_from_map(neurons::Vector{T}, mapping) where {T<:AbstractODESystem}
    # take neurons, assign ids, map connections from dict map
    n_neurons = length(neurons)
    id_map = [(i, neurons[i]) for i in 1:n_neurons]
    map_connect = Array{Union{Nothing,SynapseType}}(nothing, n_neurons, n_neurons)
    for (pre_neuron_id, post_neuron_id, synapse) in mapping
        map_connect[pre_neuron_id, post_neuron_id] = synapse
    end
    (id_map, map_connect)
end
function init_connection_map(e_neurons::Vector{T}, i_neurons::Vector{T}, connection_rules::Vector{ConnectionRule}) where {T<:AbstractODESystem}
    neurons = vcat(e_neurons, i_neurons)
    n_neurons = length(e_neurons) + length(i_neurons)
    id_map = [(i, neurons[i]) for i in 1:n_neurons]
    map_connect = Array{Union{Nothing,SynapseType}}(nothing, n_neurons, n_neurons)
    for rule in connection_rules
        pre_neurons_ids = rule.pre_neurons_type == excitator ? range(1, length(e_neurons)) : range(length(e_neurons) + 1, n_neurons)
        for i in pre_neurons_ids
            post_neurons_ids = rule.post_neurons_type == excitator ? range(1, length(e_neurons)) : range(length(e_neurons) + 1, n_neurons)
            post_prob_neurons_ids = post_neurons_ids |> x -> randsubseq(x, rule.prob)
            # disallow self connection else bursting bug
            post_neurons = [n for n in post_prob_neurons_ids if n != i]
            map_connect[i, post_neurons] .= Ref(rule.synapse_type)
        end
    end
    (id_map, map_connect)
end

function make_network(neurons::Vector{T}, connections::Vector{Pair{Vector{Equation},Vector{Equation}}})::T where {T<:AbstractODESystem}
    ModelingToolkit.compose(ODESystem([], t; continuous_events=connections, name=:connected_neurons), neurons)
end

get_varsym_from_syskey(param::SymbolicUtils.BasicSymbolic)::Symbol = split(param |> Symbol |> String, "₊") |> last |> Symbol
get_varsym_from_syskey(param::Symbol)::Symbol = split(param |> String, "₊") |> last |> Symbol
get_varsym_from_syskey(param::SubString)::Symbol = split(param, "₊") |> last |> Symbol
varsym_to_vec(param::Symbol)::Vector = split(param |> String, "₊")

function match_param(to_match, to_find, match_nums::Bool)::Bool
    if match_nums
        to_match == to_find
    else
        matched_param = match(r"\D*(\d+)", to_match)
        to_find_replaced = isnothing(matched_param) ? to_find : replace(to_find, r"(\d+)" => matched_param.captures[1])

        to_match == to_find_replaced
    end
end

function process_vecs(vec_to_find, vec_to_match, match_nums::Bool)::Bool
    if isempty(vec_to_find)
        return true
    end
    if length(vec_to_find) != length(vec_to_match) || !match_param(vec_to_find[1], vec_to_match[1], match_nums)
        return false
    end
    process_vecs(vec_to_find[2:end], vec_to_match[2:end], match_nums::Bool)
end

function find_param(param_to_find, params, match_nums::Bool)
    fallback = get_varsym_from_syskey(param_to_find)
    params_vecs = propertynames(params) .|> x -> split(x |> String, "__")
    param_to_find_vec = param_to_find |> varsym_to_vec
    match_mask = process_vecs.(Ref(param_to_find_vec), params_vecs, match_nums)
    match_idx = findall(match_mask)
    matched_properties = propertynames(params)[match_idx]
    length(matched_properties) == 1 ? matched_properties |> first : fallback
end

function instantiate_param(parameter::SymbolicUtils.BasicSymbolic, params, match_nums::Bool)::Union{Pair{SymbolicUtils.BasicSymbolic,Union{Float64,Int64}},Nothing}
    best_match(param) = find_param(param, params, match_nums)
    if hasproperty(params, get_varsym_from_syskey(parameter))
        parameter => getproperty(params, parameter |> Symbol |> best_match |> Symbol)
    end
end

function instantiate_uparam(parameter::SymbolicUtils.BasicSymbolic, uparams)::Union{Pair{SymbolicUtils.BasicSymbolic,Union{Float64,Int64}},Nothing}
    replaced_parameter = split(parameter |> Symbol |> String, "(")[1]
    if hasproperty(uparams, get_varsym_from_syskey(replaced_parameter))
        parameter => getproperty(uparams, get_varsym_from_syskey(replaced_parameter))
    end
end

function map_params(network::AbstractODESystem, params::ComponentArray, uparams::ComponentArray; match_nums::Bool=true)::Tuple{Vector{Pair{SymbolicUtils.BasicSymbolic,Union{Int64,Float64}}},Vector{Pair{SymbolicUtils.BasicSymbolic,Union{Int64,Float64}}}}
    parameters_to_replace = parameters(network)
    uparameters_to_replace = unknowns(network)

    map_params = instantiate_param.(parameters_to_replace, Ref(params), match_nums) |> x -> filter(!isnothing, x)
    map_uparams = instantiate_uparam.(uparameters_to_replace, Ref(uparams)) |> x -> filter(!isnothing, x)

    (map_params, map_uparams)
end

@setup_workload begin
    @compile_workload begin
        # tspan = (0, 1)
        # params = get_adex_neuron_params_skeleton(Float64)
        # uparams = get_adex_neuron_uparams_skeleton(Float64)
        # i_neurons_n = 2
        # e_neurons_n = 2
        # i_neurons = [Neuron.make_neuron(params, Neuron.Soma, tspan, Symbol("i_neuron_$(i)")) for i in 1:i_neurons_n]
        # e_neurons = [Neuron.make_neuron(params, Neuron.Soma, tspan, Symbol("e_neuron_$(i)")) for i in 1:e_neurons_n]

        # ee_rule = Neuron.ConnectionRule(Neuron.excitator, Neuron.excitator, Neuron.AMPA(), 1.0)
        # ei_rule = Neuron.ConnectionRule(Neuron.excitator, Neuron.inhibitor, Neuron.AMPA(), 1.0)
        # ie_rule = Neuron.ConnectionRule(Neuron.inhibitor, Neuron.excitator, Neuron.GABAa(), 1.0)

        # (id_map, map_connect) = Neuron.init_connection_map(e_neurons, i_neurons, vcat(ee_rule, ei_rule, ie_rule))
        # connections = Neuron.instantiate_connections(id_map, map_connect, vcat(e_neurons, i_neurons))

        # network = Neuron.make_network(vcat(e_neurons, i_neurons), connections)

        # simplified_model = structural_simplify(network, split=false)

        # iparams, iuparams = Neuron.map_params(simplified_model, params, uparams; match_nums=false)

        # prob = ODEProblem(simplified_model, iuparams, tspan, iparams)

        # solve(prob, Vern6(); abstol=1e-6, reltol=1e-6)
    end
end
end
