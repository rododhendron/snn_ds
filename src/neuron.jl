module Neuron
using ModelingToolkit, Parameters, StringDistances, Transducers, BangBang, StaticArrays
using ModelingToolkit: t_nounits as t, D_nounits as D

const start_input::Float64 = 200e-3
const input_duration::Float64 = 200e-3
step_fn(t, iv) = start_input < t < start_input + input_duration ? iv : 0
@register_symbolic step_fn(t, iv)

@mtkmodel Soma begin # AdEx neuron from https://journals.physiology.org/doi/pdf/10.1152/jn.00686.2005
    @variables begin
        v(t) = -65e-3
        w(t) = 0
        Ie(t), [input = true]
        Ii(t), [input = true]
        Ib(t), [input = true]
        R(t), [output = true]
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
    end
    @equations begin
        D(v) ~ (-Je * (v - vrest) + Je * delta * exp((v - vthr) / delta) - w + Ii + Ie + Ib) / Cm # voltage dynamic
        D(w) ~ (-w + a * (v - vrest)) / TauW # adaptation dynamic
        Ib ~ step_fn(t, input_value)
        D(R) ~ 0
    end
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
        tau_GABAa_rise
        tau_GABAa_fast
        vtarget_inh
        inc_gsyn
    end
    @variables begin
        g_syn(t) = 0, [input = true]
    end
    # have to bind I on connect
    @equations begin
        D(g_syn) ~ -g_syn / tau_GABAa_rise + g_syn / tau_GABAa_fast
    end
end


# Base neuron parameters
Base.@kwdef struct AdExNeuronParams
    vrest::Float64 = -65.0e-3  # Resting membrane potential (V)
    vthr::Float64 = -50.0e-3   # Spike threshold (V)
    Je::Float64 = 30.0e-9       # Membrane time constant (Siemens S)
    delta::Float64 = 2.0e-3    # Spike slope factor (V)
    Cm::Float64 = 281e-12       # Membrane capacitance (Farad F)
    TauW::Float64 = 144.0e-3    # Adaptation time constant (s)
    a::Float64 = 40.0e-9        # Subthreshold adaptation (A)
    gl::Float64 = 20e-9     # leak conductance
    b::Float64 = 0.08e-9
    input_value::Float64 = 0
    gmax::Float64 = 6.0e-9
    tau_ampa::Float64 = 5.0e-3
    tau_GABAa_rise::Float64 = 1e-3
    tau_GABAa_fast::Float64 = 6e-3
    vtarget_exc::Float64 = 0
    vtarget_inh::Float64 = -75e-3
    vspike::Float64 = 0
    inc_gsyn::Float64 = 1e-9
    e_neuron_1__soma_a__input_value::Float64 = 2e-9
end
Base.@kwdef struct AdExNeuronUParams
    v::Float64 = -65.0e-3
    w::Float64 = 0.0e-9
    Ie::Float64 = 0
    Ii::Float64 = 0
    Ib::Float64 = 1e-9
    R::Int64 = 0
    g_syn::Float64 = 0
end

struct Params{S}
    gl::Float64
    a::Float64
    vrest::Float64
    delta::Float64
    vthr::Float64
    TauW::Float64
end

abstract type SynapseType end
struct SIMPLE_E <: SynapseType end
struct AMPA <: SynapseType end
struct GABAa <: SynapseType end

function get_synapse_eq(_synapse_type::AMPA, post_neuron::ODESystem)::Equation # _ before variable is a convention for a unused variable in function body ; just used for type dispatch
    Equation(post_neuron.ampa_syn.g_syn, post_neuron.ampa_syn.g_syn + post_neuron.ampa_syn.inc_gsyn)
end

function get_synapse_eq(_synapse_type::GABAa, post_neuron::ODESystem)::Equation # _ before variable is a convention for a unused variable in function body ; just used for type dispatch
    Equation(post_neuron.gabaa_syn.g_syn, post_neuron.gabaa_syn.g_syn + post_neuron.gabaa_syn.inc_gsyn)
end

function get_synapse_eq(_synapse_type::Nothing, post_neuron::ODESystem)::Nothing # _ before variable is a convention for a unused variable in function body ; just used for type dispatch
    nothing
end

function make_neuron(params, soma_model, tspan, name)::ODESystem
    @named soma = soma_model(; name=Symbol("soma"))
    @named ampa_syn = SynapseAMPA(; name=Symbol("ampa_syn"))
    @named gabaa_syn = SynapseGABAa(; name=Symbol("gabaa_syn"))

    neuron = ModelingToolkit.compose(
        ODESystem(
            [
                soma.Ie ~ ampa_syn.g_syn * (soma.v - ampa_syn.vtarget_exc)
                soma.Ii ~ gabaa_syn.g_syn * (soma.v - gabaa_syn.vtarget_inh)
            ], t; name=name
        ), soma, ampa_syn, gabaa_syn
    )
    neuron
end
make_events(premises::Vector{Equation}, pre::ODESystem)::Pair{Vector{Equation},Vector{Equation}} = [pre.soma.v ~ pre.soma.vspike] => vcat(
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

function init_connection_map(e_neurons::Vector{ODESystem}, i_neurons::Vector{ODESystem}, connection_rules::Vector{ConnectionRule})
    prob_filter(prob) = Filter(_x -> rand() <= prob)
    neurons = vcat(e_neurons, i_neurons)
    n_neurons = length(e_neurons) + length(i_neurons)
    id_map = [(i, neurons[i]) for i in 1:n_neurons]
    map_connect = Array{Union{Nothing,SynapseType}}(nothing, n_neurons, n_neurons)
    for rule in connection_rules
        pre_neurons_ids = rule.pre_neurons_type == excitator ? range(1, length(e_neurons)) : range(length(e_neurons) + 1, n_neurons)
        for i in pre_neurons_ids
            post_neurons_ids = rule.post_neurons_type == excitator ? range(1, length(e_neurons)) : range(length(e_neurons) + 1, n_neurons)
            post_prob_neurons_ids = post_neurons_ids |> prob_filter(rule.prob) |> collect

            map_connect[pre_neurons_ids, post_prob_neurons_ids] .= Ref(rule.synapse_type)
        end
    end
    (id_map, map_connect)
end

function make_network(neurons::Vector{ODESystem}, connections::Vector{Pair{Vector{Equation},Vector{Equation}}})::ODESystem
    @mtkbuild network = ModelingToolkit.compose(ODESystem([], t; continuous_events=connections, name=:connected_neurons), neurons)
end

get_varsym_from_syskey(param::SymbolicUtils.BasicSymbolic)::Symbol = Symbol(split(param |> Symbol |> String, "₊"))
get_varsym_from_syskey(param::SubString)::Symbol = Symbol(split(param, "₊")[end])

function best_match_varsym(param::Vector{Symbol}, params)
    last_sym_p = param[end]
    naive_match = [p for p in propertynames(params) if p == last_sym_p]
    if length(naive_match) > 1
        found_match = findnearest(param |> String, propertynames(params) .|> String, Levenshtein())
        getproperty(params, found_match)
    else
        length(naive_match) == 1 ? first(naive_match) : nothing
    end
end

function instantiate_param(parameter::SymbolicUtils.BasicSymbolic, params::AdExNeuronParams)::Union{Pair{SymbolicUtils.BasicSymbolic,Union{Float64,Int64}},Nothing}
    best_match(param) = best_match_varsym(param, params)
    if hasproperty(params, get_varsym_from_syskey(parameter))
        parameter => getproperty(params, get_varsym_from_syskey(parameter) |> best_match)
    end
end

function instantiate_uparam(parameter::SymbolicUtils.BasicSymbolic, uparams::AdExNeuronUParams)::Union{Pair{SymbolicUtils.BasicSymbolic,Union{Float64,Int64}},Nothing}
    best_match(param) = best_match_varsym(param, params)
    replaced_parameter = split(parameter |> Symbol |> String, "(")[1]
    if hasproperty(uparams, get_varsym_from_syskey(replaced_parameter))
        parameter => get_varsym_from_syskey(replaced_parameter) |> best_match
    end
end

function map_params(network::ODESystem, params::AdExNeuronParams, uparams::AdExNeuronUParams)::Tuple{Vector{Pair{SymbolicUtils.BasicSymbolic,Union{Int64,Float64}}},Vector{Pair{SymbolicUtils.BasicSymbolic,Union{Int64,Float64}}}}
    parameters_to_replace = parameters(network)
    uparameters_to_replace = unknowns(network)

    map_params = instantiate_param.(parameters_to_replace, Ref(params)) |> x -> filter(!isnothing, x)
    map_uparams = instantiate_uparam.(uparameters_to_replace, Ref(uparams)) |> x -> filter(!isnothing, x)

    (map_params, map_uparams)
end

end
