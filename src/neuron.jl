module Neuron
using ModelingToolkit, Parameters, StringDistances
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

function make_neuron(params, soma_model, tspan, name)::ODESystem
    @named soma = soma_model(; name=Symbol("soma"))
    @named ampa_syn = SynapseAMPA(; name=Symbol("ampa_syn"))
    @named gabaa_syn = SynapseGABAa(; name=Symbol("gabaa_syn"))

    neuron = compose(
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

function connect_neurons(pre_neuron::ODESystem, post_neurons::Vector{ODESystem}, synapse_type::SynapseType, prob::Float64)::Pair{Vector{Equation},Vector{Equation}}
    callbacks::Vector{Equation} = []
    @inbounds for i in 1:length(post_neurons)
        if rand() < prob && pre_neuron.name != post_neurons[i].name
            push!(callbacks, get_synapse_eq(synapse_type, post_neurons[i]))
        end
    end
    make_events(callbacks, pre_neuron)
end

function connect_many_neurons(pre_neurons::Vector{ODESystem}, post_neurons::Vector{ODESystem}, synapse_type::SynapseType, prob::Float64)::Vector{Pair{Vector{Equation},Vector{Equation}}}
    all_callbacks::Vector{Pair{Vector{Equation},Vector{Equation}}} = []
    @inbounds for i in 1:length(pre_neurons)
        push!(all_callbacks, connect_neurons(pre_neurons[i], post_neurons, synapse_type, prob))
    end
    all_callbacks
end

function make_network(neurons::Vector{ODESystem}, connections::Vector{Pair{Vector{Equation},Vector{Equation}}})::ODESystem
    @mtkbuild network = compose(ODESystem([], t; continuous_events=connections, name=:connected_neurons), neurons)
end

function instantiate_params(sys_index, params)::Union{nothing,Pair{Symbol,Union{Float64,Int64}}}
    param_value = hasproperty(params, Symbol(sys_index[end])) ? getproperty(params, Symbol(sys_index[end])) : nothing
    # isnothing(param_value) ? nothing : String(Symbol(join(sys_index, "__"))) => param_value
    isnothing(param_value) ? nothing : sys_index[end] => param_value
end

get_varsym_from_syskey(param::SymbolicUtils.BasicSymbolic)::Symbol = Symbol(split(param |> Symbol |> String, "₊")[end])
get_varsym_from_syskey(param::SubString)::Symbol = Symbol(split(param, "₊")[end])

function instantiate_param(parameter::SymbolicUtils.BasicSymbolic, params::AdExNeuronParams)::Pair{SymbolicUtils.BasicSymbolic,Union{Float64,Int64}}
    if hasproperty(params, get_varsym_from_syskey(parameter))
        parameter => getproperty(params, get_varsym_from_syskey(parameter))
    end
end

function instantiate_uparam(parameter::SymbolicUtils.BasicSymbolic, uparams::AdExNeuronUParams)::Pair{SymbolicUtils.BasicSymbolic,Union{Float64,Int64}}
    replaced_parameter = split(parameter |> Symbol |> String, "(")[1]
    if hasproperty(uparams, get_varsym_from_syskey(replaced_parameter))
        parameter => getproperty(uparams, get_varsym_from_syskey(replaced_parameter))
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
