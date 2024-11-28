module Neuron
using ModelingToolkit, Parameters
using ModelingToolkit: t_nounits as t, D_nounits as D

start_input = 200e-3
input_duration = 200e-3
step_fn(t, iv) = start_input < t < start_input + input_duration ? iv : 0
@register_symbolic step_fn(t, iv)

@mtkmodel Soma begin # AdEx neuron from https://journals.physiology.org/doi/pdf/10.1152/jn.00686.2005
    @variables begin
        v(t) = -65e-3
        w(t) = 0
        Ie(t), [input = true]
        # Ii(t), [input = true]
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
        D(v) ~ (Je * (vrest - v) + delta * exp((v - vthr) / delta) - w) + Ie + Ib / Cm # voltage dynamic
        D(w) ~ (-w + a * (v - vrest)) / TauW # adaptation dynamic
        Ib ~ step_fn(t, input_value)
        D(R) ~ 0
    end
    @continuous_events begin
        [v ~ vspike] => [
            v ~ vrest,
            R ~ R + 1,
            w ~ w + b
        ]
    end
end

@mtkmodel SynapseAMPA begin # AMPA synapse https://neuronaldynamics.epfl.ch/online/Ch3.S1.html#Ch3.E2
    @parameters begin
        tau_ampa
        vtarget
        inc_gsyn
    end
    @variables begin
        g_syn(t) = 0
    end
    # have to bind I on connect
    @equations begin
        D(g_syn) ~ -g_syn / tau_ampa
    end
end

@mtkmodel SynapseGABAa begin # AMPA synapse https://neuronaldynamics.epfl.ch/online/Ch3.S1.html#Ch3.E2
    @parameters begin
        gmax
        tau
    end
    @variables begin
        g_syn(t)
    end
    # have to bind I on connect
    @equations begin
        D(g_syn) ~ -g_syn / tau
    end
end


# Base neuron parameters
Base.@kwdef mutable struct AdExNeuronParams
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
    vtarget::Float64 = 0
    vspike::Float64 = 0
    inc_gsyn::Float64 = 1e-9
end
Base.@kwdef mutable struct AdExNeuronUParams
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

function get_synapse(_synapse_type::SIMPLE_E, params) # _ before variable is a convention for a unused variable in function body ; just used for type dispatch
    @variables g_syn = 0
    [g_syn ~ params.gl * params.a]
end

function make_neuron(params, soma_model, tspan, name)
    @named soma = soma_model(; name=Symbol("soma"))
    @named ampa_syn = SynapseAMPA(; name=Symbol("ampa_syn"))
    # @mtkbuild gabaa_syn = SynapseGABAa()

    neuron = compose(
        ODESystem(
            [
                soma.Ie ~ ampa_syn.g_syn * (soma.v - ampa_syn.vtarget)
            ], t; name=name
        ), soma, ampa_syn
    )
    neuron
end

function connect_neurons(pre_neurons, post_neuron)
    callbacks = []
    for i in 1:length(pre_neurons)
        push!(callbacks, [pre_neurons[i].soma.v ~ pre_neurons[i].soma.vspike] => [
            post_neuron.ampa_syn.g_syn ~ post_neuron.ampa_syn.g_syn + post_neuron.ampa_syn.inc_gsyn
        ])
    end
    callbacks
end

function make_network(neurons)
    all_callbacks = []
    for i in 1:length(neurons)
        pre_neurons = [neurons[1:i-1]; neurons[i+1:end]] # forward connections
        append!(all_callbacks, connect_neurons(pre_neurons, neurons[i]))
    end
    @named network = compose(ODESystem([], t; continuous_events=all_callbacks, name=:connected_neurons), neurons...)
end

function instantiate_params(sys_index, params)
    param_value = hasproperty(params, Symbol(sys_index[end])) ? getproperty(params, Symbol(sys_index[end])) : nothing
    # isnothing(param_value) ? nothing : String(Symbol(join(sys_index, "__"))) => param_value
    isnothing(param_value) ? nothing : sys_index[end] => param_value
end

get_varsym_from_syskey(param::SymbolicUtils.BasicSymbolic) = Symbol(split(param |> Symbol |> String, "₊")[end])
get_varsym_from_syskey(param::SubString) = Symbol(split(param, "₊")[end])

function instantiate_param(parameter, params)
    if hasproperty(params, get_varsym_from_syskey(parameter))
        parameter => getproperty(params, get_varsym_from_syskey(parameter))
    end
end

function instantiate_uparam(parameter, uparams)
    replaced_parameter = split(parameter |> Symbol |> String, "(")[1]
    if hasproperty(uparams, get_varsym_from_syskey(replaced_parameter))
        parameter => getproperty(uparams, get_varsym_from_syskey(replaced_parameter))
    end
end

function map_params(network, params, uparams)
    parameters_to_replace = parameters(network)
    uparameters_to_replace = unknowns(network)

    map_params = instantiate_param.(parameters_to_replace, Ref(params)) |> x -> filter(!isnothing, x)
    map_uparams = instantiate_uparam.(uparameters_to_replace, Ref(uparams)) |> x -> filter(!isnothing, x)

    (map_params, map_uparams)
end

end
