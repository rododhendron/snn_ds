module Neuron
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

@mtkmodel Soma begin # AdEx neuron from https://journals.physiology.org/doi/pdf/10.1152/jn.00686.2005
    @variables begin
        v(t)
        w(t)
        I(t), [input=true]
        Ib(t), [input=true]
        R(t), [output=true]
    end
    @parameters begin
        Je
        vrest
        delta
        vthr
        Cm
        a
        TauW
    end
    @equations begin
        D(v) ~ (Je * (vrest - v) + delta * exp((v - vthr) / delta) - w) + I + Ib / Cm # voltage dynamic
        D(w) ~ (-w + a * (v - vrest)) / TauW # adaptation dynamic
        I ~ 0
        Ib ~ 0
        R ~ 0
    end
end

@mtkmodel SynapseAMPA begin # AMPA synapse https://neuronaldynamics.epfl.ch/online/Ch3.S1.html#Ch3.E2
    @parameters begin
        gmax
        tau
    end
    @variables begin
        g(t)
    end
    # have to bind I on connect
    @equations begin
        D(g) ~ (-gmax + (gmax - g)) / tau
    end
end


# Base neuron parameters
Base.@kwdef mutable struct AdExNeuronParams
    vrest::Float64 = -65.0e-3  # Resting membrane potential (V)
    vthr::Float64 = -50.0e-3   # Spike threshold (V)
    w0::Float64 = 0.0e-9       # Initial adaptation current
    Je::Float64 = 1.0e-9       # Membrane time constant (Siemens S)
    delta::Float64 = 2.0e-3    # Spike slope factor (V)
    Cm::Float64 = 1.0       # Membrane capacitance (Farad F)
    TauW::Float64 = 20.0    # Adaptation time constant (s)
    a::Float64 = 0.0        # Subthreshold adaptation (A)
    gl::Float64 = 20e-9     # leak conductance
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
    [I ~ params.gl * params.a]
end

function get_synapse(_synapse_type::AMPA, params)
    @variables I
    D(v) = params.gl * params.a * ()
end

function make_neuron(params, soma_model, synapse_fn, tspan)
    @mtkbuild soma = soma_model()
end

function connect_neurons(pre_neuron, post_neuron, synapse_type)
    synapse_eq_g = get_synapse(synapse_type) # get conductance equation from synapse type
    connected = compose([
        pre_neuron
    ])
end

end
