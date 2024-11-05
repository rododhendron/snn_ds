using ModelingToolkit

function get_soma(mparams, uparams)
    @modeling
    expr = instantiate_params_as_symbols(mparams)
   	try
		Core.eval(@__MODULE__, expr)
	catch e
		println("already evaluated")
	end
   	@variables v(t) = mparams.vrest w(t) = mparams.w0
	eqs = [
       	D(v) ~ (Je * (vrest - v) + delta * exp((v - vthr) / delta) - w) + I / Cm
       	D(w) ~ (-w + a * (v - vrest)) / TauW
	]
end
# Base neuron parameters
Base.@kwdef mutable struct AdExNeuronParams <: AbstractNeuronParams
    vrest::Float64 = -65.0  # Resting membrane potential (mV)
    vthr::Float64 = -50.0   # Spike threshold (mV)
    w0::Float64 = 0.0       # Initial adaptation current
    Je::Float64 = 1.0       # Membrane time constant
    delta::Float64 = 2.0    # Spike slope factor
    Cm::Float64 = 1.0       # Membrane capacitance
    TauW::Float64 = 20.0    # Adaptation time constant
    a::Float64 = 0.0        # Subthreshold adaptation
end

struct Params{S}
    gl::Float64
    a::Float64
    vrest::Float64
    delta::Float64
    vthr::Float64
    TauW::Float64
end

@enum NeuronType begin
    SIMPLE_E
    AMPA
end

function get_synapse(synapse_type::NeuronType{SIMPLE_E}, params)
    I = params.gl * params.a
end

function get_synapse(synapse_type::NeuronType{AMPA}, params)
    @variables I
    D(v) = params.gl * params.a * ()
end

function make_neuron(params, soma_fn, synapse_fn)
    @mtkmodel Neuron begin
        @parameters begin
        end
    end
    soma = soma_fn(params)
    neuron_ode = ODESystem(soma, t, tspan)
end

@component
