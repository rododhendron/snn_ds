function get_soma(mparams, uparams)
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
