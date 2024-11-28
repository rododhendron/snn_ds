using DrWatson # have to install it globally
quickactivate(@__DIR__) # needed to load project env

include("../src/neuron.jl")

using Symbolics, ModelingToolkit, DifferentialEquations

using .Neuron

tspan = (0, 1)
n_neurons = 1000
i_neurons = 2
e_neurons = 8

@time params = Neuron.AdExNeuronParams(;input_value=1e-9)
@time neurons = [neuron = Neuron.make_neuron(params, Neuron.Soma, tspan, Symbol("neuron_$(i)")) for i in 1:n_neurons]

@time network = Neuron.make_network(neurons)

simplified_model = network
# @time simplified_model = Neuron.structural_simplify(network)
println("jac")
@time calculate_jacobian(simplified_model)

# infere params
@time uparams = Neuron.AdExNeuronUParams()

@time iparams, iuparams = Neuron.map_params(simplified_model, params, uparams)

# a = [iparams[1:end-1]; simplified_model.neuron_3.soma.input_value => 1e-9]
# resolve
@time prob = ODEProblem(simplified_model, iuparams, tspan, iparams, jac=true, sparse=true)

function n_times_solve(problem, n)
    for i in 1:n
        solve(prob, Vern6(); abstol=1e-7, reltol=1e-7)
    end
end

sol = solve(prob, Vern6(); abstol=1e-7, reltol=1e-7)
@time time_to_run = n_times_solve(prob, 10)
