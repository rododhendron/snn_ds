module SNN

using ModelingToolkit

# include submodules
include("device.jl")
include("neuron.jl")
include("params.jl")
include("utils.jl")
include("plots.jl")

using .Neuron
using .Device
using .Utils
using .Params
using .Plots
end
