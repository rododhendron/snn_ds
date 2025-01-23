using DrWatson # have to install it globally
quickactivate(@__DIR__) # needed to load project env

include("../src/SNN.jl")

using Symbolics, ModelingToolkit, DifferentialEquations, ComponentArrays, LinearAlgebra, LinearSolve

spikes_times = Any[[1.8347263278219343, 3.0353827394091977, 4.635264114675274, 6.63358900666743], [0.23359065033107723, 1.4396074582231257, 3.434202770188318, 4.249900343805269, 5.439416357629863, 7.0338541936366346, 8.239381927138353, 9.43838178784136], [0.23610399220129302, 1.4446078165667058, 1.8412231434156683, 3.0416203052785478, 3.440546875076382, 4.252926663395414, 4.640398826792478, 5.441998822294924, 6.636098032812983, 7.038754485822805, 8.245621047798412, 9.440890261116857]]
(start, stop) = (0, 10)

SNN.Plots.plot_spikes((spikes_times, []); start=start, stop=stop, color=(:red, :blue), height=400, title="Network activity", xlabel="time (in s)", ylabel="neuron index", name="test_plot.png")
