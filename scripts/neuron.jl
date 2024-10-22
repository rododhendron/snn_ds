using Symbolics, DifferentialEquations, SciMLBase, Statistics
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D, inputs
using LabelledArrays
using CairoMakie, Latexify
using SignalAnalysis
using Distributions

const analysis_path::String = "analysis/"
const b::Int16 = 1
const offset::Float64 = 2000

const tstart::Float64 = 0.0
const tstop::Float64 = 5000.0
const tspan = (tstart, tstop)

Params = @SLVector Float64 (:Je, :El, :delta, :vthr, :Trefr, :Cm, :vrest, :TauW, :w0, :a)
p = Params(10, -65, 2, -50, 5, 200, -65, 500, 0, 4)


input_value = 65
n_spikes = 3

function get_slice_xy(x, y; start=0, stop=0)
    stop = stop == 0 ? maximum(a) : stop
    first_x_idx = findfirst(el -> el >= start, x)
    last_x_idx = findlast(el -> el <= stop, x)
    (x[first_x_idx:last_x_idx], y[first_x_idx:last_x_idx])
end

function make_fig(x, y; xlabel="", ylabel="", title="")
    f = Figure(size=(1600, 1200))
    ax = Axis(f[1, 1],
        title=title,
        xlabel=xlabel,
        ylabel=ylabel
    )
    (f, ax)
end

function plot_neuron_value(time, value, p; start=0, stop=0, xlabel="", ylabel="", title="", name="", tofile=true)
    f, ax = make_fig(time, value; xlabel=xlabel, ylabel=ylabel, title=title)
    sliced_time, sliced_value = get_slice_xy(time, value, start=start, stop=stop)
    hlines!(ax, [p.vthr, p.vrest]; color=1:2)
    vlines!(ax, [offset]; color=:grey)
    lines!(ax, sliced_time, sliced_value)
    tofile ? save(name, f) : f
end

function get_spikes_from_voltage(t, v, v_target)
    tol = 1e-3
    spike_idx = findall(x -> v_target - tol < x < v_target + tol, v)
    return (t[spike_idx], v[spike_idx])
end

function input_fn(t)
    if t < offset
        return 0
    else
        return input_value * n_spikes
    end
end
@register_symbolic input_fn(t)

function gen_spike_train(lambda, n_events, offset)
    input_spike_times = [offset]
    for event in 1:n_events
        interval = rand(Exponential(1 / lambda))
        push!(input_spike_times, interval + last(input_spike_times))
    end
    input_spike_times
end
# make PSP exponential decay 5ms and AMPA jump

spikes = gen_spike_train(0.1, 10, offset)
@show spikes

@variables v(t) = p.vrest w(t) = 0

eqs = [
    D(v) ~ (p.Je * (p.El - v) + p.delta * exp((v - p.vthr) / p.delta) - w) / p.Cm
    D(w) ~ (-w + p.a * (v - p.El)) / p.TauW
]
spike_condition = [v ~ -50]
spike_affect = [v ~ -65, w ~ w + b * 1]

discretes_spike_inputs = [[2000.0] => [v ~ v + input_value]]

input_spike_rate(u, p, t) = t > 2000.0 ? 0.05 : 0
integrate_spike!(integrator) = (integrator.u[1] += 5)
crj = VariableRateJump(input_spike_rate, integrate_spike!)

@named neuron = ODESystem(eqs, t; tspan=tspan, continuous_events=spike_condition => spike_affect)

simplified_model = structural_simplify(dae_index_lowering(neuron))
# simplified_model = complete(neuron)

prob = ODEProblem(simplified_model, [], tspan)
jumprob = JumpProblem(prob, Direct(), crj)
sol = solve(jumprob, Vern6())

before_offset = offset - 50
after_offset = offset + 200

plot_neuron_value(sol.t, sol[v], p;
    start=before_offset,
    stop=after_offset,
    title="Neuron voltage along time.",
    xlabel="time (in ms), starting 200ms before input",
    ylabel="voltage (in mV)",
    name=analysis_path * "neuron_vt.png"
)

plot_neuron_value(sol.t, sol[w], p;
    start=before_offset,
    stop=after_offset,
    title="Neuron adaptation along time.",
    xlabel="time (in ms), starting 200ms before input",
    ylabel="adaptation",
    name=analysis_path * "neuron_wt.png"
)

found_spiked_t, found_spiked_v = get_spikes_from_voltage(sol.t, sol[v], p.vthr)

window_size = 5

mean_local_frequencies = []
for window_center in 3:length(found_spiked_t)-2
    window = found_spiked_t[window_center-2:window_center+2]
    distances = [window[i+1] - window[i] for i in 1:length(window)-1]
    mean_period = mean(distances)
    push!(mean_local_frequencies, 1 / mean_period * 1e3)
end
