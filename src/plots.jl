module Plots

using CairoMakie

function get_slice_xy(x, y; start=0, stop=0)
    stop = stop == 0 ? maximum(a) : stop
    first_x_idx = findfirst(el -> el >= start, x)
    last_x_idx = findlast(el -> el <= stop, x)
    (x[first_x_idx:last_x_idx], y[first_x_idx:last_x_idx])
end

function make_fig(; xlabel="", ylabel="", title="")
    f = Figure(size=(1600, 700))
    ax = Axis(f[1, 1],
        title=title,
        xlabel=xlabel,
        ylabel=ylabel
    )
    (f, ax)
end

function plot_neuron_value(time, value, p, spikes; start=0, stop=0, xlabel="", ylabel="", title="", name="", tofile=true, is_voltage=false)
    f, ax = make_fig(; xlabel=xlabel, ylabel=ylabel, title=title)
    sliced_time, sliced_value = get_slice_xy(time, value, start=start, stop=stop)
    spikes_in_window = [spike for spike in spikes if spike != 0 && spike < stop]
    is_voltage ? hlines!(ax, [p.vthr, p.vrest]; color=1:2) : nothing
    vlines!(ax, [offset]; color=:grey, linestyle=:dashdot)
    vlines!(ax, spikes_in_window; color=:red, linestyle=:dot)
    lines!(ax, sliced_time, sliced_value)
    xlims!(ax, (start, stop))
    tofile ? save(name, f) : f
end

function sol_to_spikes(spikes)
    dims = size(spikes)
    int_range = 1:dims[1]
    spikes_in_window = Int.(spikes) .|> x -> x * int_range
    @show spikes_in_window
end

function plot_spikes(spikes; start=0, stop=0, xlabel="", ylabel="", title="", name="")
    f, ax = make_fig(; xlabel=xlabel, ylabel=ylabel, title=title)
    xlims!(ax, (start, stop))
    spikes_in_window = sol_to_spikes(spikes)
    scatter!(ax, spikes_in_window; color=:grey)
    f
end

end
