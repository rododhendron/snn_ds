module Plots

using CairoMakie, ComponentArrays
using CairoMakie: Axis

using ..Utils

function get_slice_xy(x, y; start=0, stop=0)
    stop = stop == 0 ? maximum(x) : stop
    first_x_idx = findfirst(el -> el >= start, x)
    last_x_idx = findlast(el -> el <= stop, x)
    (x[first_x_idx:last_x_idx], y[first_x_idx:last_x_idx])
end

function make_fig(; xlabel="", ylabel="", title="")
    f = Figure(size=(1600, 700))
    ax1 = Axis(f[1, 1],
        title=title,
        xlabel=xlabel,
        ylabel=ylabel
    )
    ax2 = Axis(f[1, 1],
        yaxisposition=:right,
        ylabel="input_current, in A"
    )
    linkxaxes!(ax1, ax2)
    # hidespines!(ax2)
    # hidedecorations!(ax2)
    (f, ax1, ax2)
end

function plot_neuron_value(time, value, p, input_current; start=0, stop=0, xlabel="", ylabel="", title="", name="", tofile=true, is_voltage=false)
    f, ax, ax2 = make_fig(; xlabel=xlabel, ylabel=ylabel, title=title)
    sliced_time, sliced_value = get_slice_xy(time, value, start=start, stop=stop)
    is_voltage ? hlines!(ax, [p.vthr, p.vrest]; color=1:2) : nothing
    if !isnothing(input_current)
        sliced_time, sliced_current = get_slice_xy(time, input_current, start=start, stop=stop)
        lines!(ax2, sliced_time, sliced_current, color=(:black, 0.3))
    end
    lines!(ax, sliced_time, sliced_value)
    xlims!(ax2, (start, stop))
    tofile ? save(name, f) : f
end

function sol_to_spikes(spikes_x_vec::Vector, y_value)::Vector
    spikes_values = ones(Int, size(spikes_x_vec))
    spikes_in_window = spikes_values * y_value
end

function plot_spikes(spikes; start=0, stop=0, xlabel="", ylabel="", title="", name="", tofile=true, color=:grey)
    f, ax, ax1 = make_fig(; xlabel=xlabel, ylabel=ylabel, title=title)
    xlims!(ax1, (start, stop))
    spikes_x = spikes
    int_range = 1:size(spikes, 1)
    spikes_y = sol_to_spikes.(spikes, int_range)
    x = spikes_x |> filter(!isempty)
    y = spikes_y |> filter(!isempty)
    @show size(x)
    for i in 1:size(x, 1)
        scatter!(ax, x[i], y[i]; color=color, markersize=5)
    end
    tofile ? save(name, f) : f
end

fetch_tree_neuron_value(neuron_type::String, i::Int, val::String, tree::ParamTree) = fetch_tree(["$(neuron_type)_$i", val], tree::ParamTree, false) |> first
function plot_excitator_value(i, sol, start, stop, name_interpol, tree::ParamTree)
    e_v = fetch_tree_neuron_value("e_neuron", i, "v", tree)
    e_Ib = fetch_tree_neuron_value("e_neuron", i, "Ib", tree)
    e_vtarget = fetch_tree_neuron_value("e_neuron", i, "vtarget_exc", tree)
    e_vrest = fetch_tree_neuron_value("e_neuron", i, "vrest", tree)
    ps = ComponentVector(vtarget=e_vtarget, v_rest=e_vrest)
    Plots.plot_neuron_value(sol.t, sol[e_v], ps, sol[e_Ib]; start=start, stop=stop, title="voltage of e $i", name=name_interpol("voltage_e_$i.png"))
end

end
