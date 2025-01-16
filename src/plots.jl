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

function make_fig(; xlabel="", ylabel="", title="", height=700, width=1600, yticks=Makie.automatic, call_ax2=true)
    f = Figure(size=(width, height))
    ax1 = Axis(f[1, 1],
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        yticks=yticks,
        xticks=Makie.automatic
    )
    if call_ax2
        ax2 = Axis(f[1, 1],
            yaxisposition=:right,
            ylabel="input_current, in A"
        )
        linkxaxes!(ax1, ax2)
        (f, ax1, ax2)
    else
        (f, ax1, nothing)
    end
end


function plot_neuron_value(time, value, p, input_current, offset; start=0, stop=0, xlabel="", ylabel="", title="", name="", tofile=true, is_voltage=false)
    f, ax, ax2 = make_fig(; xlabel=xlabel, ylabel=ylabel, title=title)
    sliced_time, sliced_value = get_slice_xy(time, value, start=start, stop=stop)
    is_voltage ? hlines!(ax, [p.vthr, p.vrest]; color=1:2) : nothing
    vlines!(ax, [offset]; color=:grey, linestyle=:dashdot)
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

function plot_spikes((spikes_e, spikes_i); start=0, stop=0, xlabel="", ylabel="", title="", name="", color=(:grey, :grey), height=600, tofile=true)
    spikes = vcat(spikes_e, spikes_i)
    yticks = size(spikes, 1) > 0 ? LinearTicks(size(spikes, 1)) : Makie.automatic
    f, ax, ax1 = make_fig(; xlabel=xlabel, ylabel=ylabel, title=title, height=height, yticks=yticks, call_ax2=false)
    xlims!(ax, (start, stop))
    size_e = size(spikes_e, 1)
    spikes_x = spikes
    int_range = 1:size(spikes, 1)
    spikes_y = sol_to_spikes.(spikes, int_range)
    x = spikes_x
    y = spikes_y
    for i in 1:size(x, 1)
        dot_color = color[1]
        if i > size_e
            dot_color = color[2]
        end
        scatter!(ax, x[i], y[i]; color=dot_color, markersize=8)
    end
    tofile ? save(name, f) : f
end

fetch_tree_neuron_value(neuron_type::String, i::Int, val::String, tree::ParamTree) = fetch_tree(["$(neuron_type)_$i", val], tree::ParamTree, false) |> first
function plot_excitator_value(i, sol, start, stop, name_interpol, tree::ParamTree, offset)
    @show i
    e_v = fetch_tree_neuron_value("e_neuron", i, "v", tree)
    e_Ib = fetch_tree_neuron_value("e_neuron", i, "Ib", tree)
    e_vtarget = fetch_tree_neuron_value("e_neuron", i, "vtarget_exc", tree)
    e_vrest = fetch_tree_neuron_value("e_neuron", i, "vrest", tree)
    ps = ComponentVector(vtarget=e_vtarget, v_rest=e_vrest)
    Plots.plot_neuron_value(sol.t, sol[e_v], ps, sol[e_Ib], offset; start=start, stop=stop, title="voltage of e $i", name=name_interpol("voltage_e_$i.png"))
end

end
