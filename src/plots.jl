module Plots

using CairoMakie, ComponentArrays
using CairoMakie: Axis
using Makie.GeometryBasics
using Transducers
# using Colors

using ..Utils

function get_slice_xy(x, y; start=0, stop=0)
    stop = stop == 0 ? maximum(x) : stop
    first_x_idx = findfirst(el -> el >= start, x)
    last_x_idx = findlast(el -> el <= stop, x)
    (x[first_x_idx:last_x_idx], y[first_x_idx:last_x_idx])
end

function make_fig(; xlabel="", ylabel="", title="", height=700, width=1600, yticks=Makie.automatic, call_ax2=true, plot_stims=false, schedule=[], yscale=identity)
    f = Figure(size=(width, height))
    ax1 = Axis(f[1, 1],
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        yticks=yticks,
        xticks=Makie.automatic,
        yscale=yscale
    )
    if plot_stims
        ax_stims = Axis(f[2, 1],
        )
        linkxaxes!(ax1, ax_stims)
        recs = [Rect(stim[1], 1, stim[2], stim[2]) for stim in eachcol(schedule)]
        # schedule is (t_starts, onset_period, idx_target)
        # stim_polys = poly!(ax_stims, recs, color=Int.(schedule[3, :]), colormap=Makie.Categorical(:rainbow))
        unique_stim = Int.(unique(schedule[3, :]))
        colormap = cgrad(:darktest, size(unique_stim, 1), categorical=true)
        groups_sch = schedule[3, :]
        stim_polys = poly!(ax_stims, recs, color=Int.(groups_sch), colormap=colormap)
        cbar = Colorbar(f[2, 2], label="Stimuli class", limits=(-1, 1), colormap=colormap)#, colormap=cgrad(:rainbow, categorical=true))
        nticks = size(unique_stim, 1)
        ticks = -1:2/nticks:1 |> Consecutive(2; step=1) |> Map(x -> x[1] + (x[2] - x[1]) / 2) |> collect # make ticks evenly distributed along y axis
        cbar.ticks = (ticks, string.(unique_stim |> sort))
        rowsize!(f.layout, 2, 50)
        hideydecorations!(ax_stims)
    else
        ax_stims = nothing
    end
    if call_ax2
        ax2 = Axis(f[1, 1],
            yaxisposition=:right,
            ylabel="input_current, in A",
        )
        linkxaxes!(ax1, ax2)
        (f, ax1, ax2, ax_stims)
    else
        (f, ax1, nothing, ax_stims)
    end
end

function plot_agg_value(times, values; xlabel="", ylabel="", title="", name="", tofile=true, is_voltage=false, schedule=[], offset=0)
    f, ax, ax2, ax_stims = make_fig(; xlabel=xlabel, ylabel=ylabel, title=title)

    @show size(times)
    @show size(values)
    colormap = :Set1_5
    groups = unique(schedule[3, :])
    # color = resample_cmap(colormap, size(groups, 1))
    color = [:blue, :red]
    for group in 1:length(groups)
        idxs = findall(x -> x == group, schedule[3, :])
        @show idxs
        g_times = times[idxs]
        g_values = values[idxs]
        @show size(g_times)
        @show size(g_values)
        points = []
        for i in axes(g_times, 1)
            points_vec = (g_times[i], g_values[i])
            push!(points, points_vec)
        end
        series!(ax, points; solid_color=(color[group], 0.3))
    end
    vlines!(ax, offset; color=:grey)
    tofile ? save(name, f) : f
end


function plot_neuron_value(time, value, p, input_current, offset; start=0, scale=identity, stop=0, xlabel="", ylabel="", title="", name="", tofile=true, is_voltage=false, schedule=[])
    f, ax, ax2, ax_stims = make_fig(; xlabel=xlabel, ylabel=ylabel, title=title, schedule=schedule, plot_stims=true, yticks=Makie.automatic, yscale=scale)
    sliced_time, sliced_value = get_slice_xy(time, value, start=start, stop=stop)
    if is_voltage
        hlines!(ax, p.vrest; color=:purple, label="vrest")
        hlines!(ax, p.vthr; color=:yellow, label="vthr")
    end
    vlines!(ax, [offset]; color=:grey, linestyle=:dashdot)
    if !isnothing(input_current)
        sliced_time, sliced_current = get_slice_xy(time, input_current, start=start, stop=stop)
        lines!(ax2, sliced_time, sliced_current, color=(:black, 0.3))
    end
    lines!(ax, sliced_time, sliced_value)
    xlims!(ax2, (start, stop))
    # axislegend(position=:rt)
    tofile ? save(name, f) : f
end

function sol_to_spikes(spikes_x_vec::Vector, y_value)::Vector
    spikes_values = ones(Int, size(spikes_x_vec))
    spikes_in_window = spikes_values * y_value
end

function plot_spikes((spikes_e, spikes_i); start=0, stop=0, xlabel="", ylabel="", title="", name="", color=(:grey, :grey), height=Makie.automatic, tofile=true, schedule=[])
    spikes = vcat(spikes_e, spikes_i)
    yticks = size(spikes, 1) > 0 ? (1:size(spikes, 1)) : Makie.automatic
    f, ax, ax1, ax_stims = make_fig(; xlabel=xlabel, ylabel=ylabel, title=title, height=height, yticks=yticks, call_ax2=false, schedule=schedule, plot_stims=true)
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
        if isempty(spikes_i)
            dot_color = :black
        end
        scatter!(ax, x[i], y[i]; color=dot_color, markersize=6)
    end
    tofile ? save(name, f) : f
end

fetch_tree_neuron_value(neuron_type::String, i::Int, val::String, tree::ParamTree) = fetch_tree(["$(neuron_type)_$i", val], tree::ParamTree, false) |> first
function plot_excitator_value(i, sol, start, stop, name_interpol, tree::ParamTree, offset, schedule, tofile=false)
    e_v = fetch_tree_neuron_value("e_neuron", i, "v", tree)
    # e_Ib = fetch_tree_neuron_value("e_neuron", i, "Ib", tree)
    e_vtarget = fetch_tree_neuron_value("e_neuron", i, "vtarget_exc", tree)
    e_vrest = fetch_tree_neuron_value("e_neuron", i, "vrest", tree)
    ps = ComponentVector(vtarget=e_vtarget, v_rest=e_vrest)
    Plots.plot_neuron_value(sol.t, sol[e_v], ps, nothing, offset; start=start, stop=stop, title="voltage of e $i", name=name_interpol("voltage_e_$i.png"), schedule=schedule, tofile=tofile)
end

function compute_moving_average(time_dim::Vector{Float64}, values::Vector{Float64}, time_window::Float64)
    starts = time_dim .- time_window / 2 .|> x -> max(x, first(time_dim))
    stops = time_dim .+ time_window / 2 .|> x -> min(x, last(time_dim))
    ranges = zip(starts, stops) |> collect
    counts = ranges |> Map(x -> count(el -> x[1] <= el < x[2], values)) |> collect
    counts .* (1/time_window)
end

function plot_spike_rate(i, sol, start, stop, name_interpol, tree::ParamTree, offset, schedule, tofile=false; time_window=1.0)
    e_r = fetch_tree_neuron_value("e_neuron", i, "R", tree)

    s_bool = sol[e_r] |> Utils.get_spikes_from_r .|> Bool

    spikes = Utils.get_spike_timings(s_bool, sol)
    spike_rate = compute_moving_average(sol.t, spikes, time_window)

    Plots.plot_neuron_value(sol.t, spike_rate, nothing, nothing, offset; start=start, stop=stop, title="spike rate of e $i", name=name_interpol("spike_rate_e_$i.png"), schedule=schedule, is_voltage=false, tofile=tofile)
end

function plot_isi(i, sol, start, stop, name_interpol, tree::ParamTree, offset, schedule, tofile=false)
    e_r = fetch_tree_neuron_value("e_neuron", i, "R", tree)
    s_bool = sol[e_r] |> Utils.get_spikes_from_r .|> Bool
    spikes = Utils.get_spike_timings(s_bool, sol)

    isi = diff(spikes)
    Plots.plot_neuron_value(spikes[3:end], isi[2:end], nothing, nothing, offset; scale=Makie.pseudolog10, start=start, stop=stop, title="Interspike intervals of e $i", name=name_interpol("isi_e_$i.png"), schedule=schedule, is_voltage=false, tofile=tofile)
end

function get_trials_from_schedule(schedule_p)
    trial_starts = schedule_p[1, :]
    trial_periods = diff(trial_starts)
    mean_period = reduce(+, trial_periods) / length(trial_periods)
    trial_stops = trial_starts .+ mean_period
    zip(trial_starts, trial_stops) |> collect
end

function trial_to_idxs(trial_ranges, sol_t::Vector{Float64})
    trial_ranges |> Map(x -> findall(ts -> x[1] < ts < x[2], sol_t)) |> collect
end

function plot_aggregated_rate(i, sol, name_interpol, tree::ParamTree, schedule_p, tofile=false)
    e_r = fetch_tree_neuron_value("e_neuron", i, "R", tree)
    e_v = fetch_tree_neuron_value("e_neuron", i, "v", tree)

    spikes = sol[e_r] |> Utils.get_spikes_from_r .|> Bool |> sp -> Utils.get_spike_timings(sp, sol)
    trials = get_trials_from_schedule(schedule_p) |> x -> trial_to_idxs(x, sol.t)
    trial_values = trials |> Map(trial -> sol[e_v][trial]) |> collect
    trial_starts = schedule_p[1, :]
    trial_times = trials |> Map(trial -> sol.t[trial]) |> collect
    trial_t_offsetted = [trial_times[i] .- Ref(trial_starts[i]) for i in axes(trial_starts, 1)]

    Plots.plot_agg_value(trial_t_offsetted, trial_values; title="trials of e $i", name=name_interpol("trials_$i.png"), tofile=tofile, schedule=schedule_p)
end

function plot_adaptation_value(i, sol, start, stop, name_interpol, tree::ParamTree, offset, schedule, tofile=false)
    e_w = fetch_tree_neuron_value("e_neuron", i, "w", tree)
    # e_Ib = fetch_tree_neuron_value("e_neuron", i, "Ib", tree)
    Plots.plot_neuron_value(sol.t, sol[e_w], nothing, nothing, offset; start=start, stop=stop, title="adaptation of e $i", name=name_interpol("adaptation_e_$i.png"), schedule=schedule, is_voltage=false, tofile=tofile)
end

function plot_heat_map_connection()
   	heatfig = Figure(size=(900,700))
	ticks = LinearTicks(size(heatmap_connect, 1))
	ax_heat = heatfig[1, 1] = Axis(heatfig; title="Connectivity matrix between neurons by synapse type", xlabel="Post synaptic neuron", ylabel="Pre synaptic neuron", xticks=ticks, yticks=ticks)
	elem_1 = [PolyElement(color = :red, linestyle = nothing)]
	elem_2 = [PolyElement(color = :blue, linestyle = nothing)]
	heatmap!(ax_heat, transpose(heatmap_connect); colormap=[:blue, :red], nan_color=:white)
	heatfig[1, 2] = Legend(heatfig, [elem_1, elem_2], ["AMPA", "GABAa"], framevisible = false)
	heatfig
end

end
