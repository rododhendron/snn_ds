module Plots

using CairoMakie, ComponentArrays
using CairoMakie: Axis
using Makie.GeometryBasics
using Transducers
using DataInterpolations
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

    colormap = :Set1_5
    groups = unique(schedule[3, :])
    # color = resample_cmap(colormap, size(groups, 1))
    color = [:blue, :red]
    for group in 1:length(groups)
        idxs = findall(x -> x == group, schedule[3, :])
        g_times = times[idxs]
        g_values = values[idxs]
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


function plot_neuron_value(time, value, p, input_current, offset; start=0, scale=identity, stop=0, xlabel="", ylabel="", title="", name="", tofile=true, is_voltage=false, schedule=[], multi=false, plot_stims=true)
    call_ax2 = isnothing(input_current) ? false : true
    f, ax, ax2, ax_stims = make_fig(; xlabel=xlabel, ylabel=ylabel, title=title, schedule=schedule, plot_stims=plot_stims, yticks=Makie.automatic, yscale=scale, call_ax2=call_ax2)
    if multi
        slices_xy = get_slice_xy.(Ref(time), value, start=start, stop=stop)
    else
        slices_xy = get_slice_xy(time, value, start=start, stop=stop)
    end
    if is_voltage
        hlines!(ax, p.vrest; color=:purple, label="vrest")
        hlines!(ax, p.vthr; color=:yellow, label="vthr")
    end
    if typeof(offset) == Float64
        offsets = [offset]
    else
        offsets = offset
    end
    vlines!(ax, offsets; color=:grey, linestyle=:dashdot)
    if !isnothing(input_current)
        sliced_time, sliced_current = get_slice_xy(time, input_current, start=start, stop=stop)
        lines!(ax2, sliced_time, sliced_current, color=(:black, 0.3))
        xlims!(ax2, (start, stop))
    end
    if multi
        if length(slices_xy) == 2
            colors = [:blue, :red]
        else
            colors = cgrad(:darktest, length(sliced_value))
        end
        for i in 1:length(slices_xy)
            lines!(ax, slices_xy[i][1], slices_xy[i][2], color=colors[i])
        end
    else
        (sliced_time, sliced_value) = slices_xy
        lines!(ax, sliced_time, sliced_value)
    end
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
    starts = time_dim .- time_window / 2 .|> x -> max(x, first(time_dim)) # get start if greater that first sol.t (check boundaries)
    stops = time_dim .+ time_window / 2 .|> x -> min(x, last(time_dim)) # same for last sol.t
    ranges = zip(starts, stops) |> collect
    counts = ranges |> Map(x -> count(el -> x[1] <= el < x[2], values)) |> collect # count the number of times value occurs between start and stop
    counts ./ time_window # convert in Hz
end

function plot_spike_rate(i, sol, start, stop, name_interpol, tree::ParamTree, offset, schedule, tofile=false; time_window=1.0)
    e_r = fetch_tree_neuron_value("e_neuron", i, "R", tree)

    s_bool = sol[e_r] |> Utils.get_spikes_from_r .|> Bool

    spikes = Utils.get_spike_timings(s_bool, sol)
    spike_rate = compute_moving_average(sol.t, spikes, time_window)

    Plots.plot_neuron_value(sol.t, spike_rate, nothing, nothing, offset; start=start, stop=stop, title="spike rate of e $i", name=name_interpol("spike_rate_e_$(i)_window_$(time_window).png"), schedule=schedule, is_voltage=false, tofile=tofile)
end

function plot_isi(i, sol, start, stop, name_interpol, tree::ParamTree, offset, schedule, tofile=false)
    e_r = fetch_tree_neuron_value("e_neuron", i, "R", tree)
    s_bool = sol[e_r] |> Utils.get_spikes_from_r .|> Bool
    spikes = Utils.get_spike_timings(s_bool, sol)

    isi = diff(spikes)
    Plots.plot_neuron_value(spikes[3:end], isi[2:end], nothing, nothing, offset; scale=Makie.pseudolog10, start=start, stop=stop, title="Interspike intervals of e $i", name=name_interpol("isi_e_$i.png"), schedule=schedule, is_voltage=false, tofile=tofile)
end

function get_trials_from_schedule(schedule_p; comp_std_dev=true)
    trial_starts = schedule_p[1, :]
    trial_periods = diff(trial_starts)
    mean_period = reduce(+, trial_periods) / length(trial_periods)
    if comp_std_dev
        # for each deviant trial idx
        dev_trials = findall(trial -> trial == 2, schedule_p[3, :])
        # fetch previous standard
        std_trials = dev_trials .- 1
        trials_to_take = vcat(std_trials, dev_trials) |> unique |> sort
        filtered_trial_starts = schedule_p[1, trials_to_take]
        trial_stops = filtered_trial_starts .+ mean_period
        zip(filtered_trial_starts, trial_stops) |> collect
    else
        trial_stops = trial_starts .+ mean_period
        zip(trial_starts, trial_stops) |> collect
    end
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

function plot_heatmap(values; xlabel="", ylabel="", title="", tofile=true, name="")
    # values is shape (vector x, y and z)
    heatfig = Figure(size=(900, 700))
    ax_heat = heatfig[1, 1] = Axis(heatfig; title=title, xlabel=xlabel, ylabel=ylabel)
    hm = heatmap!(ax_heat, values..., colormap=:thermal)
    Colorbar(heatfig[1, 2], hm)
    tofile ? save(name, heatfig) : heatfig
end

function plot_heat_map_connection(heatmap_connect)
    heatfig = Figure(size=(900, 700))
    ticks = LinearTicks(size(heatmap_connect, 1))
    ax_heat = heatfig[1, 1] = Axis(heatfig; title="Connectivity matrix between neurons by synapse type", xlabel="Post synaptic neuron", ylabel="Pre synaptic neuron", xticks=ticks, yticks=ticks)
    elem_1 = [PolyElement(color=:red, linestyle=nothing)]
    elem_2 = [PolyElement(color=:blue, linestyle=nothing)]
    heatmap!(ax_heat, transpose(heatmap_connect); colormap=[:blue, :red], nan_color=:white)
    heatfig[1, 2] = Legend(heatfig, [elem_1, elem_2], ["AMPA", "GABAa"], framevisible=false)
    heatfig
end

function compute_grand_average(sol, neuron_u, stim_schedule, method=:spikes; sampling_rate=200.0, start=0.0, stop=last(sol.t), offset=0.1, interpol_fn=AkimaInterpolation, time_window=0.1)
    interpolate_u(u) = interpol_fn(u, sol.t)
    trials = get_trials_from_schedule(stim_schedule)
    trials_starts = [trial[1] for trial in trials]
    @show size(trials_starts)
    groups = unique(stim_schedule[3, :]) .|> Int
    groups_stim_idxs = [findall(row -> row[3] == group && row[1] in trial_starts, stim_schedule) for group in groups]
    @show size(groups_stim_idxs[1])
    if method == :spikes
        s_bool = sol[neuron_u] |> Utils.get_spikes_from_r .|> Bool
        spikes_neuron = Utils.get_spike_timings(s_bool, sol)
        spike_rate = compute_moving_average(sol.t, spikes_neuron, time_window)
        interpolation_table = interpolate_u(spike_rate)
    elseif method == :value
        interpolation_table = interpolate_u(sol[neuron_u])
        # for each trial, get list of resampled times
    end
    trial_time_delta = first(trials)[2] - first(trials)[1] + offset
    trials_times = [collect(trial_t[1]-offset:1/sampling_rate:trial_t[2]) for trial_t in trials] |> Map(x -> x[1:floor(Int, sampling_rate * trial_time_delta)]) |> collect
    # get corresponding values
    sampled_values = trials_times |> Map(trial_times -> interpolation_table.(trial_times)) |> tcollect
    # @show sampled_values
    @assert size(trials_times, 1) == size(sampled_values, 1) == size(stim_schedule, 2)
    grouped_trials = groups_stim_idxs |>
                     Map(trial_idxs -> sum(sampled_values[trial_idxs]) ./ length(sampled_values[trial_idxs])) |>
                     collect
    # @show grouped_trials
    return (grouped_trials, trials_times[1] .- trials[1][1])
end

function csi(values, offsetted_times, target_start, target_stop; is_voltage=false)
    times_to_take = findall(time_t -> target_start <= time_t <= target_stop, offsetted_times)
    # compute mean for each values
    values_to_compare = values |> Map(x -> x[times_to_take]) |> Map(x -> sum(x) / length(x)) |> collect
    values_diff = (values_to_compare[2] - values_to_compare[1]) / (values_to_compare[1] + values_to_compare[2])
end

end
