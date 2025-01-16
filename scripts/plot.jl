using CairoMakie

module Plot
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

function plot_neuron_value(time, value, p; start=0, stop=0, xlabel="", ylabel="", title="", name="")
    f, ax = make_fig(time, value; xlabel=xlabel, ylabel=ylabel, title=title)
    sliced_time, sliced_value = get_slice_xy(time, value, start=start, stop=stop)
    hlines!(ax, [p.vthr, p.vrest]; color=1:2)
    vlines!(ax, [offset]; color=:grey)
    lines!(ax, sliced_time, sliced_value)
    save(name, f)
end

function plot_spikes(spikes_e, spikes_i; start=0, stop=0, xlabel="", ylabel="", title="", name="", color=(:grey, :grey), height=600)
    spikes = vcat(spikes_e, spikes_i)
    f, ax, ax1 = make_fig(; xlabel=xlabel, ylabel=ylabel, title=title, height=height, yticks=LinearTicks(size(spikes, 1)), call_ax2=false)
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
    f
end

end
