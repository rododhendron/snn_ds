### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ e86eea66-ad59-11ef-2550-cf2588eae9d6
begin
	using DrWatson
end

# ╔═╡ 31c85e65-cf3e-465a-86da-9a8547f7bec0
@quickactivate "snn_ds"

# ╔═╡ 2d4149ce-1a89-4087-a7e2-6cf48778ab51
using Symbolics, ModelingToolkit, DifferentialEquations, RecursiveArrayTools, SymbolicIndexingInterface, CairoMakie

# ╔═╡ 16720d97-2552-4852-a011-4ea19c8b9d8b
include("../src/neuron.jl")

# ╔═╡ cd62780e-8a5a-49eb-b3d1-33c88e966332
html"""<style>
pluto-editor main {
    max-width: 90%;
	align-self: flex-center;
	margin-right: auto;
	margin-left: auto;
}
"""

# ╔═╡ fa7d1b52-e46b-4dc3-8f08-808e36d487ee
offset = 200e-3

# ╔═╡ 2b91d94b-7cf5-4dcf-bf46-2177e021c58d


# ╔═╡ 13b221f8-8ed3-4bf4-bd67-a06440fff08e
begin
function get_slice_xy(x, y; start=0, stop=0)
    stop = stop == 0 ? maximum(a) : stop
    first_x_idx = findfirst(el -> el >= start, x)
    last_x_idx = findlast(el -> el <= stop, x)
    (x[first_x_idx:last_x_idx], y[first_x_idx:last_x_idx])
end

function make_fig(;xlabel="", ylabel="", title="")
    f = Figure(size=(1600, 700))
    ax1 = Axis(f[1, 1],
        title=title,
        xlabel=xlabel,
        ylabel=ylabel
    )
	ax2 = Axis(f[1, 1],
		yaxisposition = :right,
		ylabel="input_current, in A"
	)
	linkxaxes!(ax1, ax2)
	# hidespines!(ax2)
	# hidedecorations!(ax2)
    (f, ax1, ax2)
end

function plot_neuron_value(time, value, p, input_current; start=0, stop=0, xlabel="", ylabel="", title="", name="", tofile=true, is_voltage=false)
    f, ax, ax2 = make_fig(;xlabel=xlabel, ylabel=ylabel, title=title)
    sliced_time, sliced_value = get_slice_xy(time, value, start=start, stop=stop)
    is_voltage ? hlines!(ax, [p.vthr, p.vrest]; color=1:2) : nothing
    vlines!(ax, [offset]; color=:grey, linestyle=:dashdot)
	if !isnothing(input_current)
    	sliced_time, sliced_current = get_slice_xy(time, input_current, start=start,stop=stop)
	lines!(ax2, sliced_time, sliced_current, color=(:black, 0.3))
	end
    lines!(ax, sliced_time, sliced_value)
	xlims!(ax, (start, stop))
    tofile ? save(name, f) : f
end

function plot_spikes(spikes; start=0, stop=0, xlabel="", ylabel="", title="", name="")
	f, ax, ax2 = make_fig(;xlabel=xlabel, ylabel=ylabel, title=title)
	spikes_in_window = [spike for spike in spikes if spike != 0 && start < spike < stop]
    vlines!(ax, spikes_in_window; color=:black, linewidth=1)
	vlines!(ax, [start, stop]; color=:white)
	xlims!(ax, (start, stop))
	f
end

function get_spikes_from_voltage(t, v, v_target)
    tol = 1e-3
    spike_idx = findall(x -> v_target - tol < x < v_target + tol, v)
    return (t[spike_idx], v[spike_idx])
end
end

# ╔═╡ 905483e2-78b9-40ed-8421-cd1b406003d9
begin
	tspan = (0, 1)
	n_neurons = 3
	i_neurons = 2
	e_neurons = 8
end

# ╔═╡ 88923549-1fb1-4337-aaa7-885029ca2321
params = Neuron.AdExNeuronParams(;input_value=1e-9)

# ╔═╡ c39f4a5c-86ec-4e92-a20f-965cf37fc3cb
neurons = [neuron = Neuron.make_neuron(params, Neuron.Soma, tspan,Symbol("neuron_$(i)")) for i in 1:n_neurons]

# ╔═╡ acf117c9-cbbc-4841-8610-256b8c55c23d
equations(neurons[1])

# ╔═╡ d9062ff6-5701-4e24-8ad8-876e669bd0e2
network = Neuron.make_network(neurons)

# ╔═╡ 462d463a-7c37-4b89-8c53-6a3a3180503a
simplified_model = Neuron.structural_simplify(network)

# ╔═╡ 02ca7412-77ef-4597-97d4-800572f92c84
simplified_model.neuron_1.soma.Ib

# ╔═╡ 01baa3b5-9220-40be-b783-a194a7703b65
continuous_events(simplified_model)

# ╔═╡ d69ada25-0300-4f86-9c2f-39f23ca4a9de
uparams = Neuron.AdExNeuronUParams()

# ╔═╡ ffe8ec7d-b1ea-4b76-9d0b-86b4cc74adb5
iparams, iuparams = Neuron.map_params(simplified_model, params, uparams)

# ╔═╡ 47848510-7d59-4c34-9bb9-afbee580c07b
# a = [iparams[1:end-1]; simplified_model.neuron_2.soma.input_value => 1e-9]

# ╔═╡ 1fc4591a-ebbf-424f-9862-bb19ba7b0f38
prob = ODEProblem(simplified_model, iuparams, tspan, iparams)

# ╔═╡ 704f5c30-5c26-4dce-b152-24609263d70d
sol = solve(prob, Vern6(); abstol=1e-7, reltol=1e-7)

# ╔═╡ e69ccf44-8098-4ecc-9c4f-3f93c0000beb
sol[end-10:end]

# ╔═╡ 2da66cdc-284b-45b6-8586-5df7fbb46895
us = unknowns(simplified_model)

# ╔═╡ cd31e8df-a26a-44d1-b50b-489772a55863
ps = parameters(simplified_model)

# ╔═╡ 557e3a85-00ce-4d0e-9ac6-eabf3d29f287
iv = getp(sol, simplified_model.neuron_1.soma.input_value)

# ╔═╡ bd90c50e-9611-49f5-8156-3e4e04c7bbd8
us[1]

# ╔═╡ 2b16d9c0-bfc2-4511-bbc6-4aa01c519cf5
before_offset = 0

# ╔═╡ afff9ea4-7bc3-4944-8654-f5da300ad2dc
after_offset = 1

# ╔═╡ ab42d5d1-679b-4b30-8bfd-91a4d8fafa15
plot_neuron_value(sol.t, sol[network.neuron_1.soma.v], params, sol[network.neuron_1.soma.Ie];
    start=before_offset,
    stop=after_offset,
    title="Pre synaptic neuron voltage along time.",
    xlabel="time (in s)",
    ylabel="voltage (in V)",
	is_voltage=true,
	tofile=false
)

# ╔═╡ e398609c-25e1-491f-ae6d-4f7bc4e49ce9
plot_neuron_value(sol.t, sol[network.neuron_2.soma.v], params, sol[network.neuron_1.soma.Ie];
    start=before_offset,
    stop=after_offset,
    title="Pre synaptic neuron voltage along time.",
    xlabel="time (in s)",
    ylabel="voltage (in V)",
	is_voltage=true,
	tofile=false
)

# ╔═╡ 6fc69b0a-fba0-4361-be42-bf9dd3338c15
plot_neuron_value(sol.t, sol[network.neuron_3.soma.v], params, sol[network.neuron_1.soma.Ie];
    start=before_offset,
    stop=after_offset,
    title="Pre synaptic neuron voltage along time.",
    xlabel="time (in s)",
    ylabel="voltage (in V)",
	is_voltage=true,
	tofile=false
)

# ╔═╡ Cell order:
# ╠═e86eea66-ad59-11ef-2550-cf2588eae9d6
# ╠═31c85e65-cf3e-465a-86da-9a8547f7bec0
# ╠═cd62780e-8a5a-49eb-b3d1-33c88e966332
# ╠═16720d97-2552-4852-a011-4ea19c8b9d8b
# ╠═2d4149ce-1a89-4087-a7e2-6cf48778ab51
# ╠═fa7d1b52-e46b-4dc3-8f08-808e36d487ee
# ╠═2b91d94b-7cf5-4dcf-bf46-2177e021c58d
# ╠═13b221f8-8ed3-4bf4-bd67-a06440fff08e
# ╠═905483e2-78b9-40ed-8421-cd1b406003d9
# ╠═88923549-1fb1-4337-aaa7-885029ca2321
# ╠═c39f4a5c-86ec-4e92-a20f-965cf37fc3cb
# ╠═acf117c9-cbbc-4841-8610-256b8c55c23d
# ╠═d9062ff6-5701-4e24-8ad8-876e669bd0e2
# ╠═462d463a-7c37-4b89-8c53-6a3a3180503a
# ╠═02ca7412-77ef-4597-97d4-800572f92c84
# ╠═01baa3b5-9220-40be-b783-a194a7703b65
# ╠═d69ada25-0300-4f86-9c2f-39f23ca4a9de
# ╠═ffe8ec7d-b1ea-4b76-9d0b-86b4cc74adb5
# ╠═47848510-7d59-4c34-9bb9-afbee580c07b
# ╠═1fc4591a-ebbf-424f-9862-bb19ba7b0f38
# ╠═704f5c30-5c26-4dce-b152-24609263d70d
# ╠═e69ccf44-8098-4ecc-9c4f-3f93c0000beb
# ╠═2da66cdc-284b-45b6-8586-5df7fbb46895
# ╠═cd31e8df-a26a-44d1-b50b-489772a55863
# ╠═557e3a85-00ce-4d0e-9ac6-eabf3d29f287
# ╠═bd90c50e-9611-49f5-8156-3e4e04c7bbd8
# ╠═2b16d9c0-bfc2-4511-bbc6-4aa01c519cf5
# ╠═afff9ea4-7bc3-4944-8654-f5da300ad2dc
# ╠═ab42d5d1-679b-4b30-8bfd-91a4d8fafa15
# ╠═e398609c-25e1-491f-ae6d-4f7bc4e49ce9
# ╠═6fc69b0a-fba0-4361-be42-bf9dd3338c15
