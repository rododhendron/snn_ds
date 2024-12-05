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
begin 
	include("../src/utils.jl")
	include("../src/neuron.jl")
end

# ╔═╡ cd62780e-8a5a-49eb-b3d1-33c88e966332
html"""<style>
pluto-editor main {
    max-width: 90%;
	align-self: flex-center;
	margin-right: auto;
	margin-left: auto;
}
"""

# ╔═╡ 8eda90d1-5695-47d7-a435-79c49a17c50e


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
	ni_neurons = 2
	ne_neurons = 8
end

# ╔═╡ 88923549-1fb1-4337-aaa7-885029ca2321
# ╠═╡ disabled = true
#=╠═╡
params = Neuron.AdExNeuronParams()
  ╠═╡ =#

# ╔═╡ 32e92fca-36d3-4ebd-a228-fd6b3f965694


# ╔═╡ fa06378c-7089-419a-9c4f-65d48263155e
ee_rule = Neuron.ConnectionRule(Neuron.excitator, Neuron.excitator, Neuron.AMPA(), 1.0)

# ╔═╡ d9c156aa-f201-40f7-9be1-16f01c172afe
ei_rule = Neuron.ConnectionRule(Neuron.excitator, Neuron.inhibitor, Neuron.AMPA(), 1.0)

# ╔═╡ 70d85f5a-8565-4140-8f7a-f025040a48af
ie_rule = Neuron.ConnectionRule(Neuron.inhibitor, Neuron.excitator, Neuron.GABAa(), 1.0)

# ╔═╡ 01e97fc9-d250-4669-b642-b1357401b275
map_connect_map = Dict(Neuron.AMPA() => 1, Neuron.GABAa() => -1, nothing => 0)

# ╔═╡ acf117c9-cbbc-4841-8610-256b8c55c23d
equations(neurons[1])

# ╔═╡ d9062ff6-5701-4e24-8ad8-876e669bd0e2


# ╔═╡ bd02cf83-fb63-4a31-94fa-711c5701ac61
params = Neuron.AdExNeuronParams()

# ╔═╡ c39f4a5c-86ec-4e92-a20f-965cf37fc3cb
@time i_neurons = [Neuron.make_neuron(params, Neuron.Soma, tspan, Symbol("i_neuron_$(i)")) for i in 1:ni_neurons]

# ╔═╡ 9078ad20-c2b3-43e4-a0cf-571d857bfa41
@time e_neurons = [Neuron.make_neuron(params, Neuron.Soma, tspan, Symbol("e_neuron_$(i)")) for i in 1:ne_neurons]

# ╔═╡ ac085539-5ace-4b3f-89ad-cc76432edb17
(id_map, map_connect) = Neuron.init_connection_map(e_neurons, i_neurons, vcat(ee_rule, ei_rule, ie_rule))

# ╔═╡ 621b2c5e-4b7a-4d63-8dd6-3d68b7c6694a
map_connect

# ╔═╡ 3c548ddc-61bc-4adb-911a-a357fa370270
heatmap_connect = get.(Ref(map_connect_map), map_connect, 0)

# ╔═╡ 1479455e-162a-4f2a-89d0-45864839d6bb
heatmap(1:10, 10:-1:1, heatmap_connect)

# ╔═╡ ebd8e444-a2d1-4f8b-908f-fc1b2c44d4b8
connections = Neuron.instantiate_connections(id_map, map_connect, vcat(e_neurons, i_neurons))

# ╔═╡ 65ac9a41-e76e-419c-a1ab-21795424ddb6
@time network = Neuron.make_network(vcat(e_neurons, i_neurons), connections)

# ╔═╡ dea89bed-20c5-447b-aaea-510434099fe3
tree = Utils.make_param_tree(network)

# ╔═╡ 462d463a-7c37-4b89-8c53-6a3a3180503a
simplified_model = network

# ╔═╡ 02ca7412-77ef-4597-97d4-800572f92c84
simplified_model.e_neuron_1.soma.Ib

# ╔═╡ 01baa3b5-9220-40be-b783-a194a7703b65
continuous_events(simplified_model)

# ╔═╡ d69ada25-0300-4f86-9c2f-39f23ca4a9de
uparams = Neuron.AdExNeuronUParams()

# ╔═╡ ffe8ec7d-b1ea-4b76-9d0b-86b4cc74adb5
iparams, iuparams = Neuron.map_params(simplified_model, params, uparams)

# ╔═╡ 5f3e4953-eede-48e1-813c-3c8361699096
input_e = Utils.fetch_tree(["e_neuron_1", "Ib"], tree)

# ╔═╡ b26ac509-3cf7-4c03-b269-06d6c8b3ab87
push!(iparams, [input_e[1]] => 1.0e-9)

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
iv = getp(sol, simplified_model.e_neuron_1.soma.input_value)

# ╔═╡ bd90c50e-9611-49f5-8156-3e4e04c7bbd8
us[1]

# ╔═╡ 2b16d9c0-bfc2-4511-bbc6-4aa01c519cf5
before_offset = 0

# ╔═╡ afff9ea4-7bc3-4944-8654-f5da300ad2dc
after_offset = 1

# ╔═╡ ab42d5d1-679b-4b30-8bfd-91a4d8fafa15
plot_neuron_value(sol.t, sol[network.e_neuron_1.soma.v], params, sol[network.e_neuron_1.soma.Ie];
    start=before_offset,
    stop=after_offset,
    title="Pre synaptic neuron voltage along time.",
    xlabel="time (in s)",
    ylabel="voltage (in V)",
	is_voltage=true,
	tofile=false
)

# ╔═╡ a81c6385-0952-4005-8534-afd24f7a1a77
plot_neuron_value(sol.t, sol[network.e_neuron_2.soma.v], params, sol[network.e_neuron_2.soma.Ie];
    start=before_offset,
    stop=after_offset,
    title="Pre synaptic neuron voltage along time.",
    xlabel="time (in s)",
    ylabel="voltage (in V)",
	is_voltage=true,
	tofile=false
)

# ╔═╡ 6da82b8c-3bf0-4f98-9da4-dab982a1a741
network.i_neuron_1.ampa_syn

# ╔═╡ e398609c-25e1-491f-ae6d-4f7bc4e49ce9
plot_neuron_value(sol.t, sol[network.i_neuron_2.soma.v], params, sol[network.i_neuron_2.ampa_syn.g_syn];
    start=before_offset,
    stop=after_offset,
    title="Pre synaptic neuron voltage along time.",
    xlabel="time (in s)",
    ylabel="voltage (in V)",
	is_voltage=true,
	tofile=false
)

# ╔═╡ 6fc69b0a-fba0-4361-be42-bf9dd3338c15
typeof(unknowns(network)[1])

# ╔═╡ cf9d1f02-a5eb-4133-8938-63bac7eeb72d
propertynames(network)

# ╔═╡ f56a6e15-5df9-4c77-b5d6-4157e9eb9162
typeof(getproperty(network, :e_neuron_1))

# ╔═╡ ffe4ec62-9dc4-4cb7-bd40-774db413c6ce
typeof(getproperty(network.e_neuron_1.soma, :v))

# ╔═╡ 45f54af3-86f6-4f2e-949d-7db5f6ed3665
sol[getproperty(network.e_neuron_1.soma, :v)]

# ╔═╡ 6bb1c962-14e4-4f05-9de9-86be865b2394


# ╔═╡ f40edca9-7411-429e-926c-0366363c2d88
res = Utils.fetch_tree(["e_neuron", "R"], tree)

# ╔═╡ d0a1db60-e74a-4ea3-9ef4-50cf40dfbbf6


# ╔═╡ 0fb67ed8-5118-4a41-9d84-918a6f95144d
hcat(sol[res]...)

# ╔═╡ 206676ad-2e5a-4a64-bc8d-054d30c805d7
ris = Utils.fetch_tree(["i_neuron", "R"], tree)

# ╔═╡ 54323c9e-d5ef-409c-827e-5abaaf0c8e9e
scatter(sol[res])

# ╔═╡ 18701403-6cff-49a4-a3c2-d37f8a80c09a
parameters(network)

# ╔═╡ e39f2ce2-8f02-4a01-8756-b23184c44378
ma = reduce(hcat, sol[res])

# ╔═╡ 72c8fab9-9bff-4c93-936b-70894bce5b3f
zeros(eltype(ma), size(ma))

# ╔═╡ d8cc69d5-bf70-479b-b6b6-7de1f02f1930
function get_spikes_from_r(r_array)
    # if dims == time, features
	res_array = zeros(eltype(r_array), size(r_array))
	for i in 1:size(res_array, 1)
		for j in 2:size(res_array, 2)
			res_array[i, j] = r_array[i, j - 1] < r_array[i, j] ? 1.0 : 0.0
		end
	end
	res_array
end

# ╔═╡ 340fd247-1ef3-4f21-bb4e-95372cb9bc1a
rs = get_spikes_from_r(ma)

# ╔═╡ 9c4a47e8-c327-4bd2-a80e-0a8d898c5a6e


# ╔═╡ f815fc58-0bb4-40de-bf33-281cf155ed22
transpose(ma)[end, 1]

# ╔═╡ 5177d2c0-28e8-4de4-b6d7-5bdeab2d929a
get_spikes_from_r(transpose(ma))

# ╔═╡ a5f0bb8d-79cb-4b8b-9d47-32c76244771a
get_spikes_from_r(ma) |> sum

# ╔═╡ 453a2187-1c6d-4c5b-92bf-cc263622cf98
sol[rs][t]

# ╔═╡ Cell order:
# ╠═e86eea66-ad59-11ef-2550-cf2588eae9d6
# ╠═31c85e65-cf3e-465a-86da-9a8547f7bec0
# ╠═cd62780e-8a5a-49eb-b3d1-33c88e966332
# ╠═8eda90d1-5695-47d7-a435-79c49a17c50e
# ╠═16720d97-2552-4852-a011-4ea19c8b9d8b
# ╠═2d4149ce-1a89-4087-a7e2-6cf48778ab51
# ╠═fa7d1b52-e46b-4dc3-8f08-808e36d487ee
# ╠═2b91d94b-7cf5-4dcf-bf46-2177e021c58d
# ╠═13b221f8-8ed3-4bf4-bd67-a06440fff08e
# ╠═905483e2-78b9-40ed-8421-cd1b406003d9
# ╠═88923549-1fb1-4337-aaa7-885029ca2321
# ╠═c39f4a5c-86ec-4e92-a20f-965cf37fc3cb
# ╠═9078ad20-c2b3-43e4-a0cf-571d857bfa41
# ╠═32e92fca-36d3-4ebd-a228-fd6b3f965694
# ╠═fa06378c-7089-419a-9c4f-65d48263155e
# ╠═d9c156aa-f201-40f7-9be1-16f01c172afe
# ╠═70d85f5a-8565-4140-8f7a-f025040a48af
# ╠═ac085539-5ace-4b3f-89ad-cc76432edb17
# ╠═621b2c5e-4b7a-4d63-8dd6-3d68b7c6694a
# ╠═01e97fc9-d250-4669-b642-b1357401b275
# ╠═3c548ddc-61bc-4adb-911a-a357fa370270
# ╠═1479455e-162a-4f2a-89d0-45864839d6bb
# ╠═ebd8e444-a2d1-4f8b-908f-fc1b2c44d4b8
# ╠═65ac9a41-e76e-419c-a1ab-21795424ddb6
# ╠═dea89bed-20c5-447b-aaea-510434099fe3
# ╠═acf117c9-cbbc-4841-8610-256b8c55c23d
# ╠═d9062ff6-5701-4e24-8ad8-876e669bd0e2
# ╠═462d463a-7c37-4b89-8c53-6a3a3180503a
# ╠═02ca7412-77ef-4597-97d4-800572f92c84
# ╠═01baa3b5-9220-40be-b783-a194a7703b65
# ╠═bd02cf83-fb63-4a31-94fa-711c5701ac61
# ╠═d69ada25-0300-4f86-9c2f-39f23ca4a9de
# ╠═ffe8ec7d-b1ea-4b76-9d0b-86b4cc74adb5
# ╠═5f3e4953-eede-48e1-813c-3c8361699096
# ╠═b26ac509-3cf7-4c03-b269-06d6c8b3ab87
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
# ╠═a81c6385-0952-4005-8534-afd24f7a1a77
# ╠═6da82b8c-3bf0-4f98-9da4-dab982a1a741
# ╠═e398609c-25e1-491f-ae6d-4f7bc4e49ce9
# ╠═6fc69b0a-fba0-4361-be42-bf9dd3338c15
# ╠═cf9d1f02-a5eb-4133-8938-63bac7eeb72d
# ╠═f56a6e15-5df9-4c77-b5d6-4157e9eb9162
# ╠═ffe4ec62-9dc4-4cb7-bd40-774db413c6ce
# ╠═45f54af3-86f6-4f2e-949d-7db5f6ed3665
# ╠═6bb1c962-14e4-4f05-9de9-86be865b2394
# ╠═f40edca9-7411-429e-926c-0366363c2d88
# ╠═d0a1db60-e74a-4ea3-9ef4-50cf40dfbbf6
# ╠═0fb67ed8-5118-4a41-9d84-918a6f95144d
# ╠═206676ad-2e5a-4a64-bc8d-054d30c805d7
# ╠═54323c9e-d5ef-409c-827e-5abaaf0c8e9e
# ╠═18701403-6cff-49a4-a3c2-d37f8a80c09a
# ╠═e39f2ce2-8f02-4a01-8756-b23184c44378
# ╠═72c8fab9-9bff-4c93-936b-70894bce5b3f
# ╠═d8cc69d5-bf70-479b-b6b6-7de1f02f1930
# ╠═340fd247-1ef3-4f21-bb4e-95372cb9bc1a
# ╠═9c4a47e8-c327-4bd2-a80e-0a8d898c5a6e
# ╠═f815fc58-0bb4-40de-bf33-281cf155ed22
# ╠═5177d2c0-28e8-4de4-b6d7-5bdeab2d929a
# ╠═a5f0bb8d-79cb-4b8b-9d47-32c76244771a
# ╠═453a2187-1c6d-4c5b-92bf-cc263622cf98
