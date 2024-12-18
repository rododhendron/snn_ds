### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ e86eea66-ad59-11ef-2550-cf2588eae9d6
begin
	using DrWatson, Pkg
end

# ╔═╡ 31c85e65-cf3e-465a-86da-9a8547f7bec0
@quickactivate "snn_ds"

# ╔═╡ 2d4149ce-1a89-4087-a7e2-6cf48778ab51
using Symbolics, ModelingToolkit, DifferentialEquations, RecursiveArrayTools, SymbolicIndexingInterface, CairoMakie, ComponentArrays, AlgebraOfGraphics, Tables

# ╔═╡ 1984a541-8ba8-47ff-9cce-1ba29748e200
using CairoMakie: Axis

# ╔═╡ 16720d97-2552-4852-a011-4ea19c8b9d8b
begin 
	include("../src/utils.jl")
	include("../src/neuron.jl")
end

# ╔═╡ 178284dc-a087-4443-b34a-ed36da8bda28
Pkg.instantiate()

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

function make_fig(;xlabel="", ylabel="", title="", height=700, width=1600, yticks=Makie.automatic)
    f = Figure(size=(width, height))
    ax1 = Axis(f[1, 1],
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
		yticks=yticks,
		xticks=LinearTicks(10)
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
	xlims!(ax2, (start, stop))
    tofile ? save(name, f) : f
end

function sol_to_spikes(spikes_x_vec::Vector, y_value)::Vector
	spikes_values = ones(Int, size(spikes_x_vec))
    spikes_in_window = spikes_values * y_value
end

function plot_spikes((spikes_e, spikes_i); start=0, stop=0, xlabel="", ylabel="", title="", name="", color=(:grey, :grey), height=600)
	spikes = vcat(spikes_e, spikes_i)
    f, ax, ax1 = make_fig(; xlabel=xlabel, ylabel=ylabel, title=title, height=height, yticks=LinearTicks(size(spikes, 1)))
    xlims!(ax1, (start, stop))
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

function get_spikes_from_voltage(t, v, v_target)
    tol = 1e-3
    spike_idx = findall(x -> v_target - tol < x < v_target + tol, v)
    return (t[spike_idx], v[spike_idx])
end
end

# ╔═╡ 905483e2-78b9-40ed-8421-cd1b406003d9
begin
	tspan = (0, 1)
	ni_neurons = 100
	ne_neurons = 400
end

# ╔═╡ 88923549-1fb1-4337-aaa7-885029ca2321
params = Neuron.get_adex_neuron_params_skeleton(Float64)

# ╔═╡ c39f4a5c-86ec-4e92-a20f-965cf37fc3cb
@time i_neurons = [Neuron.make_neuron(params, Neuron.Soma, tspan, Symbol("i_neuron_$(i)")) for i in 1:ni_neurons]

# ╔═╡ 9078ad20-c2b3-43e4-a0cf-571d857bfa41
@time e_neurons = [Neuron.make_neuron(params, Neuron.Soma, tspan, Symbol("e_neuron_$(i)")) for i in 1:ne_neurons]

# ╔═╡ 32e92fca-36d3-4ebd-a228-fd6b3f965694


# ╔═╡ fa06378c-7089-419a-9c4f-65d48263155e
begin
	ee_rule = Neuron.ConnectionRule(Neuron.excitator, Neuron.excitator, Neuron.AMPA(), 0.05)
	ei_rule = Neuron.ConnectionRule(Neuron.excitator, Neuron.inhibitor, Neuron.AMPA(), 0.05)
	ie_rule = Neuron.ConnectionRule(Neuron.inhibitor, Neuron.excitator, Neuron.GABAa(), 0.08)
	ii_rule = Neuron.ConnectionRule(Neuron.inhibitor, Neuron.inhibitor, Neuron.AMPA(), 0.08)
end

# ╔═╡ ac085539-5ace-4b3f-89ad-cc76432edb17
begin
	(id_map, map_connect) = Neuron.init_connection_map(e_neurons, i_neurons, vcat(ee_rule, ei_rule, ie_rule, ii_rule))
	# map_connect[1, 3] = nothing
	# map_connect[2, 1] = nothing
	# map_connect[3, 1] = nothing
	(id_map, map_connect)
end

# ╔═╡ 621b2c5e-4b7a-4d63-8dd6-3d68b7c6694a
map_connect # pre => post

# ╔═╡ 01e97fc9-d250-4669-b642-b1357401b275
map_connect_map = Dict(Neuron.AMPA() => 1, Neuron.GABAa() => -1)

# ╔═╡ 3c548ddc-61bc-4adb-911a-a357fa370270
heatmap_connect = get.(Ref(map_connect_map), map_connect, missing)

# ╔═╡ da7bbee3-04a6-4b9a-a744-09bc46fd73ff
hs = size(heatmap_connect, 1)

# ╔═╡ 10a0e856-2412-47c6-ae54-15116cecf0bc


# ╔═╡ 1479455e-162a-4f2a-89d0-45864839d6bb
begin
	heatfig = Figure(size=(900,700))
	ticks = LinearTicks(size(heatmap_connect, 1))
	ax_heat = heatfig[1, 1] = Axis(heatfig; title="Connectivity matrix between neurons by synapse type", xlabel="Post synaptic neuron", ylabel="Pre synaptic neuron", xticks=ticks, yticks=ticks)
	elem_1 = [PolyElement(color = :red, linestyle = nothing)]
	elem_2 = [PolyElement(color = :blue, linestyle = nothing)]
	heatmap!(ax_heat, transpose(heatmap_connect); colormap=[:blue, :red], nan_color=:white)
	heatfig[1, 2] = Legend(heatfig, [elem_1, elem_2], ["AMPA", "GABAa"], framevisible = false)
	heatfig
end

# ╔═╡ c85e186e-cb66-457e-abf1-453fb4de2ec3


# ╔═╡ 95adb455-30b9-4914-8afd-e77638ecb9b4
data(DataFrame(heatmap_connect))

# ╔═╡ ebd8e444-a2d1-4f8b-908f-fc1b2c44d4b8
connections = Neuron.instantiate_connections(id_map, map_connect, vcat(e_neurons, i_neurons))

# ╔═╡ 65ac9a41-e76e-419c-a1ab-21795424ddb6
begin
	@time network = Neuron.make_network(vcat(e_neurons, i_neurons), connections)
	nothing
end

# ╔═╡ acf117c9-cbbc-4841-8610-256b8c55c23d
equations(neurons[1])

# ╔═╡ d9062ff6-5701-4e24-8ad8-876e669bd0e2


# ╔═╡ 462d463a-7c37-4b89-8c53-6a3a3180503a
begin 
	simplified_model = structural_simplify(network, split=false)
	nothing
end

# ╔═╡ dea89bed-20c5-447b-aaea-510434099fe3
begin
	tree = Utils.make_param_tree(simplified_model)
	nothing
end

# ╔═╡ 02ca7412-77ef-4597-97d4-800572f92c84
simplified_model.e_neuron_1.soma.Ib

# ╔═╡ 01baa3b5-9220-40be-b783-a194a7703b65
continuous_events(simplified_model)

# ╔═╡ 29088b47-9edc-4c5d-a8d1-44b38f9c938b


# ╔═╡ d69ada25-0300-4f86-9c2f-39f23ca4a9de
uparams = Neuron.get_adex_neuron_uparams_skeleton(Float64)

# ╔═╡ 0f0a0a07-5c02-4591-9e41-0c089dfd92cd
function override_params(params, rules)
	params_dict = Dict()
	for rule in rules
		params_dict[rule[1]] = rule[2]
	end
	ComponentArray(params; params_dict...
	)
end

# ╔═╡ f40be669-9d43-40dc-baff-d2209f35972e
begin
	params.input_value = 0e-9
	params.inc_gsyn = 5e-9
	params.tau_GABAa_fast=8e-3
	# params.vtarget_inh = -100e-3
	params.e_neuron_1__soma__input_value = 1e-9

	make_rule(prefix, range, suffix, value) = Symbol.(prefix .* "_" .* string.(range) .* "__" .* suffix) .=> value
	rules = []
	append!(rules, make_rule("e_neuron", 1:ne_neurons, "gabaa_syn__inc_gsyn", 5e-9))
	append!(rules, make_rule("e_neuron", 1:100, "soma__input_value", 1e-9))
	overriden_params = override_params(params, rules)

	iparams, iuparams = Neuron.map_params(simplified_model, overriden_params, uparams; match_nums=true)
end

# ╔═╡ 928271d8-1a75-4604-8219-df95b7170a06


# ╔═╡ 5f3e4953-eede-48e1-813c-3c8361699096
input_e = Utils.fetch_tree(["e_neuron_1", "Ib"], tree)

# ╔═╡ bc142885-33c9-40a3-a922-73699f93fa69
iparams[end]

# ╔═╡ b26ac509-3cf7-4c03-b269-06d6c8b3ab87
# push!(iparams, [input_e[1]] => 1.0e-9)

# ╔═╡ 47848510-7d59-4c34-9bb9-afbee580c07b
# a = [iparams[1:end-1]; simplified_model.neuron_2.soma.input_value => 1e-9]

# ╔═╡ 1fc4591a-ebbf-424f-9862-bb19ba7b0f38
prob = ODEProblem(simplified_model, iuparams, tspan, iparams)

# ╔═╡ 704f5c30-5c26-4dce-b152-24609263d70d
begin
	# sol = solve(prob, Vern6(); abstol=1e-9, reltol=1e-9)
	sol = solve(prob, KenCarp47(linsolve=KrylovJL_GMRES()); abstol=1e-5, reltol=1e-5, dtmax=1e-3)
end

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
plot_neuron_value(sol.t, sol[simplified_model.e_neuron_1.soma.v], params, sol[simplified_model.e_neuron_1.soma.Ib];
    start=before_offset,
    stop=after_offset,
    title="Neuron voltage along time.",
    xlabel="time (in s)",
    ylabel="voltage (in V)",
	is_voltage=true,
	tofile=false
)

# ╔═╡ 3e4feae4-7327-4537-b3e3-8958d9d6c16b
plot_neuron_value(sol.t, sol[simplified_model.e_neuron_1.soma.w], params, zeros(Int, size(sol[simplified_model.e_neuron_2.soma.w]));
    start=before_offset,
    stop=after_offset,
    title="Neuron adaptation along time.",
    xlabel="time (in s)",
    ylabel="adaptation (in A)",
	is_voltage=false,
	tofile=false
)

# ╔═╡ a81c6385-0952-4005-8534-afd24f7a1a77
plot_neuron_value(sol.t, sol[simplified_model.e_neuron_2.soma.v], params, sol[simplified_model.e_neuron_2.soma.Ii];
    start=before_offset,
    stop=after_offset,
    title="Pre synaptic neuron voltage along time.",
    xlabel="time (in s)",
    ylabel="voltage (in V)",
	is_voltage=true,
	tofile=false
)

# ╔═╡ 6da82b8c-3bf0-4f98-9da4-dab982a1a741
network.i_neuron_1.soma.a

# ╔═╡ e398609c-25e1-491f-ae6d-4f7bc4e49ce9
plot_neuron_value(sol.t, sol[simplified_model.i_neuron_1.soma.v], params, sol[simplified_model.i_neuron_1.soma.Ie];
    start=before_offset,
    stop=after_offset,
    title="Neuron voltage along time.",
    xlabel="time (in s)",
    ylabel="voltage (in V)",
	is_voltage=true,
	tofile=false
)

# ╔═╡ f71147fb-09f3-4a26-81c5-3065180155f9
plot_neuron_value(sol.t, sol[simplified_model.i_neuron_1.soma.w], params, sol[simplified_model.i_neuron_1.soma.Ie];
    start=before_offset,
    stop=after_offset,
    title="Pre synaptic neuron voltage along time.",
    xlabel="time (in s)",
    ylabel="voltage (in V)",
	is_voltage=false,
	tofile=false
)

# ╔═╡ 6fc69b0a-fba0-4361-be42-bf9dd3338c15
typeof(unknowns(network)[1])

# ╔═╡ cf9d1f02-a5eb-4133-8938-63bac7eeb72d
propertynames(simplified_model)

# ╔═╡ f56a6e15-5df9-4c77-b5d6-4157e9eb9162
typeof(getproperty(simplified_model, :e_neuron_1))

# ╔═╡ ffe4ec62-9dc4-4cb7-bd40-774db413c6ce
typeof(getproperty(simplified_model.e_neuron_1.soma, :v))

# ╔═╡ 45f54af3-86f6-4f2e-949d-7db5f6ed3665
sol[getproperty(simplified_model.e_neuron_1.soma, :v)]

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

# ╔═╡ af495aea-ae2b-4e5b-b0a0-fb87364ec4e9
mi = reduce(hcat, sol[ris])

# ╔═╡ 72c8fab9-9bff-4c93-936b-70894bce5b3f
zeros(eltype(ma), size(ma))

# ╔═╡ 340fd247-1ef3-4f21-bb4e-95372cb9bc1a
rs = Utils.get_spikes_from_r(ma) .|> Bool

# ╔═╡ 9c4a47e8-c327-4bd2-a80e-0a8d898c5a6e
sum(rs)

# ╔═╡ f815fc58-0bb4-40de-bf33-281cf155ed22
transpose(ma)[end, 1]

# ╔═╡ 5177d2c0-28e8-4de4-b6d7-5bdeab2d929a
Utils.get_spikes_from_r(transpose(ma))

# ╔═╡ a5f0bb8d-79cb-4b8b-9d47-32c76244771a
Utils.get_spikes_from_r(ma) |> sum

# ╔═╡ 194b02dc-8e70-40e8-95fb-a8ff53d2fb70
size(ma)

# ╔═╡ 453a2187-1c6d-4c5b-92bf-cc263622cf98
spikes_times = Utils.get_spike_timings(ma, sol)

# ╔═╡ 41d5ee45-958f-4b3e-ae01-3482038d34bc
spikes_inh = Utils.get_spike_timings(mi, sol)

# ╔═╡ 1915736c-3007-428c-939c-a6655d592a61
vcat(spikes_inh, spikes_times)

# ╔═╡ b43e5b71-edce-4ad9-ba35-588a5dd6cf9a
plot_spikes((spikes_times, spikes_inh); start=0, stop=1, color=(:red, :blue), height=1500)

# ╔═╡ 2341a3bb-d912-4a7e-93ba-fe856d3aaa4d


# ╔═╡ 9d2b97af-e047-41d4-a6eb-583b41139951


# ╔═╡ Cell order:
# ╠═e86eea66-ad59-11ef-2550-cf2588eae9d6
# ╠═31c85e65-cf3e-465a-86da-9a8547f7bec0
# ╠═178284dc-a087-4443-b34a-ed36da8bda28
# ╠═cd62780e-8a5a-49eb-b3d1-33c88e966332
# ╠═8eda90d1-5695-47d7-a435-79c49a17c50e
# ╠═16720d97-2552-4852-a011-4ea19c8b9d8b
# ╠═2d4149ce-1a89-4087-a7e2-6cf48778ab51
# ╠═1984a541-8ba8-47ff-9cce-1ba29748e200
# ╠═fa7d1b52-e46b-4dc3-8f08-808e36d487ee
# ╠═2b91d94b-7cf5-4dcf-bf46-2177e021c58d
# ╠═13b221f8-8ed3-4bf4-bd67-a06440fff08e
# ╠═905483e2-78b9-40ed-8421-cd1b406003d9
# ╠═88923549-1fb1-4337-aaa7-885029ca2321
# ╠═c39f4a5c-86ec-4e92-a20f-965cf37fc3cb
# ╠═9078ad20-c2b3-43e4-a0cf-571d857bfa41
# ╠═32e92fca-36d3-4ebd-a228-fd6b3f965694
# ╠═fa06378c-7089-419a-9c4f-65d48263155e
# ╠═ac085539-5ace-4b3f-89ad-cc76432edb17
# ╠═621b2c5e-4b7a-4d63-8dd6-3d68b7c6694a
# ╠═01e97fc9-d250-4669-b642-b1357401b275
# ╠═3c548ddc-61bc-4adb-911a-a357fa370270
# ╠═da7bbee3-04a6-4b9a-a744-09bc46fd73ff
# ╠═10a0e856-2412-47c6-ae54-15116cecf0bc
# ╠═1479455e-162a-4f2a-89d0-45864839d6bb
# ╠═c85e186e-cb66-457e-abf1-453fb4de2ec3
# ╠═95adb455-30b9-4914-8afd-e77638ecb9b4
# ╠═ebd8e444-a2d1-4f8b-908f-fc1b2c44d4b8
# ╠═65ac9a41-e76e-419c-a1ab-21795424ddb6
# ╠═dea89bed-20c5-447b-aaea-510434099fe3
# ╠═acf117c9-cbbc-4841-8610-256b8c55c23d
# ╠═d9062ff6-5701-4e24-8ad8-876e669bd0e2
# ╠═462d463a-7c37-4b89-8c53-6a3a3180503a
# ╠═02ca7412-77ef-4597-97d4-800572f92c84
# ╠═01baa3b5-9220-40be-b783-a194a7703b65
# ╠═29088b47-9edc-4c5d-a8d1-44b38f9c938b
# ╠═d69ada25-0300-4f86-9c2f-39f23ca4a9de
# ╠═0f0a0a07-5c02-4591-9e41-0c089dfd92cd
# ╠═f40be669-9d43-40dc-baff-d2209f35972e
# ╠═928271d8-1a75-4604-8219-df95b7170a06
# ╠═5f3e4953-eede-48e1-813c-3c8361699096
# ╠═bc142885-33c9-40a3-a922-73699f93fa69
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
# ╠═3e4feae4-7327-4537-b3e3-8958d9d6c16b
# ╠═a81c6385-0952-4005-8534-afd24f7a1a77
# ╠═6da82b8c-3bf0-4f98-9da4-dab982a1a741
# ╠═e398609c-25e1-491f-ae6d-4f7bc4e49ce9
# ╠═f71147fb-09f3-4a26-81c5-3065180155f9
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
# ╠═af495aea-ae2b-4e5b-b0a0-fb87364ec4e9
# ╠═72c8fab9-9bff-4c93-936b-70894bce5b3f
# ╠═340fd247-1ef3-4f21-bb4e-95372cb9bc1a
# ╠═9c4a47e8-c327-4bd2-a80e-0a8d898c5a6e
# ╠═f815fc58-0bb4-40de-bf33-281cf155ed22
# ╠═5177d2c0-28e8-4de4-b6d7-5bdeab2d929a
# ╠═a5f0bb8d-79cb-4b8b-9d47-32c76244771a
# ╠═194b02dc-8e70-40e8-95fb-a8ff53d2fb70
# ╠═453a2187-1c6d-4c5b-92bf-cc263622cf98
# ╠═41d5ee45-958f-4b3e-ae01-3482038d34bc
# ╠═1915736c-3007-428c-939c-a6655d592a61
# ╠═b43e5b71-edce-4ad9-ba35-588a5dd6cf9a
# ╠═2341a3bb-d912-4a7e-93ba-fe856d3aaa4d
# ╠═9d2b97af-e047-41d4-a6eb-583b41139951
