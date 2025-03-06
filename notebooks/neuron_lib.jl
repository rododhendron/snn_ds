### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ e86eea66-ad59-11ef-2550-cf2588eae9d6
begin
	using DrWatson, Pkg, PlutoUI
end

# ╔═╡ 31c85e65-cf3e-465a-86da-9a8547f7bec0
@quickactivate "SNN"

# ╔═╡ 31fff5a6-7b9f-4bb6-9c2f-8f61c7878fc4
using PlutoDevMacros

# ╔═╡ 2d4149ce-1a89-4087-a7e2-6cf48778ab51
using Symbolics, ModelingToolkit, DifferentialEquations, RecursiveArrayTools, SymbolicIndexingInterface, CairoMakie, ComponentArrays, AlgebraOfGraphics, Tables, LinearAlgebra, DataInterpolations, Transducers

# ╔═╡ 1984a541-8ba8-47ff-9cce-1ba29748e200
using CairoMakie: Axis

# ╔═╡ cd62780e-8a5a-49eb-b3d1-33c88e966332
html"""<style>
pluto-editor main {
    max-width: 90%;
	align-self: flex-center;
	margin-right: auto;
	margin-left: auto;
}
"""

# ╔═╡ 16720d97-2552-4852-a011-4ea19c8b9d8b
begin 
	@fromparent import SNN
end

# ╔═╡ 33b33822-5476-4759-a100-1b274456aecc
begin
	stim_params = SNN.Params.get_stim_params_skeleton()
	[println(labels(stim_params)[i], " = ", stim_params[i]) for i in 1:length(stim_params)]
end

# ╔═╡ 3d936faa-73aa-47fc-be6a-92d0d42d8389
begin
	params = SNN.Neuron.get_adex_neuron_params_skeleton(Float64)#, sch_t, sch_onset, sch_group)
	[println(labels(params)[i], " = ", params[i]) for i in 1:length(params)]
end

# ╔═╡ 905483e2-78b9-40ed-8421-cd1b406003d9
begin
	tspan = (0, 2)
	
	# make schedule
	# stim_params.n_trials = 20
	stim_params.amplitude = 3.0e-9
	stim_params.duration = 50.0e-3
	stim_params.deviant_idx = 2
	stim_params.standard_idx = 1
	stim_params.select_size = 0
	stim_params.p_deviant = 0.10
	stim_params.start_offset = 0.5
	stim_params.isi = 300e-3
	
	stim_schedule = SNN.Params.generate_schedule(stim_params, tspan)
	
	sch_t = deepcopy(stim_schedule[1, :])
	sch_onset = deepcopy(stim_schedule[2, :])
	sch_group = deepcopy(stim_schedule[3, :])
	
	# @time params = Neuron.AdExNeuronParams()
	params.inc_gsyn = 20.0e-9
	params.a = 0.0e-9          # Subthreshold adaptation (A)
	params.b = 4.0e-10          # Spiking adaptation (A)
	params.TauW = 600.0e-3      # Adaptation time constant (s)
	params.Cm = 3.0e-10
	
	params.Ibase = 3e-10
	# params.Ibase = 0
	params.sigma = 0.05

	rules = []
	# push!(rules, SNN.Params.make_rule("e_neuron", 3, "soma__Ibase", 2e-10))
	merged_params = SNN.Params.override_params(params, rules)
	@show params.b * params.TauW
end

# ╔═╡ dba22b66-ba23-4b2d-83bb-d6f32e9a3e59
begin
	con_mapping = [
		# (SNN.Params.@connect_neurons [1] SNN.Neuron.AMPA() 2)...;
		# (SNN.Params.@connect_neurons [5] SNN.Neuron.AMPA() 4)...;
		# (SNN.Params.@connect_neurons [1] SNN.Neuron.AMPA() 2)...;
		# (SNN.Params.@connect_neurons [2] SNN.Neuron.AMPA() 1)...
		# (SNN.Params.@connect_neurons [1, 2, 3, 5, 6, 7] SNN.Neuron.AMPA() 4)...
		(SNN.Params.@connect_neurons [1, 2] SNN.Neuron.AMPA() 3)...
		# (SNN.Params.@connect_neurons [4, 2] SNN.Neuron.AMPA() 3)...
		# (1, 2, SNN.Neuron.GABAa()),
		# (2, 1, SNN.Neuron.GABAa()),
	    # (2, 3, SNN.Neuron.AMPA())
	]
	@show con_mapping
	#
	pre_neurons = [row[1] for row in con_mapping]
	post_neurons = [row[2] for row in con_mapping]
	if !isempty(con_mapping)
		e_neurons_n = size(unique([pre_neurons; post_neurons]), 1)
		post_neurons = []
	else
		e_neurons_n = 1
	end
end

# ╔═╡ c39f4a5c-86ec-4e92-a20f-965cf37fc3cb
begin
	name = "network_base_1_neuron_adaptation"
	out_path_prefix = "results/"
end

# ╔═╡ 1b5b20d7-3934-406a-9d9e-2af0ad2c13db
arr::Matrix{Any} = fill(0.0, 10, 10)

# ╔═╡ 1f069d6f-4ead-49c5-84da-957c15d69074


# ╔═╡ 32e92fca-36d3-4ebd-a228-fd6b3f965694
begin
	(sol, simplified_model, prob) = SNN.Pipeline.run_exp(
	    out_path_prefix, name;
	    e_neurons_n=e_neurons_n,
	    params=merged_params,
	    stim_params=stim_params,
	    tspan=tspan,
	    con_mapping=con_mapping,
	    stim_schedule=stim_schedule,
	    solver=DRI1(),
	    # solver=DRI1(),
		tols=(1e-3, 1e-3)
	)
	nothing
end

# ╔═╡ e90677e4-3a63-4d1c-bc8f-0bb13f1a490c
simplified_model.e_neuron_5₊soma₊group

# ╔═╡ 786f52c6-f985-4b6f-b5db-04e24f5d48ce
begin
	(start, stop) = tspan
	start = stim_params.start_offset - 0.1
end

# ╔═╡ b8a8d05d-8e3c-4ec1-a71c-7930a907daf2
begin
	path = out_path_prefix * name * "/"
	exp_name = path * name
	name_prefix = exp_name * ""
	name_interpol(name) = name_prefix * "_" * name
end

# ╔═╡ 2341a3bb-d912-4a7e-93ba-fe856d3aaa4d
begin
	@time tree::SNN.Utils.ParamTree = SNN.Utils.make_param_tree(simplified_model)
	nothing
end

# ╔═╡ d3927c88-0b17-4fa5-98ce-9fc2378b5bb0
res = SNN.Utils.fetch_tree(["e_neuron", "R"], tree)

# ╔═╡ 34f656f8-c572-4cf5-9927-312fc2caae20
ma = SNN.Utils.hcat_sol_matrix(res, sol)

# ╔═╡ b594ab34-5cc8-4850-bb63-3748fbb418a0
spikes_times = SNN.Utils.get_spike_timings(ma, sol)

# ╔═╡ 28eda74b-3c55-4694-be86-906e51be760b
SNN.Plots.plot_spikes((spikes_times, []); start=start, stop=stop, color=(:red, :blue), height=400, title="Network activity", xlabel="time (in s)", ylabel="neuron index", name=name_interpol("rs.png"), schedule=stim_schedule, tofile=false)

# ╔═╡ eb0a72e1-a1c3-463c-a6fc-ade4f0d53eab
@bind i NumberField(1:e_neurons_n, default=1)

# ╔═╡ dabc4957-f625-40b9-b9b2-3d12bf399d0c
@time SNN.Plots.plot_excitator_value(i, sol, start, stop, name_interpol, tree, stim_params.start_offset, stim_schedule, false)

# ╔═╡ 337b1d88-5246-40d5-bc86-ee0b5dbc2218
@time SNN.Plots.plot_adaptation_value(i, sol, start, stop, name_interpol, tree, stim_params.start_offset, stim_schedule, false)

# ╔═╡ a90c94ad-924c-49d4-9526-78d5c25c9fc7
size(sol.t)

# ╔═╡ 85809d17-2d85-4de5-b450-6526d0d3cec9
@time SNN.Plots.plot_spike_rate(i, sol, start, stop, name_interpol, tree, stim_params.start_offset, stim_schedule, false; time_window=0.1)

# ╔═╡ 3ae6f318-21c3-44cf-bb92-2836883229b4
@time SNN.Plots.plot_spike_rate(i, sol, start, stop, name_interpol, tree, stim_params.start_offset, stim_schedule, false; time_window=0.05)

# ╔═╡ 6a0f1c02-02c3-4f40-9537-5a82bb773b7b
@time SNN.Plots.plot_spike_rate(i, sol, start, stop, name_interpol, tree, stim_params.start_offset, stim_schedule, false; time_window=0.1)

# ╔═╡ fc6227bc-9036-41d9-8fe3-4f947a227fe0
@time SNN.Plots.plot_aggregated_rate(i, sol, name_interpol, tree, stim_schedule, false)

# ╔═╡ 63ebbfb4-ae1e-4d80-a08f-ee35a1fdbcfb
@time SNN.Plots.plot_isi(i, sol, start, stop, name_interpol, tree, stim_params.start_offset, stim_schedule, false)

# ╔═╡ 1a97fd70-8dc2-45ab-b480-70e7e3651205
stim_schedule

# ╔═╡ d0fdb66a-c87b-40d5-8caa-04484b0cb5c6


# ╔═╡ af3cd829-d768-417a-aaea-cf0dde63ba54
readout = SNN.Utils.fetch_tree(["e_neuron_3", "R"], tree)

# ╔═╡ baa3784f-70a5-46f7-a8db-d5e102026411
begin
	mr = SNN.Utils.hcat_sol_matrix(readout, sol)
    spikes_readout = SNN.Utils.get_spike_timings(mr, sol) |> first # take first as I have one readout
    trials = SNN.Plots.get_trials_from_schedule(stim_schedule)
	@show trials
    trials_response = [count(x -> trial_t[1] < x < trial_t[2], spikes_readout) for trial_t in trials]
    groups = unique(stim_schedule[3, :]) .|> Int
    groups_stim_idxs = [findall(row -> row == group, stim_schedule[3, :]) for group in groups]
	@show groups
	@show groups_stim_idxs
    groups_spikes = [sum(trials_response[gsi]) / length(trials_response[gsi]) for gsi in groups_stim_idxs]
    @show groups_spikes
	first(diff(groups_spikes)) / sum(groups_spikes)
end

# ╔═╡ bc881bde-e581-4459-8ee3-ce0cc7fefb6d
sol[0.0, readout]

# ╔═╡ 3a999581-169f-4bc9-a74a-8317fb2b7523
sol.t

# ╔═╡ 5f8072f2-aa4c-4150-97f7-6db1fe89bfdf
readout_v = SNN.Utils.fetch_tree(["e_neuron_3", "v"], tree, false) |> first

# ╔═╡ 6f1e73ba-cde0-4915-828b-299017f274cd
SNN.Plots.fetch_tree_neuron_value("e_neuron", 3, "v", tree)

# ╔═╡ 35d3c8cb-f3f7-4dac-91f1-538f6b9fb29e
sol[readout_v]

# ╔═╡ 1935208e-36bc-42e2-9b75-e8050ed86b42
sol.t

# ╔═╡ c766d492-d335-45bd-8468-5701262235c0
voltage_interpolation = AkimaInterpolation(sol[readout_v], sol.t)

# ╔═╡ 7adf645a-9a09-4193-bf19-0c847350ae56
voltage_interpolation(0.1)

# ╔═╡ 77617fc7-fb5e-4d52-a0dd-c2c89c7c9110
size(stim_schedule, 2)

# ╔═╡ fa2d60e5-89af-41ba-af1e-516a71c43f5a
# begin
# 	neuron_u = readout_v
# 	sampling_rate = 2000.0
# 	method = :value
# 	mru = SNN.Utils.hcat_sol_matrix(neuron_u, sol)
#     # trials = get_trials_from_schedule(stim_schedule)
#     # groups = unique(stim_schedule[3, :]) .|> Int
#     # groups_stim_idxs = [findall(row -> row == group, stim_schedule[3, :]) for group in groups]
#     if method == :spikes
#         spikes_neuron = Utils.get_spike_timings(mr)
#         # trials_responses = [count(x -> trial_t[1] < x < trial_t[2], spikes_readout) for trial_t in trials]
#         # groups_spikes = [sum(trials_response[gsi]) / length(trials_response[gsi]) for gsi in groups_stim_idxs]
#         value = first(diff(groups_spikes)) / sum(groups_spikes)
#     elseif method == :value
#         interpolate_u(u) = AkimaInterpolation(u, sol.t)
#         interpolation_table = interpolate_u(sol[neuron_u])
# 		start_offset = -0.1
#         # for each trial, get list of resampled times
#         trials_times = [collect(trial_t[1]-0.2:(trial_t[2]-trial_t[1])/sampling_rate:trial_t[2]) for trial_t in trials]
# 		# @show trials_times
#         # get corresponding values
#         sampled_values = trials_times |> Map(trial_times -> interpolation_table.(trial_times)) |> collect
#         # trials_responses = [count(x -> trial_t[1] < x < trial_t[2], sampled_values) for trial_t in trials]
# 		@show size(trials_times)
# 		@assert size(trials_times, 1) == size(sampled_values, 1) == size(stim_schedule, 2)
# 		# for each group, for each trial in group, I compute the mean of all values at time t
# 		grouped_trials = groups_stim_idxs |> Map(trial_idxs -> sum(sampled_values[trial_idxs]) ./ length(sampled_values[trial_idxs])) |> collect
#     end
# end

# ╔═╡ cc2d1c5a-5f57-4ce9-9e31-236b6f81a9fd
trials_times

# ╔═╡ 7e7f24b1-5ed7-48ea-ab91-061a6383cc29
# offsetted_times = trials_times[1] .- stim_params.start_offset

# ╔═╡ e46d4fb1-63ee-48a8-adde-f284ffe71cf8
sum(sampled_values[1:3])

# ╔═╡ a5a05532-9ebe-454e-983c-5a74436e3e3f
(grouped_trials, offsetted_times) = SNN.Plots.compute_grand_average(sol, readout_v, stim_schedule, :value; interpol_fn=AkimaInterpolation, sampling_rate=20000)

# ╔═╡ 0bc9dd59-e29f-4854-ba6b-697adde0105d
grouped_trials[1]

# ╔═╡ 9eebabbb-e0f6-4444-a63a-d750be61523d
grouped_trials

# ╔═╡ 23a9143a-d394-41ed-89de-505c86a64de0
offsetted_times

# ╔═╡ 0ef1fd05-7ed6-422b-8a87-c4c8011f5435
sum(rand(239, 10))

# ╔═╡ 71612ba8-c43c-490c-8581-def0085eedf2
typeof(64)

# ╔═╡ 5174d0e7-9d37-42a7-a13f-30d8f71a52ef
@time SNN.Plots.plot_neuron_value(offsetted_times, grouped_trials, nothing, nothing, [0.0, 0.05]; start=-0.1, stop=maximum(offsetted_times), title="gdavg e 3", name=name_interpol("gdavg_e_3.png"), schedule=stim_schedule, tofile=false, ylabel="voltage (in V)", xlabel="Time (in s)", multi=true, plot_stims=false)

# ╔═╡ 672a833a-5c91-4397-8ce2-fbcf343beb75
sol[first(readout)]

# ╔═╡ 96951fbb-ba5a-4606-ac04-e3c9bd70c0d2
(agg_rate, ot) = SNN.Plots.compute_grand_average(sol, first(readout), stim_schedule, :spikes; interpol_fn=AkimaInterpolation, time_window=0.01, sampling_rate=20000)

# ╔═╡ a0923e08-6ad3-45c9-9d0e-3a5d78338749
function compute_grand_average(sol, neuron_u, stim_schedule, method=:spikes; sampling_rate=200.0, start=0.0, stop=last(sol.t), offset=0.1, interpol_fn=AkimaInterpolation, time_window=0.1)
	interpolate_u(u) = interpol_fn(u, sol.t)
    trials = SNN.Plots.get_trials_from_schedule(stim_schedule)
    groups = unique(stim_schedule[3, :]) .|> Int
    groups_stim_idxs = [findall(row -> row == group, stim_schedule[3, :]) for group in groups]
    if method == :spikes
        s_bool = sol[neuron_u] |> SNN.Utils.get_spikes_from_r .|> Bool
        spikes_neuron = SNN.Utils.get_spike_timings(s_bool, sol)
		@show size(spikes_neuron)
		@show spikes_neuron
        spike_rate = SNN.Plots.compute_moving_average(sol.t, spikes_neuron, time_window)
		@show spike_rate[1:20]
        @show size(s_bool)
        @show size(spikes_neuron)
        @show size(spike_rate)
        interpolation_table = interpolate_u(spike_rate)
    elseif method == :value
        interpolation_table = interpolate_u(sol[neuron_u])
        # for each trial, get list of resampled times
    end
    @show trials[1]
    trial_time_delta = first(trials)[2] - first(trials)[1] + offset
    trials_times = [collect(trial_t[1]-offset:1/sampling_rate:trial_t[2]) for trial_t in trials] |> Map(x -> x[1:floor(Int, sampling_rate * trial_time_delta)]) |> collect
    # get corresponding values
    sampled_values = trials_times |> Map(trial_times -> interpolation_table.(trial_times)) |> tcollect
    # @show sampled_values
    @assert size(trials_times, 1) == size(sampled_values, 1) == size(stim_schedule, 2)
    @show size(sampled_values)
    @show size(trials_times)
    @show size(sampled_values[groups_stim_idxs[1]])
    @show size(sampled_values[groups_stim_idxs[2]])
    grouped_trials = groups_stim_idxs |>
        Map(trial_idxs -> sum(sampled_values[trial_idxs]) ./ length(sampled_values[trial_idxs])) |>
        collect
    # @show grouped_trials
    return(grouped_trials, trials_times[1] .- trials[1][1])
end

# ╔═╡ 46073470-7f50-4cd4-98a7-3eb5043e6bab
(agg_rate_1, ot_1) = compute_grand_average(sol, first(readout), stim_schedule, :spikes; interpol_fn=AkimaInterpolation, time_window=0.1, sampling_rate=2000)

# ╔═╡ 81eaed1e-b1f8-4129-92e1-d6b33bc4106e
size(agg_rate[1])

# ╔═╡ 4e1edae3-c304-43f1-8aa6-8fee655436e7
first(readout)

# ╔═╡ 6107f3f1-1e0c-4ec4-b5ea-49216e5472ff
agg_rate

# ╔═╡ 696d3954-6373-4d3e-8ff5-764814cd4bec
@time SNN.Plots.plot_neuron_value(ot, agg_rate, nothing, nothing, [0.0, 0.05]; start=-0.1, stop=maximum(offsetted_times), title="Spike rate average per trials, deviant in red, standard in blue.", name=name_interpol("gdavg_e_3.png"), schedule=stim_schedule, plot_stims=false, tofile=false, multi=true, ylabel="spike rate (in Hz)", xlabel="time (in s)")

# ╔═╡ c0021651-ed41-4b04-b9f2-5b282aa2553f
function csi(values, offsetted_times, target_start, target_stop; is_voltage=false)
	times_to_take = findall(time_t -> target_start <= time_t <= target_stop, offsetted_times)
	# compute mean for each values
	values_to_compare = values |> Map(x -> x[times_to_take]) |> Map(x -> sum(x) / length(x)) |> collect
	values_diff = (values_to_compare[2] - values_to_compare[1]) / (values_to_compare[1] + values_to_compare[2])
end

# ╔═╡ a0b9ea82-a228-4e8a-bba3-cd255708380f
SNN.Plots.csi(agg_rate, ot, 0.0, 0.1)

# ╔═╡ 3996e2c9-905f-424b-8acb-f9ba49346a17
csi(agg_rate, ot, 0.0, 0.25)

# ╔═╡ a9aecee8-45bc-4f6e-b68c-5884c6d8a58c
CSI_v = csi(grouped_trials, ot, 0.0, 0.1, is_voltage=true)

# ╔═╡ b5cb6703-e19a-4e22-b9a8-dd8d037448eb


# ╔═╡ ed473424-9f60-480b-9669-46a37648043f
csi([[0, 0, 0, 0], [0, 0, 0, 1]], [1, 2, 3, 4], 0, 5)

# ╔═╡ 80646a5b-ab83-4214-979a-152815b42f51
sampled_values[1:3]

# ╔═╡ 6f92bcf3-1f6f-46dd-91ff-871b448b27e3
matmat = [
	1 2 3 4 5;
	6 7 8 9 10;
	11 12 13 14 15
] .|> Float64

# ╔═╡ eebd69b0-f702-40cb-ab3a-f2b1fd154c0d
heatmap(matmat)

# ╔═╡ 70321d6d-8ec0-43a3-934b-35548e11819e
SNN.Plots.plot_heatmap(matmat, tofile=false, xlabel="ok")

# ╔═╡ 300947b8-d5a3-468e-a42e-3b5c77f3122c
length(1:2:10)

# ╔═╡ 7ca48fda-9ea0-4018-b8af-5225a4b73cda
heatmap_values = ([1.3e-10, 1.5e-10], [4.0e-12, 7.0e-12], Any[0.012666132363720597 0.01653196021779987; -0.044990909793167166 -0.06343718548031357])

# ╔═╡ 3c6a202b-2b0a-43cc-99bc-2d05d50ea4b6
SNN.Plots.plot_heatmap(
    heatmap_values,
	title="csi over params search", name="results/base_3_adaptation_scan_a_b.png", tofile=false, xlabel="a", ylabel="b"
)

# ╔═╡ Cell order:
# ╠═e86eea66-ad59-11ef-2550-cf2588eae9d6
# ╠═31c85e65-cf3e-465a-86da-9a8547f7bec0
# ╠═31fff5a6-7b9f-4bb6-9c2f-8f61c7878fc4
# ╠═cd62780e-8a5a-49eb-b3d1-33c88e966332
# ╠═16720d97-2552-4852-a011-4ea19c8b9d8b
# ╠═2d4149ce-1a89-4087-a7e2-6cf48778ab51
# ╠═1984a541-8ba8-47ff-9cce-1ba29748e200
# ╠═33b33822-5476-4759-a100-1b274456aecc
# ╠═3d936faa-73aa-47fc-be6a-92d0d42d8389
# ╠═905483e2-78b9-40ed-8421-cd1b406003d9
# ╠═dba22b66-ba23-4b2d-83bb-d6f32e9a3e59
# ╠═c39f4a5c-86ec-4e92-a20f-965cf37fc3cb
# ╠═1b5b20d7-3934-406a-9d9e-2af0ad2c13db
# ╠═1f069d6f-4ead-49c5-84da-957c15d69074
# ╠═32e92fca-36d3-4ebd-a228-fd6b3f965694
# ╠═e90677e4-3a63-4d1c-bc8f-0bb13f1a490c
# ╠═786f52c6-f985-4b6f-b5db-04e24f5d48ce
# ╠═b8a8d05d-8e3c-4ec1-a71c-7930a907daf2
# ╠═2341a3bb-d912-4a7e-93ba-fe856d3aaa4d
# ╠═d3927c88-0b17-4fa5-98ce-9fc2378b5bb0
# ╠═34f656f8-c572-4cf5-9927-312fc2caae20
# ╠═b594ab34-5cc8-4850-bb63-3748fbb418a0
# ╠═28eda74b-3c55-4694-be86-906e51be760b
# ╠═eb0a72e1-a1c3-463c-a6fc-ade4f0d53eab
# ╠═dabc4957-f625-40b9-b9b2-3d12bf399d0c
# ╠═337b1d88-5246-40d5-bc86-ee0b5dbc2218
# ╠═a90c94ad-924c-49d4-9526-78d5c25c9fc7
# ╠═85809d17-2d85-4de5-b450-6526d0d3cec9
# ╠═3ae6f318-21c3-44cf-bb92-2836883229b4
# ╠═6a0f1c02-02c3-4f40-9537-5a82bb773b7b
# ╠═fc6227bc-9036-41d9-8fe3-4f947a227fe0
# ╠═63ebbfb4-ae1e-4d80-a08f-ee35a1fdbcfb
# ╠═1a97fd70-8dc2-45ab-b480-70e7e3651205
# ╠═d0fdb66a-c87b-40d5-8caa-04484b0cb5c6
# ╠═af3cd829-d768-417a-aaea-cf0dde63ba54
# ╠═baa3784f-70a5-46f7-a8db-d5e102026411
# ╠═bc881bde-e581-4459-8ee3-ce0cc7fefb6d
# ╠═3a999581-169f-4bc9-a74a-8317fb2b7523
# ╠═5f8072f2-aa4c-4150-97f7-6db1fe89bfdf
# ╠═6f1e73ba-cde0-4915-828b-299017f274cd
# ╠═35d3c8cb-f3f7-4dac-91f1-538f6b9fb29e
# ╠═1935208e-36bc-42e2-9b75-e8050ed86b42
# ╠═c766d492-d335-45bd-8468-5701262235c0
# ╠═7adf645a-9a09-4193-bf19-0c847350ae56
# ╠═77617fc7-fb5e-4d52-a0dd-c2c89c7c9110
# ╠═fa2d60e5-89af-41ba-af1e-516a71c43f5a
# ╠═0bc9dd59-e29f-4854-ba6b-697adde0105d
# ╠═cc2d1c5a-5f57-4ce9-9e31-236b6f81a9fd
# ╠═7e7f24b1-5ed7-48ea-ab91-061a6383cc29
# ╠═e46d4fb1-63ee-48a8-adde-f284ffe71cf8
# ╠═a5a05532-9ebe-454e-983c-5a74436e3e3f
# ╠═9eebabbb-e0f6-4444-a63a-d750be61523d
# ╠═23a9143a-d394-41ed-89de-505c86a64de0
# ╠═0ef1fd05-7ed6-422b-8a87-c4c8011f5435
# ╠═71612ba8-c43c-490c-8581-def0085eedf2
# ╠═5174d0e7-9d37-42a7-a13f-30d8f71a52ef
# ╠═672a833a-5c91-4397-8ce2-fbcf343beb75
# ╠═96951fbb-ba5a-4606-ac04-e3c9bd70c0d2
# ╠═a0923e08-6ad3-45c9-9d0e-3a5d78338749
# ╠═46073470-7f50-4cd4-98a7-3eb5043e6bab
# ╠═81eaed1e-b1f8-4129-92e1-d6b33bc4106e
# ╠═4e1edae3-c304-43f1-8aa6-8fee655436e7
# ╠═6107f3f1-1e0c-4ec4-b5ea-49216e5472ff
# ╠═696d3954-6373-4d3e-8ff5-764814cd4bec
# ╠═c0021651-ed41-4b04-b9f2-5b282aa2553f
# ╠═a0b9ea82-a228-4e8a-bba3-cd255708380f
# ╠═3996e2c9-905f-424b-8acb-f9ba49346a17
# ╠═a9aecee8-45bc-4f6e-b68c-5884c6d8a58c
# ╠═b5cb6703-e19a-4e22-b9a8-dd8d037448eb
# ╠═ed473424-9f60-480b-9669-46a37648043f
# ╠═80646a5b-ab83-4214-979a-152815b42f51
# ╠═6f92bcf3-1f6f-46dd-91ff-871b448b27e3
# ╠═eebd69b0-f702-40cb-ab3a-f2b1fd154c0d
# ╠═70321d6d-8ec0-43a3-934b-35548e11819e
# ╠═300947b8-d5a3-468e-a42e-3b5c77f3122c
# ╠═7ca48fda-9ea0-4018-b8af-5225a4b73cda
# ╠═3c6a202b-2b0a-43cc-99bc-2d05d50ea4b6
