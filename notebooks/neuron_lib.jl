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
using Symbolics, ModelingToolkit, DifferentialEquations, RecursiveArrayTools, SymbolicIndexingInterface, CairoMakie, ComponentArrays, AlgebraOfGraphics, Tables, LinearAlgebra

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
	tspan = (0, 20)
	
	# make schedule
	# stim_params.n_trials = 20
	stim_params.amplitude = 8e-9
	stim_params.duration = 50.0e-3
	stim_params.deviant_idx = 5
	stim_params.standard_idx = 1
	stim_params.p_deviant = 0.1
	stim_params.start_offset = 1
	stim_params.isi = 300e-3
	
	stim_schedule = SNN.Params.generate_schedule(stim_params, tspan)
	
	sch_t = deepcopy(stim_schedule[1, :])
	sch_onset = deepcopy(stim_schedule[2, :])
	sch_group = deepcopy(stim_schedule[3, :])
	
	# @time params = Neuron.AdExNeuronParams()
	params.inc_gsyn = 20e-9
	params.a = 3e-9          # Subthreshold adaptation (A)
	params.b = 6e-10          # Spiking adaptation (A)
	params.TauW = 350.0e-3      # Adaptation time constant (s)
	params.Cm = 3e-10
	
	params.Ibase = 4.33e-10
	# params.Ibase = 0
	params.sigma = 0.08

	rules = []
	# push!(rules, SNN.Params.make_rule("e_neuron", 3, "soma__Ibase", 2e-10))
	merged_params = SNN.Params.override_params(params, rules)
end

# ╔═╡ 1e684c34-1818-483d-a0a0-11e044c8fa70


# ╔═╡ dba22b66-ba23-4b2d-83bb-d6f32e9a3e59
begin
	con_mapping = [
		(SNN.Params.@connect_neurons [1] SNN.Neuron.AMPA() 2)...;
		(SNN.Params.@connect_neurons [5] SNN.Neuron.AMPA() 4)...;
		(SNN.Params.@connect_neurons [2, 4] SNN.Neuron.AMPA() 3)...
	    # (1, 3, SNN.Neuron.AMPA()),
		# (1, 2, SNN.Neuron.GABAa()),
		# (2, 1, SNN.Neuron.GABAa()),
	    # (2, 3, SNN.Neuron.AMPA())
	]
	@show con_mapping
	#
	pre_neurons = [row[1] for row in con_mapping]
	post_neurons = [row[2] for row in con_mapping]
	e_neurons_n = size(unique([pre_neurons; post_neurons]), 1)
	# e_neurons_n = 3
end

# ╔═╡ c39f4a5c-86ec-4e92-a20f-965cf37fc3cb
begin
	name = "jitter_network_noise"
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
	    # solver=ISSEulerHeun(),
	    solver=DRI1(),
		tols=(1e-3, 1e-3)
	)
	nothing
end

# ╔═╡ e90677e4-3a63-4d1c-bc8f-0bb13f1a490c
simplified_model.e_neuron_5₊soma₊group

# ╔═╡ 786f52c6-f985-4b6f-b5db-04e24f5d48ce
(start, stop) = tspan

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

# ╔═╡ 3ae6f318-21c3-44cf-bb92-2836883229b4
@time SNN.Plots.plot_spike_rate(i, sol, start, stop, name_interpol, tree, stim_params.start_offset, stim_schedule, false; time_window=1e-1)

# ╔═╡ fc6227bc-9036-41d9-8fe3-4f947a227fe0
@time SNN.Plots.plot_aggregated_rate(i, sol, name_interpol, tree, stim_schedule, false)

# ╔═╡ 63ebbfb4-ae1e-4d80-a08f-ee35a1fdbcfb
@time SNN.Plots.plot_isi(i, sol, start, stop, name_interpol, tree, stim_params.start_offset, stim_schedule, false)

# ╔═╡ 1a97fd70-8dc2-45ab-b480-70e7e3651205
stim_schedule

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
# ╠═1e684c34-1818-483d-a0a0-11e044c8fa70
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
# ╠═3ae6f318-21c3-44cf-bb92-2836883229b4
# ╠═fc6227bc-9036-41d9-8fe3-4f947a227fe0
# ╠═63ebbfb4-ae1e-4d80-a08f-ee35a1fdbcfb
# ╠═1a97fd70-8dc2-45ab-b480-70e7e3651205
