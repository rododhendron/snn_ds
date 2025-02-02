module Utils
using ComponentArrays: ComponentVecOrMat
using ModelingToolkit, DifferentialEquations, Symbolics, Transducers, YAML
using ComponentArrays, JLD2
using ModelingToolkit: AbstractODESystem

export ParamTree, fetch_tree, get_spike_timings, get_spikes_from_r

mutable struct ParamNode
    nodes::Vector{Union{ParamNode,Num,Nothing}}
    system::ODESystem
end

struct ParamTree
    master_node::ParamNode
end

function instantiate_systems(model::AbstractODESystem)
    nodes = []
    for system in propertynames(model)
        push!(nodes, instantiate_systems(getproperty(model, system)))
    end
    ParamNode(nodes, model)
end

function instantiate_systems(model::Num)
    model
end

function instantiate_systems(model::Nothing)
    nothing
end

function fetch_node(_, node::Nothing, _)
    nothing
end

function fetch_node(path, node::Num, isin::Bool)
    split_sym = "₊"
    node_string = node |> Symbol |> String |> x -> split(x, split_sym) |> last |> x -> split(x, "(") |> first
    if !isin && path[1] == node_string
        node
    elseif isin && occursin(path[1], node_string)
        node
    else
        nothing
    end
end

function take_current_node_name(name::Symbol)::String
    split_sym = "₊"
    split(String(name), split_sym) |> first
end

function fetch_node(path, node::ParamNode, isin::Bool)
    match_fn(x) = isin ? occursin(path[1], x.system.name |> String) : path[1] == x.system.name |> take_current_node_name
    matched_nodes = filter(node -> typeof(node) == ParamNode && match_fn(node), node.nodes)
    if !isempty(matched_nodes)
        fetch_node.(Ref(path), matched_nodes, isin)
    else
        fetch_node.(Ref(path[2:end]), node.nodes, isin)
    end
end

function fetch_tree(path::Vector{String}, tree::ParamTree, isin::Bool=true) # isin lets us match exactly specified numerical value
    # knowing required path, fetch specific matching param in tree
    fetch_node(path, tree.master_node, isin) |> Cat() |> Cat() |> Filter(!isnothing) |> collect
end

function make_param_tree(model::AbstractODESystem)::ParamTree
    master_node = instantiate_systems(model)
    tree::ParamTree = ParamTree(master_node)
end

function fetch_uparameters_from_symbol(model::AbstractODESystem, sym_to_fetch::Symbol)::Vector{SymbolicUtils.BasicSymbolic{Real}}
    replaced_parameter(param) = split(param |> Symbol |> String, "(")[1]
    params = parameters(model) .|> replaced_parameter
end

function get_spikes_from_r(r_array)
    # if dims == time, features
    res_array = zeros(eltype(r_array), size(r_array))
    for i in 1:size(res_array, 1)
        for j in 2:size(res_array, 2)
            res_array[i, j] = r_array[i, j-1] < r_array[i, j] ? 1.0 : 0.0
        end
    end
    res_array
end

function get_spike_timings(r_spikes::BitVector, sol)
    sol.t[r_spikes]
end

function get_spike_timings(r_array::Matrix, sol)
    r_spikes = get_spikes_from_r(r_array) .|> Bool
    timings_vec = []
    for i in 1:size(r_array, 1)
        push!(timings_vec, get_spike_timings(r_spikes[i, :], sol))
    end
    timings_vec
end

function hcat_sol_matrix(res, sol)
    reduce(hcat, sol[res])
end

function write_params(params; name)
    YAML.write_file(name, Dict(params))
end

function write_params(components::ComponentVecOrMat; name)
    params_values = getindex.(Ref(components), labels(components))
    params = Dict(zip(labels(components), params_values))
    YAML.write_file(name, params)
end

function write_params(params::Dict; name)
    YAML.write_file(name, params)
end

function write_sol(sol; name)
    @save name sol
end

function get_matching_timings(stims::Vector, spikes::Vector, window::Float64)
    @show stims
    @show spikes
    stims |> Map(x -> count(i -> x < i < x + window, spikes)) |> collect
end

end
