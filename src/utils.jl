module Utils
using ModelingToolkit, DifferentialEquations, Symbolics, Transducers

mutable struct ParamNode
    nodes::Vector{Union{ParamNode,Num}}
    system::ODESystem
end

struct ParamTree
    master_node::ParamNode
end

function instantiate_systems(model::ODESystem)
    nodes = []
    for system in propertynames(model)
        push!(nodes, instantiate_systems(getproperty(model, system)))
    end
    ParamNode(nodes, model)
end

function instantiate_systems(model::Num)
    model
end

function fetch_node(path, node::Num)
    split_sym = "₊"
    node_string = node |> Symbol |> String |> x -> split(x, split_sym) |> last |> x -> split(x, "(") |> first
    if occursin(path[1], node_string)
        node
    else
        nothing
    end
end

function fetch_node(path, node::ParamNode)
    matched_nodes = filter(node -> typeof(node) == ParamNode && occursin(path[1], node.system.name |> String), node.nodes)
    if !isempty(matched_nodes)
        fetch_node.(Ref(path), matched_nodes)
    else
        fetch_node.(Ref(path[2:end]), node.nodes)
    end
end

function fetch_tree(path::Vector{String}, tree::ParamTree)
    fetch_node(path, tree.master_node) |> Cat() |> Cat() |> Filter(!isnothing) |> collect
end

function make_param_tree(model::ODESystem)::ParamTree
    master_node = instantiate_systems(model)
    tree::ParamTree = ParamTree(master_node)
end

function fetch_uparameters_from_symbol(model::ODESystem, sym_to_fetch::Symbol)::Vector{SymbolicUtils.BasicSymbolic{Real}}
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

function get_spike_timings(r_array::Matrix, sol)
    spikes_times = r_array |> get_spikes_from_r .|> s -> sol[s][t]
end


end
