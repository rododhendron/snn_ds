module SymbolicsAMDGPU


using ModelingToolkit: DiffEqBase
using ModelingToolkit
using AMDGPU
using Symbolics
using SciMLBase
using Adapt

# Function cache for GPU-compiled functions
const GPU_FUNCTION_CACHE = Dict{UInt64,Tuple{Function,Function}}()

function get_or_compile_gpu_functions(sys::ModelingToolkit.AbstractSystem)
    # Create a unique hash for this system
    sys_hash = hash(sys)

    # Check if we've already compiled functions for this system
    if haskey(GPU_FUNCTION_CACHE, sys_hash)
        return GPU_FUNCTION_CACHE[sys_hash]
    end

    # Compile new functions
    f_func, g_func = generate_gpu_compatible_functions(sys)

    # Cache the compiled functions
    GPU_FUNCTION_CACHE[sys_hash] = (f_func, g_func)

    return f_func, g_func
end

# Enhanced SDEProblem constructor with error handling
function ModelingToolkit.SDEProblem(
    sys::SDESystem,
    u0::AbstractArray,
    tspan::Tuple,
    p::AbstractArray=SciMLBase.NullParameters();
    use_gpu::Bool=false,
    fallback_to_cpu::Bool=true,
    kwargs...
)
    # If GPU usage is not requested, use the standard constructor
    if !use_gpu
        return invoke(ModelingToolkit.SDEProblem,
            Tuple{ModelingToolkit.SDESystem,AbstractArray,Tuple,AbstractArray},
            sys, u0, tspan, p; kwargs...)
    end

    try
        # Try to create GPU-accelerated problem
        f_func, g_func = generate_gpu_compatible_functions(sys)
        u0_gpu = convert_to_rocarray(u0)
        p_gpu = convert_to_rocarray(p)
        return SDEProblem(f_func, g_func, u0_gpu, tspan, p_gpu; kwargs...)
    catch e
        if fallback_to_cpu
            @warn "GPU acceleration failed: $e. Falling back to CPU implementation."
            return invoke(ModelingToolkit.SDEProblem,
                Tuple{ModelingToolkit.SDESystem,AbstractArray,Tuple,AbstractArray},
                sys, u0, tspan, p; kwargs...)
        else
            rethrow(e)
        end
    end
end

# Helper function to generate GPU-compatible functions from symbolic system
function generate_gpu_compatible_functions(sys::ModelingToolkit.AbstractSystem)
    # Get the drift and diffusion functions from the system
    f_expr = ModelingToolkit.generate_function(sys, expression=Val{false})
    g_expr = ModelingToolkit.generate_diffusion_function(sys, expression=Val{false})

    # Create GPU-compatible versions of these functions
    function f_gpu(du, u, p, t)
        # Ensure inputs are on GPU
        du_gpu = convert_to_rocarray(du)
        u_gpu = convert_to_rocarray(u)
        p_gpu = convert_to_rocarray(p)

        # Call the generated function
        f_expr[1](du_gpu, u_gpu, p_gpu, t)

        # If du was not originally a ROCArray, convert back
        if !(du isa ROCArray)
            copyto!(du, Array(du_gpu))
        end
    end

    function g_gpu(du, u, p, t)
        # Ensure inputs are on GPU
        du_gpu = convert_to_rocarray(du)
        u_gpu = convert_to_rocarray(u)
        p_gpu = convert_to_rocarray(p)

        # Call the generated function
        g_expr[1](du_gpu, u_gpu, p_gpu, t)

        # If du was not originally a ROCArray, convert back
        if !(du isa ROCArray)
            copyto!(du, Array(du_gpu))
        end
    end

    return f_gpu, g_gpu
end

# 1. Type conversion utilities
function convert_to_rocarray(x::AbstractArray)
    if x isa ROCArray
        return x
    else
        # Handle symbolic types by converting to numerical values
        if eltype(x) <: Union{Symbolics.Num,SymbolicUtils.BasicSymbolic}
            x_numeric = convert_symbolic_to_numeric(x)
            return ROCArray(x_numeric)
        elseif eltype(x) <: Pair{<:SymbolicUtils.BasicSymbolic,<:Any}
            # Extract just the numeric values from pairs
            numeric_values = [pair.second for pair in x]
            return ROCArray(numeric_values)
        else
            return ROCArray(x)
        end
    end
end

# # Enhanced convert_to_rocarray with memory management
# function convert_to_rocarray(x::AbstractArray, reuse_buffer=nothing)
#     if x isa ROCArray
#         return x
#     else
#         # Convert to numeric if symbolic
#         if eltype(x) <: Symbolics.Num
#             x_numeric = convert_symbolic_to_numeric(x)

#             # Reuse existing GPU buffer if provided and compatible
#             if reuse_buffer isa ROCArray && size(reuse_buffer) == size(x_numeric)
#                 copyto!(reuse_buffer, x_numeric)
#                 return reuse_buffer
#             else
#                 return ROCArray(x_numeric)
#             end
#         else
#             # Reuse existing GPU buffer if provided and compatible
#             if reuse_buffer isa ROCArray && size(reuse_buffer) == size(x)
#                 copyto!(reuse_buffer, x)
#                 return reuse_buffer
#             else
#                 return ROCArray(x)
#             end
#         end
#     end
# end

# For non-array types, return as is
convert_to_rocarray(x) = x

# Helper function to convert symbolic values to numeric values
function convert_symbolic_to_numeric(x::AbstractArray{<:Symbolics.Num})
    # Extract the numerical value from each symbolic expression
    return map(val -> convert_symbolic_to_numeric(val), x)
end

# More sophisticated symbolic-to-numeric conversion
function convert_symbolic_to_numeric(x::Symbolics.Num)
    if Symbolics.is_constant(x)
        return Symbolics.value(x)
    elseif Symbolics.istree(x)
        # Handle symbolic expressions by substituting known values
        # This would require context from the problem definition
        op = Symbolics.operation(x)
        args = Symbolics.arguments(x)
        numeric_args = map(convert_symbolic_to_numeric, args)

        # Apply the operation to numeric arguments
        if op == +
            return sum(numeric_args)
        elseif op == *
            return prod(numeric_args)
            # Add more operations as needed
        else
            error("Unsupported operation in symbolic expression: $op")
        end
    else
        error("Cannot convert symbolic expression to numeric value: $x")
    end
end

# GPU-optimized function generation
function generate_optimized_gpu_function(expr)
    # Create a function that operates on entire arrays at once
    function optimized_gpu_func(du, u, p, t)
        # Ensure all operations are vectorized
        # Avoid scalar indexing and loops where possible

        # Example of vectorized operation:
        # Instead of:
        # for i in eachindex(u)
        #     du[i] = p[1] * u[i]
        # end

        # Use:
        du .= p[1] .* u

        return nothing
    end

    return optimized_gpu_func
end

# For non-symbolic types, return as is
convert_symbolic_to_numeric(x) = x

end
