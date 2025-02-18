module Device

export to_device_fn


function to_device_fn(; backend="")
    if backend != ""
        if backend == "CUDA"
            @eval begin
                using CUDA#, AMDGPU
            end
            if CUDA.functional()
                return CuArray
            else
                return x -> x
            end
        else
            @eval begin
                using AMDGPU
            end
            if AMDGPU.functional()
                return ROCArray
            else
                return x -> x
            end
        end
    else
        return x -> x
    end
end

end
