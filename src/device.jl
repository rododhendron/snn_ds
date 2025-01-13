module Device

export to_device_fn


function to_device_fn(;backend="")
    if backend != ""
        @eval begin
            using CUDA, AMDGPU
        end
        if CUDA.functional()
            return CuArray
        elseif AMDGPU.functional()
            return ROCArray
        else
            return x -> x
        end
    else
        return x -> x
    end
end

end
