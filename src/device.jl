module Device

export to_device_fn

using CUDA, AMDGPU

function to_device_fn()
    if CUDA.functional()
        return CuArray
    elseif AMDGPU.functional()
        return ROCArray
    else
        return x -> x
    end
end

end
