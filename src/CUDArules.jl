module CUDArules
    using CUDA
    using ChainRulesCore

    to_gpu(a::AbstractArray) = CuArray(a)
    function rrule(::typeof(to_gpu), a::AbstractArray)
        project_a = ProjectTo(a)
        Ω = to_gpu(a)
        pb(Δ) = (NoTangent(), project_a(collect(Δ)))
        return Ω, pb
    end

    to_cpu(a::AbstractArray) = a
    to_cpu(a::CuArray) = Array(a)
    function rrule(::typeof(to_cpu), a::CuArray)
        Ω = to_cpu(a)
        pb(Δ) = (NoTangent(), CuArray(Δ))
        return Ω, pb
    end

end