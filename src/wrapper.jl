
export greedysolver


function greedysolver(algo::String, ψ::AbstractMatrix{T}, u::AbstractVector{T}; invert::Bool=true, verbose::Bool = true, ϵrel::Float64 = 1e-1, maxterms::Int64=typemax(Int64)) where {T}
    if algo ∈ ["pivotedqr", "pivot"]
        return pivotedqr(ψ, u; invert = invert, verbose = verbose, ϵrel = ϵrel, maxterms = maxterms)
    elseif algo ∈ ["omp", "lsomp"]
        return lsomp(ψ, u; invert = invert, verbose = verbose, ϵrel = ϵrel, maxterms = maxterms)
    elseif algo ∈ ["qromp"]
        return qromp(ψ, u; invert = invert, verbose = verbose, ϵrel = ϵrel, maxterms = maxterms)
    end
end
