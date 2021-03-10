export ElasticQR, elasticqrfact


struct ElasticQR{T} <: Factorization{T}
    factors::ElasticArray{T,2,1,Array{T,1}}
    τ::Array{T,1}
    # cache::Array{Float64,1}
end

function elasticqrfact(A::AbstractMatrix{T}, arg...; kwargs...) where T
    LinearAlgebra.require_one_based_indexing(A)
    AA = similar(A, LinearAlgebra._qreltype(T), size(A))
    copyto!(AA, A)
    return LinearAlgebra.qrfactUnblocked!(AA, arg...; kwargs...)
end


function updateqrfactUnblocked!(S::ElasticQR{T}, a::AbstractVector{T}) where {T}
    m, n = size(S.factors)
    # Add one entry to F.τ

    # Add one column to F.factors
    # factors = hcat(S.factors, S.Q'*a)

    # @inbounds for k = n+1:min(m - 1 + !(T<:Real), n+1)
        mul!(S.cache, S.Q', a)
        push!(S.factors, copy(cache))
        # view(factors,:,1:n) .= S.factors
        # mul!(view(factors,:,n+1), , a)
        # factors = hcat(S.factors, S.Q'*a)
        x = view(S.factors, n+1:m, n+1)
        τk = LinearAlgebra.reflector!(x)
        push!(S.τ, τk)
        # LinearAlgebra.reflectorApply!(x, τk, view(factors, k:m, k + 1:n + 1))
    # end
    # return QR(factors, S.τ)
end
