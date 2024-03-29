export elasticqrfact!, elasticqrfact, updateelasticqrfact!

# We cannot define subtypes of concrete types in Julia
# struct ElasticQR{T, S<:AbstractMatrix{T}} <: Factorization{T}
#     factors::ElasticArray{T,2,1,Array{T,1}}
#     τ::Array{T,1}
#     # cache::Array{Float64,1}
# end
# ElasticQR(factors::AbstractMatrix{T}, τ::AbstractVector{T}) where {T} = ElasticQR{T}(ElasticMatrix(factors), τ)

function elasticqrfact!(A::AbstractMatrix{T}) where {T}
    LinearAlgebra.require_one_based_indexing(A)
    m, n = size(A)
    τ = zeros(T, min(m,n))
    @inbounds for k = 1:min(m - 1 + !(T<:Real), n)
        x = view(A, k:m, k)
        τk = LinearAlgebra.reflector!(x)
        τ[k] = τk
        LinearAlgebra.reflectorApply!(x, τk, view(A, k:m, k + 1:n))
    end
    QR(ElasticMatrix(A), τ)
end

function elasticqrfact(A::AbstractMatrix{T}, arg...; kwargs...) where T
    LinearAlgebra.require_one_based_indexing(A)
    AA = similar(A, LinearAlgebra._qreltype(T), size(A))
    copyto!(AA, A)
    return elasticqrfact!(AA, arg...; kwargs...)
end

function updateelasticqrfact!(S::QR{T,ElasticArray{T,2,1,Array{T,1}}}, a::AbstractVector{T}, cache::AbstractVector{T}) where {T}
    m, n = size(S.factors)
    mul!(cache, S.Q', a)
    @inbounds append!(S.factors, cache)
    x = view(S.factors, n+1:m, n+1)
    τk = LinearAlgebra.reflector!(x)
    @inbounds push!(S.τ, τk)
    nothing
end

updateelasticqrfact!(S::QR{T,ElasticArray{T,2,1,Array{T,1}}}, a::AbstractVector{T}) where {T} = updateelasticqrfact!(S, a, zero(a))

function updateelasticqrfact!(S::QR{T,ElasticArray{T,2,1,Array{T,1}}}, A::AbstractMatrix{T}, cache) where {T}
    m, n = size(S.factors)
    mA, nA = size(A)
    @assert mA == m "Length of the columns do not match"

    @inbounds mul!(cache, S.Q', A)
    @inbounds append!(S.factors, cache)
    # @for k=n+1:n+nA
    #     append!(S.factors, S.Q'

    @inbounds for k = n+1:min(m - 1 + !(T<:Real), n+nA)
        x = view(S.factors, k:m, k)
        τk = LinearAlgebra.reflector!(x)
        push!(S.τ, τk)
        LinearAlgebra.reflectorApply!(x, τk, view(S.factors, k:m, k + 1:n + nA))
    end
end

updateelasticqrfact!(S::QR{T,ElasticArray{T,2,1,Array{T,1}}}, A::AbstractMatrix{T}) where {T} = updateelasticqrfact!(S, A, zero(A))
