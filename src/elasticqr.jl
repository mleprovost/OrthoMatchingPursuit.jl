export elasticqrfact!, elasticqrfact, updateelasticqrfact!

import Base: size, show, getproperty

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
    for k = 1:min(m - 1 + !(T<:Real), n)
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

function updateelasticqrfact!(S::QR{T,ElasticArray{T,2,1,Array{T,1}}}, a::AbstractVector{T}) where {T}
    m, n = size(S.factors)
    @inbounds append!(S.factors, S.Q'*a)
    # Add one entry to F.τ

    # Add one column to F.factors
    # factors = hcat(S.factors, S.Q'*a)

    # @inbounds for k = n+1:min(m - 1 + !(T<:Real), n+1)
        # mul!(S.cache, S.Q', a)
        # push!(S.factors, copy(cache))
        # view(factors,:,1:n) .= S.factors
        # mul!(view(factors,:,n+1), , a)
        # factors = hcat(S.factors, S.Q'*a)
    x = view(S.factors, n+1:m, n+1)
    τk = LinearAlgebra.reflector!(x)
    @inbounds push!(S.τ, τk)
    nothing
        # LinearAlgebra.reflectorApply!(x, τk, view(factors, k:m, k + 1:n + 1))
    # end
    # return QR(factors, S.τ)
end


#
# function show(io::IO, mime::MIME{Symbol("text/plain")}, F::ElasticQR{T}) where {T}
#     summary(io, F); println(io)
#     println(io, "Q factor:")
#     show(io, mime, F.Q)
#     println(io, "\nR factor:")
#     show(io, mime, F.R)
# end
#
# function getproperty(F::ElasticQR{T}, d::Symbol) where {T}
#     m, n = size(F)
#     if d === :R
#         return triu!(getfield(F, :factors)[1:min(m,n), 1:n])
#     elseif d === :Q
#         return QRPackedQ(getfield(F, :factors), F.τ)
#     else
#         getfield(F, d)
#     end
# end


# Define tools for matrix multiplication
