export UpdatableQR, updateqrfactUnblocked!

struct UpdatableQR{T} <: Factorization{T}  begin
    factors::Array{Array{T,1},1}
    τ::Array{Float64,1}
end

function UpdatableQR(A::AbstractMatrix{T}) where T


end

function updateqrfactUnblocked!(S::QR{T,Array{T,2}}, a::AbstractVector{T}) where {T}
    m, n = size(S.factors)
    # Add one entry to F.τ
    push!(S.τ, 0.0)

    # Add one column to F.factors
    factors = hcat(S.factors, S.Q'*a)

    @inbounds for k = n+1:min(m - 1 + !(T<:Real), n+1)
        x = view(factors, k:m, k)
        τk = LinearAlgebra.reflector!(x)
        S.τ[k] = τk
        LinearAlgebra.reflectorApply!(x, τk, view(factors, k:m, k + 1:n + 1))
    end
    S = QR(factors, S.τ)
    return S
end
