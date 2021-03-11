# Write a batch version of the QR-OMP code

function batchqromp(ψ::AbstractMatrix{T}, u::AbstractVector{T}; invert::Bool=true, verbose::Bool = true, ϵrel::Float64 = 1e-1, batchterms::Int64 = 2, maxterms::Int64=typemax(Int64)) where {T}
    m, n = size(ψ)

    residue = copy(u)
    idxset = Int64[]
    dict = collect(1:n)
    ϵu = norm(u)
    ϵhist = [ϵu]
    ϵrel *= ϵu
    ϕ = zeros(T, n)


    @inbounds for j=1:n
        for i=1:m
        ϕ[j] += ψ[i,j]^2
        end
    end

    # F = qrfactUnblocked(zeros(0,0))
    # F = elasticqrfact(zeros(m,0))
    cache = zeros(T, m)
    F = elasticqrfact(reshape(cache, length(cache),1))
    ek = zeros(T, n)
    # ektest = zeros(T, n)
    q = zeros(T, m)
    new_idx = 1

    @inbounds for k=1:n
        # Step 6 & 7 update denominators and choose best candidate

        entry = 0.0
        @fastmath @inbounds for (j, idx) in enumerate(dict)
            ψj = view(ψ,:,idx)
            ratio = dot(residue, ψj)
            ratio = ratio^2
            ratio /= ϕ[idx]

            if ratio >= entry
                entry = copy(ratio)
                new_idx = copy(idx)
            end
        end

        # Step 8 Update set of selected basis
        push!(idxset, new_idx)

        # Step 9 Update candidate dictionary
        filter!(x-> x != new_idx, dict)

        # Step 10 Update Q(k-1) and R(k-1) with ψi(k)
        if k>1
            # F = updateqrfactUnblocked!(F, view(ψ,:,new_idx))
            updateelasticqrfact!(F, view(ψ,:,new_idx), cache)

            ek[k-1] = 0.0
        else
            # F = qrfactUnblocked(ψ[:,new_idx:new_idx])
            F = elasticqrfact(view(ψ,:,new_idx:new_idx))
        end

        # Step 11 Update residual
        ek[k] = 1.0
        mul!(q, F.Q, view(ek,1:k))

        @inbounds for (j, idx) in enumerate(dict)
            ψj = view(ψ,:,idx)
            ϕj = dot(q, ψj)
            ϕ[idx] -= ϕj^2
        end
        factor = dot(q, u)
        residue -= factor*q

        # Step 12 Calculate stopping critera
        ϵ = norm(residue)

        if verbose == true
            push!(ϵhist, copy(ϵ))
        end
        if ϵ < ϵrel || k == maxterms
            break
        end
    end

    # Solve the system with the set of indices
    if verbose == true
        if invert == true
            c = zeros(length(idxset))
            ldiv!(c, F, u)
            return idxset, c, ϵhist
        else
            return idxset, ϵhist
        end
    else
        if invert == true
            c = zeros(length(idxset))
            ldiv!(c, F, u)
            return idxset, c
        else
            return idxset
        end
    end
end
