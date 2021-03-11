export batchqromp

# Write a batch version of the QR-OMP code

function batchqromp(ψ::AbstractMatrix{T}, u::AbstractVector{T}; invert::Bool=true, verbose::Bool = true, ϵrel::Float64 = 1e-1, batchterms::Int64 = 1, maxterms::Int64=typemax(Int64)) where {T}
    m, n = size(ψ)

    residue = copy(u)
    idxset = Int64[]
    dict = collect(1:n)
    ϵu = norm(u)
    ϵhist = [ϵu]
    ϵrel *= ϵu
    ϕ = zeros(T, n)
    ratio = zeros(T,n)
    cache = zeros(T, m, batchterms)


    @inbounds for j=1:n
        for i=1:m
        ϕ[j] += ψ[i,j]^2
        end
    end

    cache = zeros(T, m, batchterms)
    F = elasticqrfact(cache)
    ek = zeros(T, n, batchterms)
    q = zeros(T, m, batchterms)
    new_idx = zeros(Int64, batchterms)

    @inbounds for k=1:div(n, batchterms)
        # Step  7 choose best candidate

        @fastmath @inbounds for (j, idx) in enumerate(dict)
            ψj = view(ψ,:,idx)
            ratio[idx] = dot(residue, ψj)^2/ϕ[idx]

            # if ratio >= entry
            #     entry = copy(ratio)
            #     new_idx = copy(idx)
            # end
        end
        new_idx .= partialsortperm(view(ratio, dict), 1:batchterms, rev = true)

        # Step 8 Update set of selected basis
        append!(idxset, new_idx)

        # Step 9 Update candidate dictionary
        filter!(x-> x != new_idx, dict)

        # Step 10 Update Q(k-1) and R(k-1) with ψi(k)
        if k>1
            # F = updateqrfactUnblocked!(F, view(ψ,:,new_idx))
            @time updateelasticqrfact!(F, view(ψ,:,new_idx), cache)
            @inbounds for i=1:batchterms
                ek[(k-2)*batchterms+i,i] = 0.0
            end
        else
            # F = qrfactUnblocked(ψ[:,new_idx:new_idx])
            F = elasticqrfact(view(ψ,:,new_idx))
        end

        # Step 11 Update residual
        @inbounds for i=1:batchterms
            ek[(k-1)*batchterms+i,i] = 1.0
        end

        mul!(q, F.Q, view(ek,1:k,1:batchterms))

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
        if ϵ < ϵrel || k == maxterms #|| size(idxset,1) + batchterms > n
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
