export pivotedqr

function pivotedqr(ψ::AbstractMatrix{T}, u::AbstractVector{T}; invert::Bool=true, verbose::Bool = true, ϵrel::Float64 = 1e-1, maxterms::Int64=typemax(Int64)) where {T}

    m, n = size(ψ)

    # Compute a pivoted QR decomposition of ψ, use the pivot for the sensor placement

    F = qr(ψ, Val(true))

    residue = copy(u)
    idxset = Int64[]
    dict = collect(1:n)
    ϵu = norm(u)
    ϵhist = [ϵu]
    ϵrel *= ϵu

    @inbounds for k=1:n
        # Update set of selected basis
        push!(idxset, F.p[k])

        # Update candidate dictionary
        filter!(x-> x != F.p[k], dict)

        # Compute residual
        c = view(ψ,:,idxset)\u
        residue = u - view(ψ,:,idxset)*c

        # Calculate stopping critera
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
            c .= view(ψ,:, idxset)\u
            return idxset, c, ϵhist
        else
            return idxset, ϵhist
        end
    else
        if invert == true
            c = zeros(length(idxset))
            c .= view(ψ,:, idxset)\u
            return idxset, c
        else
            return idxset
        end
    end
end
