
@testset "Test ElasticQR decompostion" begin

    ψ = randn(100, 20)
    c = randn(20)
    u = ψ*c

    F = qrfactUnblocked(ψ)
    Felastic = elasticqrfact(ψ)

    @test norm(F.factors - Felastic.factors)<5e-14
    @test norm(F.τ - Felastic.τ)<5e-14
    @test norm(Felastic\u - c)<5e-14
end

@testset "Verify updateelasticqrfact!" begin

    ψ = randn(100, 20)
    ϕ = randn(100)

    F = elasticqrfact(ψ)
    Fold = deepcopy(F)
    G = elasticqrfact(hcat(ψ, ϕ))
    updateelasticqrfact!(F, ϕ)
    @test norm(F.τ-G.τ)<1e-14
    @test norm(F.factors - G.factors)<1e-14
end
