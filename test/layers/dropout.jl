using JET, Lux, Random, Test

include("../utils.jl")

rng = Random.default_rng()
Random.seed!(rng, 0)

@testset "Dropout" begin
    layer = Dropout(0.5f0)
    println(layer)
    ps, st = Lux.setup(rng, layer)
    x = randn(Float32, 5, 2)

    x_, st_ = layer(x, ps, st)
    x__, st__ = layer(x, ps, st)
    x___, st___ = layer(x_, ps, st_)

    @test st_.rng != st.rng
    @test st_.rng == st__.rng
    @test x_ == x__
    @test x_ != x___

    @test_call layer(x, ps, st)
    @test_opt target_modules=(Lux,) layer(x, ps, st)
    test_gradient_correctness_fdm(x -> sum(layer(x, ps, st)[1]), x; atol=1.0f-3,
                                  rtol=1.0f-3)

    st = Lux.testmode(st)

    @test first(layer(x, ps, st)) == x
end

@testset "VariationalHiddenDropout" begin
    layer = VariationalHiddenDropout(0.5f0)
    println(layer)
    ps, st = Lux.setup(rng, layer)
    x = randn(Float32, 5, 2)

    x_, st_ = layer(x, ps, st)
    x__, st__ = layer(x, ps, st)
    x___, st___ = layer(x_, ps, st_)

    @test st_.rng != st.rng
    @test st_.rng == st__.rng
    @test st_.mask == st__.mask
    @test x_ == x__
    @test x_ != x___

    @test_call layer(x, ps, st)
    @test_call layer(x, ps, st_)
    @test_opt target_modules=(Lux,) layer(x, ps, st)
    @test_opt target_modules=(Lux,) layer(x, ps, st_)
    test_gradient_correctness_fdm(x -> sum(layer(x, ps, st)[1]), x; atol=1.0f-3,
                                  rtol=1.0f-3)
    test_gradient_correctness_fdm(x -> sum(layer(x, ps, st_)[1]), x; atol=1.0f-3,
                                  rtol=1.0f-3)

    st__ = Lux.update_state(st_, :update_mask, Val(true))
    x___, st___ = layer(x, ps, st__)

    @test st___.mask != st__.mask
    @test x___ != x_

    @test_call layer(x, ps, st__)
    @test_opt target_modules=(Lux,) layer(x, ps, st__)
    test_gradient_correctness_fdm(x -> sum(layer(x, ps, st__)[1]), x; atol=1.0f-3,
                                  rtol=1.0f-3)
end
