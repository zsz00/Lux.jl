using SafeTestsets, Test, Pkg

const GROUP = get(ENV, "GROUP", "All")

function dev_subpkg(subpkg)
    subpkg_path = joinpath(dirname(@__DIR__), "lib", subpkg)
    Pkg.develop(PackageSpec(path=subpkg_path))
end

function activate_subpkg_env(subpkg)
    subpkg_path = joinpath(dirname(@__DIR__), "lib", subpkg)
    Pkg.activate(subpkg_path)
    Pkg.develop(PackageSpec(path=subpkg_path))
    Pkg.instantiate()
end

groups = if GROUP == "All"
    ["Lux", "Boltz"]
else
    [GROUP]
end

@time begin for group in groups
    @info "Testing GROUP $group"
    if group == "Lux"
        @testset "Lux.jl" begin
            @testset "Layers" begin
                @time @safetestset "Basic" begin include("layers/basic.jl") end
                @time @safetestset "Convolution" begin include("layers/conv.jl") end
                @time @safetestset "Normalization" begin include("layers/normalize.jl") end
                @time @safetestset "Recurrent" begin include("layers/recurrent.jl") end
                @time @safetestset "Dropout" begin include("layers/dropout.jl") end
            end

            @time @safetestset "Functional Operations" begin include("functional.jl") end
        end
    else
        dev_subpkg(group)
        subpkg_path = joinpath(dirname(@__DIR__), "lib", group)
        Pkg.test(PackageSpec(name=group, path=subpkg_path))
    end
end end
