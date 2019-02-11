using Documenter, ScatteredInterpolation

makedocs(
    sitename = "ScatteredInterpolation.jl",
    pages = [
        "Home" => "index.md",
        "Supported methods" => "methods.md",
        "API" => "api.md",
    ]
)

deploydocs(
    repo = "github.com/eljungsk/ScatteredInterpolation.jl.git",
    target = "build",
)
