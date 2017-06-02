using Documenter, ScatteredInterpolation

makedocs(
    format = :html,
    sitename = "ScatteredInterpolation.jl",
    pages = [
        "Home" => "index.md",
        "Supported methods" => "methods.md",
        "API" => "api.md",
    ]
)
