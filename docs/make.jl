using Documenter, ScatteredInterpolation

DocMeta.setdocmeta!(ScatteredInterpolation, :DocTestSetup,
                    :(using ScatteredInterpolation); recursive = true)

makedocs(
    modules = [ScatteredInterpolation],
    sitename = "ScatteredInterpolation.jl",
    authors = "Emil Ljungskog and contributors",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://eljungsk.github.io/ScatteredInterpolation.jl",
    ),
    # Only require exported symbols to be documented, so an internal docstring can't
    # unexpectedly fail the docs build (and block deployment).
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
        "Supported methods" => "methods.md",
        "API" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/eljungsk/ScatteredInterpolation.jl.git",
    devbranch = "master",
    push_preview = true,
)
