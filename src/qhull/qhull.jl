using StaticArrays

struct Facet{N, S <: Integer}

    vertices::SVector{N, S}
    neighbors::SVector{N, S}
end

struct Mesh{N, T <: Real, S <: Integer}

    facets::Vector{Facet{N,S}}
    points::Matrix{T}
end

"""
    delaunay(points::Matrix{<:Real}) -> Mesh

Compute the Delaunay triangulation of a set of points.

`points` should be an `m`x`n` matrix, where `m` is the number of dimensions and `n` the
number of points, i.e. each column of `points` define one point.
"""
function delaunay(points::Matrix{<:Real})

    # Setup for a ccall
    intPtr = Array{Ptr{Cint}}
    vertexList_c = intPtr(1)
    facetList_c = intPtr(1)
    neighborList_c = intPtr(1)
    nFacets_c = Array{Cint}(1)

    nDims, nPoints = size(points)

    # Call qhull to construct the Delaunay triangulation
    ccall((:delaunay, "../../deps/libwrapqhull"),
          Void,
          (Ptr{Ptr{Cint}}, 
           Ptr{Ptr{Cint}}, 
           Ptr{Ptr{Cint}}, 
           Ptr{Cint}, 
           Ptr{Cdouble}, 
           Cint, 
           Cint), 
          vertexList_c, 
          neighborList_c, 
          facetList_c, 
          nFacets_c,
          points,
          nDims,
          nPoints)

    # Convert back to Julia types
    nFacets = Int(nFacets_c[1])
    vertexList = unsafe_wrap(Array, vertexList_c[1], (nDims+1, nFacets), false)
    neighborList = unsafe_wrap(Array, neighborList_c[1], (nDims+1, nFacets), false)
    facetList = unsafe_wrap(Array, facetList_c[1], nFacets, false)

    facets = Vector{Facet{nDims+1, eltype(neighborList)}}()
    neighbors = MVector{nDims+1, eltype(neighborList)}()

    # Build a mesh
    for j = 1:nFacets

        vertices = @view vertexList[:,j]

        # Find the neighbors. Use indices from 1:nFacets instead of the id:s from qhull
        # returned by the ccall
        for i = 1:nDims+1

            n = neighborList[i,j]

            # Check for neighbors outside of the domain
            if n == 0
                neighbors[i] = 0
            else
                pos = searchsortedfirst(facetList, n)
                neighbors[i] = pos
            end
        end

        push!(facets, Facet(SVector{nDims+1}(vertices), SVector(neighbors)))
    end

    Mesh(facets, points)
end
