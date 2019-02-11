var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#ScatteredInterpolation.jl-1",
    "page": "Home",
    "title": "ScatteredInterpolation.jl",
    "category": "section",
    "text": "Interpolation of scattered data in JuliaScatteredInterpolation.jl interpolates scattered data in arbitrary dimensions. Pages = [\n    \"index.md\",\n    \"methods.md\",\n    \"api.md\"\n]\nDepth = 2"
},

{
    "location": "#Example-Usage-1",
    "page": "Home",
    "title": "Example Usage",
    "category": "section",
    "text": "using ScatteredInterpolationInterpolate data defined on the vertices and in the center of a square using a multiquadratic radial basis function:samples = [0.0; 0.5; 0.5; 0.5; 1.0];\npoints = [0.0 0.0; 0.0 1.0; 1.0 0.0; 0.5 0.5; 1.0 1.0]\';\nitp = interpolate(Multiquadratic(), points, samples)\ninterpolated = evaluate(itp, [0.6; 0.6])If we instead want to use nearest neighbor interpolation, we can runitp = interpolate(NearestNeighbor(), points, samples)\ninterpolated = evaluate(itp, [0.6; 0.6])"
},

{
    "location": "#Gridding-of-data-1",
    "page": "Home",
    "title": "Gridding of data",
    "category": "section",
    "text": "A common use case for scattered interpolation is gridding of data, i.e. interpolation of scattered data to a grid. Using the same data as above, we can interpolate it to a 5x5 gridn = 5\nx = range(0, stop = 1, length = n)\ny = range(0, stop = 1, length = n)\nX = repeat(x, n)[:]\nY = repeat(y\', n)[:]\ngridPoints = [X Y]\'\n\nitp = interpolate(Multiquadratic(), points, samples)\ninterpolated = evaluate(itp, gridPoints)\ngridded = reshape(interpolated, n, n)"
},

{
    "location": "methods/#",
    "page": "Supported methods",
    "title": "Supported methods",
    "category": "page",
    "text": ""
},

{
    "location": "methods/#Supported-methods-1",
    "page": "Supported methods",
    "title": "Supported methods",
    "category": "section",
    "text": "Currently, three different interpolation methods are available; Radial Basis Functions, Inverse Distance Weighting and Nearest Neighbor."
},

{
    "location": "methods/#Radial-Basis-Functions-1",
    "page": "Supported methods",
    "title": "Radial Basis Functions",
    "category": "section",
    "text": "For radial basis function interpolation, the interpolated value at some point  mathbfx is given byu(mathbfx) = displaystyle sum_i = 1^N w_i phi(mathbfx - mathbfx_i)where mathbfx - mathbfx_i is the distance between mathbfx and  mathbfx_i, and phi(r) is one of the basis functions defined below.To use radial basis function interpolation, pass one of the available basis functions as  method to interpolate.If a GeneralizedRadialBasisFunction is used, an additional polynomial term is added in order for the resulting matrix to be positive definite:u(mathbfx) = displaystyle sum_i = 1^N w_i phi(mathbfx - mathbfx_i) + mathbfP(mathbfx)mathbfλwhere mathbfP(mathbfx) is the matrix defining a complete homogeneous symmetric polynomial of degree degree, and mathbfλ is a vector containing the polynomial coefficients."
},

{
    "location": "methods/#Available-basis-functions-1",
    "page": "Supported methods",
    "title": "Available basis functions",
    "category": "section",
    "text": "Multiquadratic\nϕ(r) = sqrt1 + (ɛr)^2\nInverseMultiquadratic\nϕ(r) = frac1sqrt1 + (ɛr)^2\nGaussian\nϕ(r) = e^-(ɛr)^2\nInverseQuadratic\nϕ(r) = frac11 + (ɛr)^2\nPolyharmonic spline\nϕ(r) = \nbegincases\n    beginalign*\n        r^k                       k = 1 3 5  \n        r^k mathrmln(r)        k = 2 4 6 \n    endalign*\nendcases\nThinPlate spline\nA thin plate spline is the special case k = 2 of the polyharmonic splines. ThinPlate() is a shorthand for Polyharmonic(2).\nGeneralizedMultiquadratic\nϕ(r) = left(1 + (ɛr)^2right)^beta\nThe generalzized multiquadratic results in a positive definite system for polynomials of degree m geq lceilbetarceil.\nGeneralizedPolyharmonic spline\nϕ(r) = \nbegincases\n    beginalign*\n        r^k                       k = 1 3 5  \n        r^k mathrmln(r)        k = 2 4 6 \n    endalign*\nendcases\nThe generalized polyharmonic spline results in a positive definite system for polynomials of degree\nbegincases\n    beginalign*\n        m geq leftlceilfracbeta2rightrceil             k = 1 3 5  \n        m = beta + 1                                             k = 2 4 6 \n    endalign*\nendcases"
},

{
    "location": "methods/#Inverse-Distance-Weighting-1",
    "page": "Supported methods",
    "title": "Inverse Distance Weighting",
    "category": "section",
    "text": "Also called Shepard interpolation, the basic version computes the interpolated value at some point mathbfx byu(mathbfx) = \nbegincases \n    fracdisplaystyle sum_i = 1^N w_i(mathbfx) u_i   \n         displaystyle sum_i = 1^N w_i(mathbfx)   \n          textif  mathbfx - mathbfx_i neq 0 text for all  i  \n    u_i  textif  mathbfx - mathbfx_i = 0 text for some  i\nendcaseswhere mathbfx - mathbfx_i is the distance between mathbfx and  mathbfx_i, and w_i(mathbfx) = frac1mathbfx - mathbfx_i^P.This model is selected by passing a Shepard object to interpolate."
},

{
    "location": "methods/#Nearest-Neighbor-1",
    "page": "Supported methods",
    "title": "Nearest Neighbor",
    "category": "section",
    "text": "Nearest neighbor interpolation produces piecewise constant interpolations by returning the  data value of the nearest sample point.To use nearest neighbor interpolation, pass a NearestNeighbor object to  interpolate."
},

{
    "location": "api/#",
    "page": "API",
    "title": "API",
    "category": "page",
    "text": ""
},

{
    "location": "api/#API-1",
    "page": "API",
    "title": "API",
    "category": "section",
    "text": "Order   = [:function, :type]"
},

{
    "location": "api/#ScatteredInterpolation.interpolate",
    "page": "API",
    "title": "ScatteredInterpolation.interpolate",
    "category": "function",
    "text": "interpolate(method, points, samples; metric = Euclidean(), returnRBFmatrix = false; smooth = false)\n\nCreate an interpolation of the data in samples sampled at the locations defined in points based on the interpolation method method. metric is any of the metrics defined by the Distances package. The RBF matrix used for solving the weights can be returned with the boolean returnRBFmatrix. Note that this option is only valid for RadialBasisFunction interpolations.\n\npoints should be an nk matrix, where n is dimension of the sampled space and k is the number of points. This means that each column in the matrix defines one point.\n\nsamples is an km array, where k is the number of sampled points (same as for points) and m is the dimension of the sampled data.\n\nThe RadialBasisFunction interpolation supports the use of unique RBF functions and widths for each sampled point by supplying method with a vector of interpolation methods of length k.\n\nThe RadialBasisFunction interpolation also supports smoothing of the data points using ridge regression. All points can be smoothed equally supplying a scalar value, alternatively each point can be smoothed independently by supplying a vector of smoothing values. Note that it is no longer interpolating when using smoothing. \n\nThe returned ScatteredInterpolant object can be passed to evaluate to interpolate the data to new points.\n\n\n\n\n\n"
},

{
    "location": "api/#ScatteredInterpolation.evaluate",
    "page": "API",
    "title": "ScatteredInterpolation.evaluate",
    "category": "function",
    "text": "evaluate(itp, points)\n\nEvaluate an interpolation object itp at the locations defined in points.\n\npoints should be an nk matrix, where n is dimension of the sampled space and k is the number of points. This means that each column in the matrix defines one point.\n\n\n\n\n\n"
},

{
    "location": "api/#Functions-1",
    "page": "API",
    "title": "Functions",
    "category": "section",
    "text": "interpolate\nevaluate"
},

{
    "location": "api/#Types-1",
    "page": "API",
    "title": "Types",
    "category": "section",
    "text": ""
},

{
    "location": "api/#ScatteredInterpolation.Multiquadratic",
    "page": "API",
    "title": "ScatteredInterpolation.Multiquadratic",
    "category": "type",
    "text": "Multiquadratic(ɛ = 1)\n\nDefine a Multiquadratic Radial Basis Function\n\nϕ(r) = sqrt1 + (ɛr)^2\n\n\n\n\n\n"
},

{
    "location": "api/#ScatteredInterpolation.InverseMultiquadratic",
    "page": "API",
    "title": "ScatteredInterpolation.InverseMultiquadratic",
    "category": "type",
    "text": "InverseMultiquadratic(ɛ = 1)\n\nDefine an Inverse Multiquadratic Radial Basis Function\n\nϕ(r) = frac1sqrt1 + (ɛr)^2\n\n\n\n\n\n"
},

{
    "location": "api/#ScatteredInterpolation.Gaussian",
    "page": "API",
    "title": "ScatteredInterpolation.Gaussian",
    "category": "type",
    "text": "Gaussian(ɛ = 1)\n\nDefine a Gaussian Radial Basis Function\n\nϕ(r) = e^-(ɛr)^2\n\n\n\n\n\n"
},

{
    "location": "api/#ScatteredInterpolation.InverseQuadratic",
    "page": "API",
    "title": "ScatteredInterpolation.InverseQuadratic",
    "category": "type",
    "text": "InverseQuadratic(ɛ = 1)\n\nDefine an Inverse Quadratic Radial Basis Function\n\nϕ(r) = frac11 + (ɛr)^2\n\n\n\n\n\n"
},

{
    "location": "api/#ScatteredInterpolation.Polyharmonic",
    "page": "API",
    "title": "ScatteredInterpolation.Polyharmonic",
    "category": "type",
    "text": "Polyharmonic(k = 1)\n\nDefine a Polyharmonic Spline Radial Basis Function\n\nϕ(r) = r^k k = 1 3 5 \n\nϕ(r) = r^k ln(r) k = 2 4 6 \n\n\n\n\n\n"
},

{
    "location": "api/#ScatteredInterpolation.ThinPlate",
    "page": "API",
    "title": "ScatteredInterpolation.ThinPlate",
    "category": "function",
    "text": "ThinPlate()\n\nDefine a Thin Plate Spline Radial Basis Function\n\nϕ(r) = r^2 ln(r)\n\nThis is a shorthand for Polyharmonic(2).\n\n\n\n\n\n"
},

{
    "location": "api/#ScatteredInterpolation.GeneralizedMultiquadratic",
    "page": "API",
    "title": "ScatteredInterpolation.GeneralizedMultiquadratic",
    "category": "type",
    "text": "GeneralizedMultiquadratic(ɛ, β, degree)\n\nDefine a generalized Multiquadratic Radial Basis Function\n\nϕ(r) = (1 + (ɛ*r)^2)^β\n\nResults in a positive definite system for a \'degree\' of ⌈β⌉ or higher.\n\n\n\n\n\n"
},

{
    "location": "api/#ScatteredInterpolation.GeneralizedPolyharmonic",
    "page": "API",
    "title": "ScatteredInterpolation.GeneralizedPolyharmonic",
    "category": "type",
    "text": "GeneralizedPolyharmonic(k, degree)\n\nDefine a generalized Polyharmonic Radial Basis Function\n\nϕ(r) = r^k k = 1 3 5 \n\nϕ(r) = r^k ln(r) k = 2 4 6 \n\nResults in a positive definite system for a \'degree\' of ⌈k/2⌉ or higher for k = 1, 3, 5, ... and of exactly k + 1 for k = 2, 4, 6, ...\n\n\n\n\n\n"
},

{
    "location": "api/#Radial-Basis-Functions-1",
    "page": "API",
    "title": "Radial Basis Functions",
    "category": "section",
    "text": "Multiquadratic\nInverseMultiquadratic\nGaussian\nInverseQuadratic\nPolyharmonic\nThinPlate\nGeneralizedMultiquadratic\nGeneralizedPolyharmonic"
},

{
    "location": "api/#ScatteredInterpolation.Shepard",
    "page": "API",
    "title": "ScatteredInterpolation.Shepard",
    "category": "type",
    "text": "Shepard(P = 2)\n\nStandard Shepard interpolation with power parameter P.\n\n\n\n\n\n"
},

{
    "location": "api/#Inverse-Distance-Weighting-(Shepard)-1",
    "page": "API",
    "title": "Inverse Distance Weighting (Shepard)",
    "category": "section",
    "text": "Shepard"
},

{
    "location": "api/#ScatteredInterpolation.NearestNeighbor",
    "page": "API",
    "title": "ScatteredInterpolation.NearestNeighbor",
    "category": "type",
    "text": "NearestNeigbor()\n\nNearest neighbor interpolation.\n\n\n\n\n\n"
},

{
    "location": "api/#Nearest-Neighbor-1",
    "page": "API",
    "title": "Nearest Neighbor",
    "category": "section",
    "text": "NearestNeighbor"
},

]}
