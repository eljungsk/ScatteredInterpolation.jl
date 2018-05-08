var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#ScatteredInterpolation.jl-1",
    "page": "Home",
    "title": "ScatteredInterpolation.jl",
    "category": "section",
    "text": "Interpolation of scattered data in JuliaScatteredInterpolation.jl interpolates scattered data in arbitrary dimensions. Pages = [\n    \"index.md\",\n    \"methods.md\",\n    \"api.md\"\n]\nDepth = 2"
},

{
    "location": "index.html#Example-Usage-1",
    "page": "Home",
    "title": "Example Usage",
    "category": "section",
    "text": "using ScatteredInterpolationInterpolate data defined on the vertices and in the center of a square using a multiquadratic radial basis function:samples = [0.0; 0.5; 0.5; 0.5; 1.0];\npoints = [0.0 0.0; 0.0 1.0; 1.0 0.0; 0.5 0.5; 1.0 1.0]\';\nitp = interpolate(Multiquadratic(), points, samples)\ninterpolated = evaluate(itp, [0.6; 0.6])If we instead want to use nearest neighbor interpolation, we can runitp = interpolate(NearestNeighbor(), points, samples)\ninterpolated = evaluate(itp, [0.6; 0.6])"
},

{
    "location": "index.html#Gridding-of-data-1",
    "page": "Home",
    "title": "Gridding of data",
    "category": "section",
    "text": "A common use case for scattered interpolation is gridding of data, i.e. interpolation of scattered data to a grid. Using the same data as above, we can interpolate it to a 5x5 gridn = 5\nx = linspace(0, 1, n)\ny = linspace(0, 1, n)\nX = repmat(x, 1, n)[:]\nY = repmat(y\', n, 1)[:]\ngridPoints = [X Y]\'\n\nitp = interpolate(Multiquadratic(), points, samples)\ninterpolated = evaluate(itp, gridPoints)\ngridded = reshape(interpolated, n, n)"
},

{
    "location": "methods.html#",
    "page": "Supported methods",
    "title": "Supported methods",
    "category": "page",
    "text": ""
},

{
    "location": "methods.html#Supported-methods-1",
    "page": "Supported methods",
    "title": "Supported methods",
    "category": "section",
    "text": "Currently, three different interpolation methods are available; Radial Basis Functions, Inverse Distance Weighting and Nearest Neighbor."
},

{
    "location": "methods.html#Radial-Basis-Functions-1",
    "page": "Supported methods",
    "title": "Radial Basis Functions",
    "category": "section",
    "text": "For radial basis function interpolation, the interpolated value at some point  mathbfx is given byu(mathbfx) = displaystyle sum_i = 1^N w_i phi(mathbfx - mathbfx_i)where mathbfx - mathbfx_i is the distance between mathbfx and  mathbfx_i, and phi(r) is one of the basis functions defined below.To use radial basis function interpolation, pass one of the available basis functions as  method to interpolate."
},

{
    "location": "methods.html#Available-basis-functions-1",
    "page": "Supported methods",
    "title": "Available basis functions",
    "category": "section",
    "text": "Multiquadratic\n(r) = sqrt1 + (r)^2\nInverseMultiquadratic\n(r) = frac1sqrt1 + (r)^2\nGaussian\n(r) = e^-(r)^2\nInverseQuadratic\n(r) = frac11 + (r)^2\nPolyharmonic spline\n(r) = \nbegincases\n    beginalign*\n        r^k                       k = 1 3 5  \n        r^k mathrmln(r)        k = 2 4 6 \n    endalign*\nendcases\nThinPlate spline\nA thin plate spline is the special case k = 2 of the polyharmonic splines. ThinPlate() is a shorthand for Polyharmonic(2)."
},

{
    "location": "methods.html#Inverse-Distance-Weighting-1",
    "page": "Supported methods",
    "title": "Inverse Distance Weighting",
    "category": "section",
    "text": "Also called Shepard interpolation, the basic version computes the interpolated value at some point mathbfx byu(mathbfx) = \nbegincases \n    fracdisplaystyle sum_i = 1^N w_i(mathbfx) u_i   \n         displaystyle sum_i = 1^N w_i(mathbfx)   \n          textif  mathbfx - mathbfx_i neq 0 text for all  i  \n    u_i  textif  mathbfx - mathbfx_i = 0 text for some  i\nendcaseswhere mathbfx - mathbfx_i is the distance between mathbfx and  mathbfx_i, and w_i(mathbfx) = frac1mathbfx - mathbfx_i^P.This model is selected by passing a Shepard object to interpolate."
},

{
    "location": "methods.html#Nearest-Neighbor-1",
    "page": "Supported methods",
    "title": "Nearest Neighbor",
    "category": "section",
    "text": "Nearest neighbor interpolation produces piecewise constant interpolations by returning the  data value of the nearest sample point.To use nearest neighbor interpolation, pass a NearestNeighbor object to  interpolate."
},

{
    "location": "api.html#",
    "page": "API",
    "title": "API",
    "category": "page",
    "text": ""
},

{
    "location": "api.html#API-1",
    "page": "API",
    "title": "API",
    "category": "section",
    "text": "Order   = [:function, :type]"
},

{
    "location": "api.html#ScatteredInterpolation.interpolate",
    "page": "API",
    "title": "ScatteredInterpolation.interpolate",
    "category": "function",
    "text": "interpolate(method, points, samples; metric = Euclidean(), returnRBFmatrix = false)\n\nCreate an interpolation of the data in samples sampled at the locations defined in points based on the interpolation method method. metric is any of the metrics defined by the Distances package. The RBF matrix used for solving the weights can be returned with the boolean returnRBFmatrix. Note that this option is only valid for RadialBasisFunction interpolations.\n\npoints should be an nk matrix, where n is dimension of the sampled space and k is the number of points. This means that each column in the matrix defines one point.\n\nsamples is an km array, where k is the number of sampled points (same as for points) and m is the dimension of the sampled data.\n\nThe returned ScatteredInterpolant object can be passed to evaluate to interpolate the data to new points.\n\n\n\n"
},

{
    "location": "api.html#ScatteredInterpolation.evaluate",
    "page": "API",
    "title": "ScatteredInterpolation.evaluate",
    "category": "function",
    "text": "evaluate(itp, points)\n\nEvaluate an interpolation object itp at the locations defined in points.\n\npoints should be an nk matrix, where n is dimension of the sampled space and k is the number of points. This means that each column in the matrix defines one point.\n\n\n\n"
},

{
    "location": "api.html#Functions-1",
    "page": "API",
    "title": "Functions",
    "category": "section",
    "text": "interpolate\nevaluate"
},

{
    "location": "api.html#Types-1",
    "page": "API",
    "title": "Types",
    "category": "section",
    "text": ""
},

{
    "location": "api.html#ScatteredInterpolation.Multiquadratic",
    "page": "API",
    "title": "ScatteredInterpolation.Multiquadratic",
    "category": "type",
    "text": "Multiquadratic(ɛ = 1)\n\nDefine a Multiquadratic Radial Basis Function\n\n(r) = sqrt1 + (r)^2\n\n\n\n"
},

{
    "location": "api.html#ScatteredInterpolation.InverseMultiquadratic",
    "page": "API",
    "title": "ScatteredInterpolation.InverseMultiquadratic",
    "category": "type",
    "text": "InverseMultiquadratic(ɛ = 1)\n\nDefine an Inverse Multiquadratic Radial Basis Function\n\n(r) = frac1sqrt1 + (r)^2\n\n\n\n"
},

{
    "location": "api.html#ScatteredInterpolation.Gaussian",
    "page": "API",
    "title": "ScatteredInterpolation.Gaussian",
    "category": "type",
    "text": "Gaussian(ɛ = 1)\n\nDefine a Gaussian Radial Basis Function\n\n(r) = e^-(r)^2\n\n\n\n"
},

{
    "location": "api.html#ScatteredInterpolation.InverseQuadratic",
    "page": "API",
    "title": "ScatteredInterpolation.InverseQuadratic",
    "category": "type",
    "text": "InverseQuadratic(ɛ = 1)\n\nDefine an Inverse Quadratic Radial Basis Function\n\n(r) = frac11 + (r)^2\n\n\n\n"
},

{
    "location": "api.html#ScatteredInterpolation.Polyharmonic",
    "page": "API",
    "title": "ScatteredInterpolation.Polyharmonic",
    "category": "type",
    "text": "Polyharmonic(k = 1)\n\nDefine a Polyharmonic Spline Radial Basis Function\n\n(r) = r^k k = 1 3 5 \n\n(r) = r^k ln(r) k = 2 4 6 \n\n\n\n"
},

{
    "location": "api.html#ScatteredInterpolation.ThinPlate",
    "page": "API",
    "title": "ScatteredInterpolation.ThinPlate",
    "category": "type",
    "text": "ThinPlate()\n\nDefine a Thin Plate Spline Radial Basis Function\n\n(r) = r^2 ln(r)\n\nThis is a shorthand for Polyharmonic(2).\n\n\n\n"
},

{
    "location": "api.html#Radial-Basis-Functions-1",
    "page": "API",
    "title": "Radial Basis Functions",
    "category": "section",
    "text": "Multiquadratic\nInverseMultiquadratic\nGaussian\nInverseQuadratic\nPolyharmonic\nThinPlate"
},

{
    "location": "api.html#ScatteredInterpolation.Shepard",
    "page": "API",
    "title": "ScatteredInterpolation.Shepard",
    "category": "type",
    "text": "Shepard(P = 2)\n\nStandard Shepard interpolation with power parameter P.\n\n\n\n"
},

{
    "location": "api.html#Inverse-Distance-Weighting-(Shepard)-1",
    "page": "API",
    "title": "Inverse Distance Weighting (Shepard)",
    "category": "section",
    "text": "Shepard"
},

{
    "location": "api.html#ScatteredInterpolation.NearestNeighbor",
    "page": "API",
    "title": "ScatteredInterpolation.NearestNeighbor",
    "category": "type",
    "text": "NearestNeigbor()\n\nNearest neighbor interpolation.\n\n\n\n"
},

{
    "location": "api.html#Nearest-Neighbor-1",
    "page": "API",
    "title": "Nearest Neighbor",
    "category": "section",
    "text": "NearestNeighbor"
},

]}
