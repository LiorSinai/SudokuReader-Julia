#=
Lior Sinai, 25 July 2021
=#

module PerspectiveTransformations

using ImageTransformations, CoordinateTransformations
using StaticArrays
using Images

export fit_rectangle, fit_quad
export get_perspective_matrix, four_point_transform, perspective_transform
    
"""
    fit_rectangle(points)

Fit a tight rectangle which encompasses all points.
Corners are: (top-left, top-right, bottom-right, bottom-left).
"""
function fit_rectangle(points::Vector{<:CartesianIndex})
    # return corners in top-left, top-right, bottom-right, bottom-left
    min_x, max_x, min_y, max_y = typemax(Int), typemin(Int), typemax(Int), typemin(Int)
    for point in points
        min_x = min(min_x, point[1])
        max_x = max(max_x, point[1])
        min_y = min(min_y, point[2])
        max_y = max(max_y, point[2])
    end
    
    corners = [
        CartesianIndex(min_x, min_y),
        CartesianIndex(max_x, min_y),
        CartesianIndex(max_x, max_y),
        CartesianIndex(min_x, max_y),
    ]

    return corners
end

"""
    fit_quad(points)

Fit a tight quadrilateral which encompasses all points.
Corners are: (top-left, top-right, bottom-right, bottom-left).
"""
function fit_quad(points::AbstractVector) 
    rect = fit_rectangle(points)

    corners = copy(rect)
    distances = [Inf, Inf, Inf, Inf]

    for point in points
        for i in 1:4
            d = abs(point[1] - rect[i][1]) + abs(point[2] - rect[i][2])
            if d < distances[i]
                corners[i] = point
                distances[i] = d
            end
        end
    end
    return corners
end

"""
    get_perspective_matrix(source::AbstractArray, destination::AbstractArray)

Compute the elements of the matrix for projective transformation from source to destination.
Source and destination must have the same number of points.

The transformation is:
```
| u' |   | c11 c12 c13 | | x |
| v' | = | c21 c22 c23 |·| y |
| w  |   | c31 c32   1 | | 1 |
```

The warped pixel points `u` and `v` are normalized by the focal length `w = 1/(focal length)` of the pinhole camera:
```
u = u'/w
v = v'/w
```

Substituting for `u'` and `v'`:
```
      u = (c11·x + c12·y + c13)/(c31·x + c32·y + 1)
    ∴ u = c11·x + c12·y + c13 - c31·x·u - c32·y·u
      v = (c21·x + c22·y + c23)/(c31·x + c32·y + 1)
    ∴ v =  c21·x + c22·y + c23 - c31·x·v - c32·y·v
```

Rearrange into a matrix for all points:
```
| u1 |   | x1 y1 1 0  0  0 -x1u1 -y1u1 ||c11|
| u2 |   | x2 y2 1 0  0  0 -x2u2 -y2u2 ||c12|
| u3 | = | x3 y3 1 0  0  0 -x3u3 -y3u3 ||c13|
| u4 |   | x4 y4 1 0  0  0 -x4u4 -y4u4 ||c21|
| v1 |   | 0  0  0 x1 y1 1 -x1v1 -y1v1 ||c22|
| v2 |   | 0  0  0 x2 y2 1 -x2v2 -y1v2 ||c23|
| v3 |   | 0  0  0 x3 y3 1 -x3v3 -y3v3 ||c31|
| v3 |   | 0  0  0 x4 y4 1 -x4u4 -y4v4 ||c32|
U = X ⋅ M
```
Solve for `M`.
"""
function get_perspective_matrix(source::AbstractArray, destination::AbstractArray)
    if (length(source) != length(destination))
        error("$(length(source))!=$(length(destination)). Source must have the same number of points as destination")
    elseif length(source) < 4
        error("length(source)=$(length(source)). Require at least 4 points")
    end
    indx, indy = 1, 2
    n = length(source)
    X = zeros(2n, 8)
    U = zeros(2n)
    for i in 1:n
        X[i, 1] = source[i][indx]
        X[i, 2] = source[i][indy]
        X[i, 3] = 1
        X[i, 7] = -source[i][indx] * destination[i][indx]
        X[i, 8] = -source[i][indy] * destination[i][indx]
        U[i] = destination[i][indx]
    end
    for i in 1:n
        X[i + n, 4] = source[i][indx]
        X[i + n, 5] = source[i][indy]
        X[i + n, 6] = 1
        X[i + n, 7] = -source[i][indx] * destination[i][indy]
        X[i + n, 8] = -source[i][indy] * destination[i][indy]
        U[i + n] = destination[i][indy]
    end
    M = inv(X) * U
    M = [
        M[1] M[2] M[3];
        M[4] M[5] M[6];
        M[7] M[8] 1;
    ]
    M
end

function order_points(corners)
	# points: top-left, top-right, bottom-right, bottom-left
	rect = zeros(typeof(corners[1]), 4)
	# the top-left point will have the smallest sum, whereas the bottom-right point will have the largest sum
	s = [point[1] + point[2] for point in corners]
	rect[1] = corners[argmin(s)]
	rect[3] = corners[argmax(s)]
	# compute the difference between the points. The top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = [point[2] - point[1] for point in corners]
	rect[2] = corners[argmin(diff)]
	rect[4] = corners[argmax(diff)]
	rect
end

"""
    four_point_transform(image, corners)

Map an image from a quadrilateral defined by 4 `corners` to a rectangle.
"""
function four_point_transform(image::AbstractArray, corners::AbstractVector)
    quad = order_points(corners)
    rect = fit_rectangle(corners)
    destination = [CartesianIndex(point[1] - rect[1][1] + 1, point[2] - rect[1][2] + 1) for point in rect]
    maxWidth = destination[2][1] - destination[1][1] 
    maxHeight = destination[3][2] - destination[2][2] 

    M = get_perspective_matrix(quad, destination)
    invM = inv(M)
    transform = perspective_transform(invM)

    warped = warp(image, transform, (1:maxWidth, 1:maxHeight))
    warped, invM
end

extend1(v) = [v[1], v[2], 1]
perspective_transform(M::Matrix) = PerspectiveMap() ∘ LinearMap(M) ∘ extend1

"""
    imwarp(image, invM, dest_size)

This function is only for illustrative purposes only.
It is a slower and less accurate version of ImageTransformations.warp.
It implements a backwards transformation algorithm for a homography transformation matrix. 
Colors are approximated as the pixel the inverse transform lands in.
No further interpolation is done.
"""
function imwarp(image::AbstractArray{T}, invM::Matrix, dest_size::Tuple{Int, Int}) where T
    warped = zeros(T, dest_size...)

    height, width = dest_size
    for i in 1:height
        for j in 1:width
            ind = apply_homography((i, j), invM)
            warped[CartesianIndex(i, j)] = get_color(image, ind)
        end
    end          
    warped
end

function apply_homography(point::Tuple{Int, Int}, M::Matrix)
    U = M * [point[1]; point[2]; 1]
    scale = 1/U[3]
    [U[1] * scale, U[2] * scale]
end

function rgb_to_float(pixel::AbstractRGB)
    Float32.([red(pixel), green(pixel), blue(pixel)])
end

function get_color(image::AbstractArray{T}, point) where T
    ind = CartesianIndex(Int(floor(point[1])), Int(floor(point[2])))
    image[ind]
end

end # Transforms