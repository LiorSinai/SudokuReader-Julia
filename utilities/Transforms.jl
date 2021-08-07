#=
Lior Sinai, 25 July 2021
=#

module Transforms

using ImageTransformations, CoordinateTransformations
using StaticArrays
using Images

export fit_rectangle, fit_parallelogram, 
    getPerspectiveTransform, fourPointTransform, 
    imwarp, apply_homography
    

function fit_rectangle(points::AbstractVector)
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


function fit_parallelogram(points::AbstractVector) 
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
getPerspectiveTransform(source::AbstractArray, destination::AbstractArray)

Compute the elements of the matrix for projective transformation from source to destination.
Source and destination must have the same number of points.

Transformation is:
| u |   | c11 c12 c13 | | x |
| v | = | c21 c22 c23 |·| y |
| w |   | c31 c32   1 | | 1 |

So that u' = u/w, v'=v/w where w = 1/(focal length) of the pinhole camera. 

Sovling for u and v:
    (c31·x + c32·y + 1)u = c11·x + c12·y + c13
    (c31·x + c32·y + 1)v = c21·x + c22·y + c23
Similarly for v and rearrange into the following matrix:
    [u1; u2; u3; u4; v1; v2; v3; v4] = A·[c11; c12; c13; c21; c22; c23; c31; c32]
    B = A ⋅ X
"""
function getPerspectiveMatrix(source::AbstractArray, destination::AbstractArray)
    if (length(source) != length(destination))
        error("$(length(source))!=$(length(destination)). Source must have the same number of points as destination")
    elseif length(source) < 4
        error("length(source)=$(length(source)). Require at least 4 points")
    end
    indx, indy = 1, 2
    n = length(source)
    A = zeros(2n, 8)
    B = zeros(2n)
    for i in 1:n
        A[i, 1] = source[i][indx]
        A[i, 2] = source[i][indy]
        A[i, 3] = 1
        A[i, 7] = -source[i][indx] * destination[i][indx]
        A[i, 8] = -source[i][indy] * destination[i][indx]
        B[i] = destination[i][indx]
    end
    for i in 1:n
        A[i + n, 4] = source[i][indx]
        A[i + n, 5] = source[i][indy]
        A[i + n, 6] = 1
        A[i + n, 7] = -source[i][indx] * destination[i][indy]
        A[i + n, 8] = -source[i][indy] * destination[i][indy]
        B[i + n] = destination[i][indy]
    end
    M = inv(A) * B
    M = [
        M[1] M[2] M[3];
        M[4] M[5] M[6]'
        M[7] M[8] 1
    ]
    M
end


function order_points(corners)
	# order points: top-left, top-right, bottom-right, bottom-left
	rect = zeros(typeof(corners[1]), 4)
	# the top-left point will have the smallest sum, whereas the bottom-right point will have the largest sum
	s = [point[1] + point[2] for point in corners]
	rect[1] = corners[argmin(s)]
	rect[3] = corners[argmax(s)]
	# now, compute the difference between the points, the top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = [point[2] - point[1] for point in corners]
	rect[2] = corners[argmin(diff)]
	rect[4] = corners[argmax(diff)]
	# return the ordered coordinates
	return rect
end


function fourPointTransform(image::AbstractArray, corners::AbstractVector)
    parallelogram = order_points(corners)
    (topleft, topright, bottomright, bottomleft) = parallelogram

	# compute the width of the new image
	widthA = (bottomright[1] - bottomleft[1])^2 + (bottomright[2] - bottomleft[2])^2
	widthB = (topright[1] - topleft[1])^2 + (topright[2] - topleft[2])^2
	maxWidth = Int(round(sqrt(max(widthA, widthB))))

    # compute the height of the new image
	heightA = (topright[1] - bottomright[1]) ^2 + (topright[2] - bottomright[2]) ^ 2
	heightB = (topleft[1] - bottomleft[1]) ^ 2 + (topleft[2] - bottomleft[2]) ^ 2
	maxHeight = Int(round(sqrt(max(heightA, heightB))))	

    destination = [
        CartesianIndex(1, 1),
        CartesianIndex(maxWidth, 1),
        CartesianIndex(maxWidth, maxHeight),
        CartesianIndex(1, maxHeight)
    ]

    M = getPerspectiveMatrix(parallelogram, destination)
    invM = inv(M)
    extend1(v) = [v[1], v[2], 1]
    perspective_transform = PerspectiveMap() ∘ LinearMap(invM) ∘ extend1

    # function perspective_transform(v::SVector) # slower code
    #     U = invM * [v[1], v[2], 1]
    #     scale = 1/U[3]
    #     [U[1] * scale, U[2] * scale]
    # end

    warped = warp(image, perspective_transform, (1:maxWidth, 1:maxHeight))
    warped, invM
end


function rgb_to_float(pixel::AbstractRGB)
    Float32.([red(pixel), green(pixel), blue(pixel)])
end

"""
apply_homography(point, M::Matrix)
"""
function apply_homography(point, M::Matrix)
    U = M * [point[1]; point[2]; 1]
    scale = 1/U[3]
    [U[1] * scale, U[2] * scale]
end


function get_color(image::AbstractArray{T}, point) where T
    ind = CartesianIndex(Int(floor(point[1])), Int(floor(point[2])))
    image[ind]
end


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

end # Transforms