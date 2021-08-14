#= 
Lior Sinai, 25 July 20210
=#

module GridDetection

using Images
using ImageFiltering
using ImageBinarization

include("../utilities/Contours.jl");
include("../utilities/invert_image.jl");
include("../utilities/Transforms.jl");

using .Contours
using .Transforms

export detect_grid, preprocess, construct_grid


function detect_grid(image::AbstractArray; kwargs...)
    blackwhite = preprocess(image; kwargs...)

    # assumption: grid is the largest contour in the image
    contours = find_contours(blackwhite, external_only=true)
    idx_max = argmax(map(calc_area_contour, contours))
    quad = fit_quad(contours[idx_max])
    
    blackwhite, quad
end


function preprocess(
    image::AbstractArray; 
    max_size=1024, 
    blur_window_size=5, σ=1, 
    threshold_window_size=15, threshold_percentage=7
    )
    gray = Gray.(image)

    # resize
    ratio = max_size/size(gray, argmax(size(gray)))
    if ratio < 1
        gray = imresize(gray, ratio=ratio)
    end
    
    # blur
    kernel = Kernel.gaussian((σ, σ), (blur_window_size, blur_window_size))
    gray = imfilter(gray, kernel)

    #binarize
    blackwhite = binarize(gray, AdaptiveThreshold(window_size=threshold_window_size, percentage=threshold_percentage))
    blackwhite = invert_image(blackwhite)

    blackwhite
end


function construct_grid(height::Int, width::Int; nblocks::Int=3)
    grid = []
    step_i = height/nblocks
    step_j = width/nblocks
    for i in 0:nblocks
        push!(grid, [(step_i * i, 1), (step_i * i, width)])
    end
    for j in 0:nblocks
        push!(grid, [(1, step_j * j), (height, step_j * j)])
    end
    grid
end


end # module GridDetection