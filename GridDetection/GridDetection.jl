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

export detect_grid, invert_image


function detect_grid(image::AbstractArray)
    blackwhite = preprocess(image)

    # assumption: grid is the largest contour in the image
    contours = find_contours(blackwhite, external_only=true)
    idx_max = argmax(map(calc_area_contour, contours))
    par = fit_parallelogram(contours[idx_max])
    
    blackwhite, par
end


function invert_image(image)
    image_inv = Gray.(image)
    height, width = size(image)
    for i in 1:height
        for j in 1:width
            image_inv[i, j] = 1 - image_inv[i, j]
        end
    end
    return image_inv
end


function preprocess(image::AbstractArray; max_size=1024, blur_window_size=5, σ=1, threshold_window_size=25)
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
    blackwhite = binarize(gray, AdaptiveThreshold(window_size=threshold_window_size))
    blackwhite = invert_image(blackwhite)

    blackwhite
end

end # module GridDetection