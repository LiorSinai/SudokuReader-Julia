
module DigitExtration

include("../utilities/ConnectedComponents.jl")
using .ConnectedComponents
using Flux: softmax, batch, unsqueeze
using Images: imresize


export read_digits, extract_digit,
    detect_in_centre, make_circle_kernel,
    prediction


function read_digits(
    image::AbstractArray,
    model; 
    offset_ratio=0.1,
    radius_ratio::Float64=0.25, 
    detection_threshold::Float64=0.02, 
    )
    height, width = size(image)
    step_i = ceil(Int, height / 9)
    step_j = ceil(Int, width / 9)
    offset_i = round(Int, offset_ratio * step_i)
    offset_j = round(Int, offset_ratio * step_j)

    grid = zeros(Int, (9, 9))
    centres =  [(-1.0, -1.0) for i in 1:9, j in 1:9]
    probs = zeros(Float32, (9, 9))

    for (i_grid, i_img) in enumerate(1:step_i:height)
        for (j_grid, j_img) in enumerate(1:step_j:width)
            prev_i = max(1, i_img - offset_i)
            prev_j = max(1, j_img - offset_j)
            next_i = min(i_img + step_i + offset_i, height)
            next_j = min(j_img + step_j + offset_j, width)
            RoI = image[prev_i:next_i, prev_j:next_j]
            if detect_in_centre(RoI)
                centre, digit = extract_digit(RoI, radius_ratio=radius_ratio, threshold=detection_threshold)
                ŷ, prob = prediction(model, digit)
                grid[i_grid, j_grid] = ŷ
                
                centre = (centre[1] + prev_i, centre[2] + prev_j)
                probs[i_grid, j_grid] = prob
            else
                centre = (prev_i + step_i/2, prev_j + step_j/2)
            end
            centres[i_grid, j_grid] = centre
        end
    end
    grid, centres, probs
end


function extract_digit(image::AbstractArray; radius_ratio::Float64=0.25, threshold::Float64=0.02)
    labels, statistics = get_connected_components(image) 
    height, width = size(image)
    for i in 1:length(statistics)
        image_label = copy(image)
        image_label[labels .!= i] .= 0
        if detect_in_centre(image_label, radius_ratio=radius_ratio, threshold=threshold)
            stats = statistics[i]
            width_label = abs(stats.right - stats.left)
            height_label = abs(stats.bottom - stats.top)
            length_  = max(width_label, height_label)

            # note: the centroid is not a good chocie for a visual centre 
            centre = (stats.top + Int(round(height_label/2)), stats.left + Int(round(width_label/2)))

            # make square and pad
            top = max(1, floor(Int, centre[1] - length_/2))
            left = max(1, floor(Int,centre[2] - length_/2))
            bottom = min(height, ceil(Int, centre[1] + length_/2))
            right = min(width, ceil(Int, centre[2] + length_/2))
            return centre, image_label[top:bottom, left:right]
        end
    end
    (height/2, width/2), image
end


"""
detect_in_centre(image::AbstractArray; [radius_ratio], [threshold])

Detect an object in a region of interest. This is done by convolving it with a circle.
"""
function detect_in_centre(image::AbstractArray; radius_ratio::Float64=0.25, threshold::Float64=0.02)
    height, width = size(image)
    radius = min(height, width) * radius_ratio
    kernel = make_circle_kernel(height, width, radius)
    conv = kernel .* image
    detected = sum(conv .!= 0)/(height * width) > threshold
    detected
end


function make_circle_kernel(height::Int, width::Int, radius::Float64)
    # backward algorithm
    kernel = zeros((height, width))
    centre = (width/2, height/2)
    for i in 1:height
        for j in 1: width
            z = radius^2 - (j - centre[1])^2 - (i - centre[2])^2
            if z > 0
                kernel[CartesianIndex(i, j)] = 1
            end
        end
    end
    kernel
end

make_circle_kernel(height::Int, width::Int, radius::Int) = make_circle_kernel(height, width, Float64(radius))


function pad_image(image::AbstractArray{T}; pad_ratio=0.1) where T
    height, width = size(image)
    pad = floor(Int, pad_ratio * max(height, width))
    imnew = zeros(T, (height + 2pad, width + 2pad))
    imnew[(pad + 1):(pad + height), (pad + 1):(pad + width)] = image
    imnew
end


function prediction(model, image::AbstractArray, pad_ratio=0.1)
    image = pad_image(image, pad_ratio=pad_ratio)
    image = imresize(image, (28, 28))
    x = batch([unsqueeze(Float32.(image), 3)])
    logits = model(x)
    probabilites = softmax(logits)
    idx = argmax(probabilites)
    ŷ = idx[1] - 1
    ŷ, probabilites[idx]
end


end # module DigitExtration