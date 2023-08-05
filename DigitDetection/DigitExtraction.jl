module DigitExtraction

include("ConnectedComponentStatistics.jl")
using Images: imresize, label_components
using ImageBinarization

export extract_digits_from_grid, extract_digit,
    detect_in_centre, make_circle_kernel

GRID_SIZE = (9, 9)
MODEL_SIZE = (28, 28)

"""
    extract_digits_from_grid(image, model; offset_ratio=0.1, detect_options...)

Extract multiple digits from a 9×9 grid using a model.
`predictor` should take an input image of a digit and return a tuple of `(label, confidence)`.
`offset_ratio` is extra overlap from neighbouring cells to include in each cell.
This is recommended if the cells are not exact squares.
See `detect_in_centre` for `detect_options`.
"""
function extract_digits_from_grid(
    image::AbstractArray,
    predictor; 
    offset_ratio::Float64=0.1,
    detect_options...
    )
    height, width = size(image)
    step_i = ceil(Int, height / GRID_SIZE[1])
    step_j = ceil(Int, width / GRID_SIZE[2])
    offset_i = round(Int, offset_ratio * step_i)
    offset_j = round(Int, offset_ratio * step_j)

    grid = zeros(Int, GRID_SIZE)
    centres =  [(-1.0, -1.0) for i in 1:GRID_SIZE[1], j in 1:GRID_SIZE[2]]
    confidences = zeros(Float32, GRID_SIZE)

    for (i_grid, i_img) in enumerate(1:step_i:height)
        for (j_grid, j_img) in enumerate(1:step_j:width)
            prev_i = max(1, i_img - offset_i)
            prev_j = max(1, j_img - offset_j)
            next_i = min(i_img + step_i + offset_i, height)
            next_j = min(j_img + step_j + offset_j, width)
            RoI = image[prev_i:next_i, prev_j:next_j]
            if detect_in_centre(RoI; detect_options...)
                digit, centre = extract_digit(RoI; detect_options...)
                label, confidence = predictor(digit)
                grid[i_grid, j_grid] = label
                confidences[i_grid, j_grid] = confidence
            else
                centre = (step_i/2, step_j/2)
            end
            centres[i_grid, j_grid] = (centre[1] + prev_i, centre[2] + prev_j)
        end
    end
    grid, centres, confidences
end

"""
    extract_digt(image; detect_options...)

Extract a single digit from an image.
Returns an image centred on the digit with all other components removed.
Returns the original image if it does not detect anything, 
See `detect_in_centre` for `detect_options`.
"""
function extract_digit(image_in::AbstractArray; detect_options...)
    image = copy(image_in)
    # have to binarize again because of warping
    image = binarize(image, Otsu()) # global binarization algorithm
    # check each unique connected component in the image
    labels = label_components(image) 
    for i in 1:length(unique(labels))
        image_label = copy(image)
        image_label[labels .!= i] .= 0
        if detect_in_centre(image_label; detect_options...)
            return extract_component_in_square(image_label)
        end
    end
    height, width = size(image)
    image, (height/2, width/2)
end

function extract_component_in_square(image_label::AbstractArray)
    stats = calc_connected_component_statistics(image_label .> 0, 1)
    width_label = abs(stats.right - stats.left)
    height_label = abs(stats.bottom - stats.top)
    length_  = max(width_label, height_label)

    # note: the centroid is not a good choice for a visual centre
    centre = (stats.top + Int(round(height_label/2)), stats.left + Int(round(width_label/2)))

    # make square
    height, width = size(image_label)
    top = max(1, floor(Int, centre[1] - length_/2))
    left = max(1, floor(Int,centre[2] - length_/2))
    bottom = min(height, ceil(Int, centre[1] + length_/2))
    right = min(width, ceil(Int, centre[2] + length_/2))
    image_label[top:bottom, left:right], centre
end

"""
    detect_in_centre(image::AbstractArray; radius_ratio=0.25, threshold=0.10)

Detect an object in a region of interest. This is done by convolving it with a circle.
Returns `true` if the area overlap with the circle is greater than `threshold`.
"""
function detect_in_centre(image::AbstractArray; radius_ratio::Float64=0.25, threshold::Float64=0.10)
    height, width = size(image)
    radius = min(height, width) * radius_ratio
    kernel = make_circle_kernel(height, width, radius)
    conv = kernel .* image
    detected = sum(conv .!= 0)/(pi * radius * radius) > threshold
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

## functions for a model trained on MNIST style images

function prepare_digit_for_model(digit::AbstractArray; pad_ratio=0.15)
    x = digit
    x = Float32.(x)
    x = pad_image(x; pad_ratio=pad_ratio)
    x = imresize(x, MODEL_SIZE)
    x = reshape(x, MODEL_SIZE[1], MODEL_SIZE[2], 1, 1)
    x
end

function pad_image(image::AbstractArray{T}; pad_ratio=0.15) where T
    height, width = size(image)
    pad = floor(Int, pad_ratio * max(height, width))
    imnew = zeros(T, (height + 2pad, width + 2pad))
    imnew[(pad + 1):(pad + height), (pad + 1):(pad + width)] = image
    imnew
end

end # module DigitExtraction