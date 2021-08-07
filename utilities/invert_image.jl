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