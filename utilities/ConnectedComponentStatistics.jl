struct ConnectedComponentStatistics
    left::Int
    top::Int
    right::Int
    bottom::Int
    area::Int
    centroid::Tuple{Float64, Float64}
end

"""
Get statistics for the output of Images.label_components
"""
function calc_connected_component_statistics(labels::AbstractArray, label::Int)
    height, width = size(labels)

    left = width
    top = height
    right = 0
    bottom = 0
    area = 0
    Cx, Cy = 0.0, 0.0

    for i in 1:height
        for j in 1:width
            if labels[i,j] == label
                area += 1
                left = min(left, j)
                top = min(top, i)
                right = max(right, j)
                bottom = max(bottom, i)
                Cx += 1.0
                Cy += 1.0
            end
        end
    end
    ConnectedComponentStatistics(left, top, right, bottom, area, (Cx/area, Cy/area))
end

