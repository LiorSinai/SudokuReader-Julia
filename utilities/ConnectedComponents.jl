#=
Connected Components

Lior Sinai, 30 July 2021

TODO: consider the OpenCV algorithm: "Optimizing two-pass connected-component labeling algorithms. Pattern Analysis 
and Applications" by Kesheng Wu, Ekow Otoo, and Kenji Suzuki (2009). THis is a 2-pass algorithm which accesses the matrix
in a regular way. The current algorithm uses a 1-pass irregular access which may slow progress.

=#


module ConnectedComponents

export get_connected_components,
    ConnectedComponentStatistics


struct ConnectedComponentStatistics
    left::Int
    top::Int
    right::Int
    bottom::Int
    area::Int
    centroid::Tuple{Float64, Float64}
end


function get_connected_components(image::AbstractArray)
    labels = -ones(size(image))
    statistics = ConnectedComponentStatistics[]
    background_color = zero(typeof(first(image)))

    height, width = size(image)
    current_label = 0
    for i in 1:height
        for j in 1:width
            if labels[i, j] == -1
                if image[i, j] == background_color
                    labels[i, j] = 0
                else
                    current_label += 1
                    labels, comp_stats = flood_label(labels, image, current_label, CartesianIndex(i, j))
                    push!(statistics, comp_stats)
                end
            end
        end
    end

    labels, statistics
end


function update_statistics(stats_old, i, j)
    left = min(stats_old.left, j)
    top = min(stats_old.top, i)
    right = max(stats_old.right, j)
    bottom = max(stats_old.bottom, i)
    area = stats_old.area + 1
    centroid = (stats_old.centroid[1] + j, stats_old.centroid[2] + i)
    ConnectedComponentStatistics(left, top, right, bottom, area, centroid)
end


function flood_label(labels_in::AbstractArray, image::AbstractArray, label::Int, seed)
    labels = copy(labels_in)
    height, width = size(image)
    background_label = zero(first(image))
    stack = [seed]
    stats = ConnectedComponentStatistics(seed[2], seed[1], seed[2], seed[1], 0, (0, 0))

    while length(stack) > 0
        ind = pop!(stack)
        if labels[ind] != -1
            continue
        end
        labels[ind] = label
        stats = update_statistics(stats, ind[1], ind[2])
        i, j = ind[1], ind[2]
        for neighbour in ((i + 1, j), (i, j + 1), (i - 1, j), (i, j - 1))
            neighbour = CartesianIndex(neighbour)
            if (0 < neighbour[1] <= height) && 
                (0 < neighbour[2] <= width) && 
                (image[neighbour] != background_label)
                    push!(stack, neighbour)
            end
        end
    end

    centroid = (stats.centroid[1]/stats.area, stats.centroid[2]/stats.area)
    stats = ConnectedComponentStatistics(
        stats.left, 
        stats.top, 
        stats.right, 
        stats.bottom, 
        stats.area, 
        centroid)
    labels, stats
end


function get_connected_components_2pass(image::AbstractArray)
    labels = -ones(size(image))
    background_color = zero(typeof(first(image)))

    height, width = size(image)
    current_label = 0
    for i in 1:height
        for j in 1:width
            north = CartesianIndex(i - 1, j)
            west  = CartesianIndex(i, j - 1)
            east = CartesianIndex(i, j + 1)
            northwest = CartesianIndex(i - 1, j - 1)
            northeast = CartesianIndex(i - 1, j + 1)
            if image[i, j] == background_color
                labels[i, j] = 0
            elseif 1==1 && 
                ((i > 1) && image[north] == background_color) && 
                ((j > 1) && image[west] == background_color)
                    current_label += 1
                    labels[i, j] = current_label
            else
                label_north = (i > 1 && labels[north] > 0) ? labels[north] : current_label
                label_west = (j > 1 && labels[west] > 0) ? labels[west] : current_label
                label = min(label_north, label_west)
                if (i > 1 && j > 1) && (labels[northwest] > 0) && (image[northwest] != background_color)
                    label = min(label, labels[northwest])
                end
                if (i > 1 && 1 < j < width) && (labels[northeast] > 0) && (image[east] != background_color)
                    label = min(label, labels[northeast])
                end
                labels[i, j] = label                
            end
        end
    end

    # for i in 1:height
    #     for j in 1:width
    #         if image[i, j] != background_color
    #             north = CartesianIndex(i - 1, j)
    #             west  = CartesianIndex(i, j - 1)
    #             east = CartesianIndex(i, j + 1)
    #             northwest = CartesianIndex(i - 1, j - 1)
    #             northeast = CartesianIndex(i - 1, j + 1)
    #             label_north = (i > 1 && labels[north] > 0) ? labels[north] : current_label
    #             label_west = (j > 1 && labels[west] > 0) ? labels[west] : current_label
    #             label = min(label_north, label_west)
    #             if (i > 1 && j > 1) && (labels[northwest] > 0) && (image[northwest] != background_color)
    #                 label = min(label, labels[northwest])
    #             end
    #             if (i > 1 && 1 < j < width) && (labels[northeast] > 0) && (image[east] != background_color)
    #                 label = min(label, labels[northeast])
    #             end
    #             labels[i, j] = label   
    #         end
    #     end
    # end       


    labels
end


end # module ConnectedComponents