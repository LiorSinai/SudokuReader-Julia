#=
Test Connected Components

Lior Sinai, 30 July 2021

=#

using Images
using FileIO
using Plots

include("../Utilities/ConnectedComponents.jl")
using.ConnectedComponents 

image_path = "images/blackwhite.png"
image = load(image_path)
image = Gray.(image)

labels, statistics = get_connected_components(image)
#labels = get_connected_components_multipass(image)

canvas = heatmap(labels, yflip=true)

canvas