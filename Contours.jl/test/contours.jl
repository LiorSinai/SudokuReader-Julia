#=
Lior Sinai, 28 July 2021
=#

using Test
include("../src/Contours.jl")
using .Contours

square = [
    0  0  0  0  0  0  0  0  0  0;
    0  1  1  1  1  1  1  1  1  0;
    0  1  1  1  1  1  1  1  1  0;
    0  1  1  1  1  1  1  1  1  0;
    0  1  1  1  1  1  1  1  1  0;
    0  1  1  1  1  1  1  1  1  0;
    0  1  1  1  1  1  1  1  1  0;
    0  1  1  1  1  1  1  1  1  0;
    0  1  1  1  1  1  1  1  1  0;
    0  0  0  0  0  0  0  0  0  0;
]
square_cnt = CartesianIndex[CartesianIndex(2, 2), CartesianIndex(3, 2), CartesianIndex(4, 2), CartesianIndex(5, 2), CartesianIndex(6, 2), CartesianIndex(7, 2), CartesianIndex(8, 2), CartesianIndex(9, 2), CartesianIndex(9, 3), CartesianIndex(9, 4), CartesianIndex(9, 5), CartesianIndex(9, 6), CartesianIndex(9, 7), CartesianIndex(9, 8), CartesianIndex(9, 9), CartesianIndex(8, 9), CartesianIndex(7, 9), CartesianIndex(6, 9), CartesianIndex(5, 9), CartesianIndex(4, 9), CartesianIndex(3, 9), CartesianIndex(2, 9), CartesianIndex(2, 8), CartesianIndex(2, 7), CartesianIndex(2, 6), CartesianIndex(2, 5), CartesianIndex(2, 4), CartesianIndex(2, 3)]

wings = [
    0  0  0  0  0  0  0  0  0  0;
    0  0  1  0  0  0  1  0  0  0;
    0  1  1  1  0  1  1  1  0  0;
    0  1  1  1  1  1  1  1  1  0;
    0  1  1  1  1  1  1  1  1  0;
    0  1  1  1  1  1  1  1  1  0;
    0  1  1  1  1  1  1  1  1  0;
    0  1  1  1  0  1  1  1  1  0;
    0  1  1  0  0  0  1  1  0  0;
    0  0  0  0  0  0  0  0  0  0;
]
wings_cnt = CartesianIndex[CartesianIndex(2, 3), CartesianIndex(3, 2), CartesianIndex(4, 2), CartesianIndex(5, 2), CartesianIndex(6, 2), CartesianIndex(7, 2), CartesianIndex(8, 2), CartesianIndex(9, 2), CartesianIndex(9, 3), CartesianIndex(8, 4), CartesianIndex(7, 5), CartesianIndex(8, 6), CartesianIndex(9, 7), CartesianIndex(9, 8), CartesianIndex(8, 9), CartesianIndex(7, 9), CartesianIndex(6, 9), CartesianIndex(5, 9), CartesianIndex(4, 9), CartesianIndex(3, 8), CartesianIndex(2, 7), CartesianIndex(3, 6), CartesianIndex(4, 5), CartesianIndex(3, 4)]

snake = [
    0  0  0  0  0  0  0  0  0  0;
    0  0  1  0  0  0  1  1  0  0;
    0  1  1  0  0  1  1  1  1  0;
    0  1  1  0  0  1  1  1  1  1;
    0  1  1  0  0  1  1  0  1  1;
    0  1  1  1  1  1  1  0  1  1;
    0  0  1  1  1  1  0  0  1  1;
    0  0  0  1  1  0  0  0  0  1;
    0  0  0  0  0  0  0  0  0  0;
    0  0  0  0  0  0  0  0  0  0;
]
snake_cnt = CartesianIndex[CartesianIndex(2, 3), CartesianIndex(3, 2), CartesianIndex(4, 2), CartesianIndex(5, 2), CartesianIndex(6, 2), CartesianIndex(7, 3), CartesianIndex(8, 4), CartesianIndex(8, 5), CartesianIndex(7, 6), CartesianIndex(6, 7), CartesianIndex(5, 7), CartesianIndex(4, 8), CartesianIndex(5, 9), CartesianIndex(6, 9), CartesianIndex(7, 9), CartesianIndex(8, 10), CartesianIndex(7, 10), CartesianIndex(6, 10), CartesianIndex(5, 10), CartesianIndex(4, 10), CartesianIndex(3, 9), CartesianIndex(2, 8), CartesianIndex(2, 7), CartesianIndex(3, 6), CartesianIndex(4, 6), CartesianIndex(5, 6), CartesianIndex(6, 5), CartesianIndex(6, 4), CartesianIndex(5, 3), CartesianIndex(4, 3), CartesianIndex(3, 3)]

fish = [
    0  0  0  0  0  0  0  0  0  0;
    0  0  0  0  0  0  0  0  1  0;
    0  0  0  0  1  0  0  0  1  0;
    0  0  0  1  1  0  0  1  1  0;
    0  0  1  1  1  1  1  1  1  0;
    0  1  1  1  1  1  1  1  1  0;
    0  1  1  1  1  1  1  1  1  0;
    0  0  1  1  1  1  0  1  1  0;
    0  0  0  0  0  0  0  0  1  0;
    0  0  0  0  0  0  0  0  0  0;
]
fish_cnt = CartesianIndex[CartesianIndex(2, 9), CartesianIndex(3, 9), CartesianIndex(4, 8), CartesianIndex(5, 7), CartesianIndex(5, 6), CartesianIndex(4, 5), CartesianIndex(3, 5), CartesianIndex(4, 4), CartesianIndex(5, 3), CartesianIndex(6, 2), CartesianIndex(7, 2), CartesianIndex(8, 3), CartesianIndex(8, 4), CartesianIndex(8, 5), CartesianIndex(8, 6), CartesianIndex(7, 7), CartesianIndex(8, 8), CartesianIndex(9, 9), CartesianIndex(8, 9), CartesianIndex(7, 9), CartesianIndex(6, 9), CartesianIndex(5, 9), CartesianIndex(4, 9), CartesianIndex(3, 9)]

grids = [square, wings, snake, fish]
contours = [square_cnt, wings_cnt, snake_cnt, fish_cnt]


@testset "point in contour" begin
    for (idx, contour) in enumerate(contours)
        grid =  zeros(Int, 10, 10)
        for i in 1:10
            for j in 1:10
                grid[i, j] = point_in_polygon(CartesianIndex(i, j), contour) ? 1 : 0
            end
        end
        @test all(grids[idx] .== grid)
    end
end;

@testset "fill contour" begin
    for (idx, contour) in enumerate(contours)
        blank = zeros(Int, 10, 10)
        fill_contour!(blank, contour, 1)
        @test all(grids[idx] .== blank)
    end
end;