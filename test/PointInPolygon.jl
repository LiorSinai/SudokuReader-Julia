#=
Lior Sinai, 28 July 2021
=#

using Test
include("../Utilities/PointInPolygon.jl")
using .PointInPolygon

H_polygon = (
    [(2, 10), (4, 10), (4, 6), (6, 6), (6, 10), (8, 10), (8, 1), (6, 1), (6, 4), (4, 4), (4, 1), (2, 1)],
    BitArray([
        0  0  0  0  0  0  0  0  0  0  0
        0  0  1  1  1  0  1  1  1  0  0;
        0  0  1  1  1  0  1  1  1  0  0;
        0  0  1  1  1  0  1  1  1  0  0;
        0  0  1  1  1  1  1  1  1  0  0;
        0  0  1  1  1  1  1  1  1  0  0;
        0  0  1  1  1  1  1  1  1  0  0;
        0  0  1  1  1  0  1  1  1  0  0;
        0  0  1  1  1  0  1  1  1  0  0;
        0  0  1  1  1  0  1  1  1  0  0;
        0  0  1  1  1  0  1  1  1  0  0;
        0  0  0  0  0  0  0  0  0  0  0
    ])
    )
L_polygon = (
        [(2, 10), (4, 10), (4, 3), (7, 3), (7, 1), (2, 1)],
        BitArray([
            0  0  0  0  0  0  0  0  0  0  0;
            0  0  1  1  1  1  1  1  0  0  0;
            0  0  1  1  1  1  1  1  0  0  0;
            0  0  1  1  1  1  1  1  0  0  0;
            0  0  1  1  1  0  0  0  0  0  0;
            0  0  1  1  1  0  0  0  0  0  0;
            0  0  1  1  1  0  0  0  0  0  0;
            0  0  1  1  1  0  0  0  0  0  0;
            0  0  1  1  1  0  0  0  0  0  0;
            0  0  1  1  1  0  0  0  0  0  0;
            0  0  1  1  1  0  0  0  0  0  0;
            0  0  0  0  0  0  0  0  0  0  0
        ])
    )
rectangle = (
    [(2, 9), (8, 9), (8, 2), (2, 2)],
    BitArray([
        0  0  0  0  0  0  0  0  0  0  0;
        0  0  0  0  0  0  0  0  0  0  0;
        0  0  1  1  1  1  1  1  1  0  0;
        0  0  1  1  1  1  1  1  1  0  0;
        0  0  1  1  1  1  1  1  1  0  0;
        0  0  1  1  1  1  1  1  1  0  0;
        0  0  1  1  1  1  1  1  1  0  0;
        0  0  1  1  1  1  1  1  1  0  0;
        0  0  1  1  1  1  1  1  1  0  0;
        0  0  1  1  1  1  1  1  1  0  0;
        0  0  0  0  0  0  0  0  0  0  0;
        0  0  0  0  0  0  0  0  0  0  0;
    ])
    )
pentagon = (
    [(1, 7), (5, 10), (9, 7), (7, 1), (3, 1)],
    BitArray([
        0  0  0  0  0  0  0  0  0  0  0;
        0  0  0  1  1  1  1  1  0  0  0;
        0  0  0  1  1  1  1  1  0  0  0;
        0  0  0  1  1  1  1  1  0  0  0;
        0  0  1  1  1  1  1  1  1  0  0;
        0  0  1  1  1  1  1  1  1  0  0;
        0  0  1  1  1  1  1  1  1  0  0;
        0  1  1  1  1  1  1  1  1  1  0;
        0  0  0  1  1  1  1  1  0  0  0;
        0  0  0  0  1  1  1  0  0  0  0;
        0  0  0  0  0  1  0  0  0  0  0;
        0  0  0  0  0  0  0  0  0  0  0
    ])
    )
scew_H = (
    [(2, 10), (4, 10), (4, 6), (6, 6), (6, 10), (8, 10), (8, 1), (6, 1), (6, 4), (4, 4), (4, 2), (2, 2)],
    BitArray([
        0  0  0  0  0  0  0  0  0  0  0
        0  0  0  0  0  0  1  1  1  0  0
        0  0  1  1  1  0  1  1  1  0  0
        0  0  1  1  1  0  1  1  1  0  0
        0  0  1  1  1  1  1  1  1  0  0
        0  0  1  1  1  1  1  1  1  0  0
        0  0  1  1  1  1  1  1  1  0  0
        0  0  1  1  1  0  1  1  1  0  0
        0  0  1  1  1  0  1  1  1  0  0
        0  0  1  1  1  0  1  1  1  0  0
        0  0  1  1  1  0  1  1  1  0  0
        0  0  0  0  0  0  0  0  0  0  0
    ])
    )



@testset "centre points" begin
    @test point_in_polygon((3, 8), H_polygon[1])
    @test point_in_polygon((7, 8), H_polygon[1])
    @test point_in_polygon((5, 5), H_polygon[1])

    @test point_in_polygon((3, 7), L_polygon[1])
    @test point_in_polygon((6, 2), L_polygon[1])
    @test point_in_polygon((3, 2), L_polygon[1])

    @test point_in_polygon((5, 5), rectangle[1])

    @test point_in_polygon((5, 5), pentagon[1])
    @test point_in_polygon((5, 8), pentagon[1])
end;


@testset "outside points" begin
    @test !point_in_polygon((5, 3), H_polygon[1])
    @test !point_in_polygon((5, 8), H_polygon[1])
    @test !point_in_polygon((1, 8), H_polygon[1])

    @test !point_in_polygon((6, 8), L_polygon[1])
    @test !point_in_polygon((8, 2), L_polygon[1])

    @test !point_in_polygon((5, 10), rectangle[1])
    @test !point_in_polygon((1, 5), rectangle[1])
    @test !point_in_polygon((10, 5), rectangle[1])

    @test !point_in_polygon((2, 9), pentagon[1])
    @test !point_in_polygon((8, 9), pentagon[1])
end;


@testset "horiztonal edges" begin
    @test point_in_polygon((5, 6), H_polygon[1])
    @test !point_in_polygon((5, 10), H_polygon[1])
    @test point_in_polygon((3, 6), H_polygon[1])

    @test !point_in_polygon((8, 3), L_polygon[1])
    @test point_in_polygon((3, 3), L_polygon[1])
    @test point_in_polygon((6, 3), L_polygon[1])

    @test !point_in_polygon((5, 10), rectangle[1])
    @test !point_in_polygon((1, 5), rectangle[1])
    @test !point_in_polygon((10, 5), rectangle[1])

    @test !point_in_polygon((1, 1), pentagon[1])
    @test point_in_polygon((5, 1), pentagon[1])

    @test !point_in_polygon((1, 2), scew_H[1])
    @test !point_in_polygon((5, 2), scew_H[1])
    @test point_in_polygon((7, 2), scew_H[1])
end;


@testset "vertix intersections" begin
    @test !point_in_polygon((2, 10), pentagon[1])
    @test point_in_polygon((5, 10), pentagon[1])
    @test !point_in_polygon((8, 10), pentagon[1])

    @test !point_in_polygon((0, 7), pentagon[1])
    @test point_in_polygon((1, 7), pentagon[1])
    @test point_in_polygon((5, 7), pentagon[1])
    @test point_in_polygon((9, 7), pentagon[1])
    @test !point_in_polygon((10, 7), pentagon[1])
end;


@testset "full grids" begin
    polygons = [
        H_polygon, 
        L_polygon,
        rectangle,
        pentagon,
        scew_H
    ]
    for (polygon, answer) in polygons
        grid_ =  BitArray(undef, 12, 11)
        for x in 0:10
            for y in 0:11
                grid_[y + 1, x + 1] = point_in_polygon((x, y), polygon)
            end
        end
        @test answer == grid_
    end
end;