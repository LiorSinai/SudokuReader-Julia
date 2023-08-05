#=
1 August 2021

Convert to MNIST format so can use the same models for MNIST and the Char74K dataset

The Chars74K dataset
Source:  http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/   
=#

using Images

indir = "path\\to\\datasets/Char74k/74k_numbers"
outpath = indir * "_28x28"
include("..\\utilities\\invert_image.jl");

if !isdir(outpath)
    mkdir(outpath)
end

for indir in readdir(indir)
    println("working in $(joinpath(inpath, indir))")
    outdir = joinpath(outpath, string(parse(Int, indir[(end-1):end]) - 1))
    if !isdir(outdir)
        mkdir(outdir)
    end
    num_saved = 0
    for filename in readdir(joinpath(inpath, indir))
        num_saved += 1
        image = load(joinpath(inpath, indir, filename))
        image = imresize(image, (28, 28))
        image = invert_image(image)
        save(joinpath(outdir, filename), image) 
    end
    println("saved $num_saved files to $outdir")
end