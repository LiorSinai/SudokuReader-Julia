
function align_centres(centres::Matrix, guides::BitMatrix)
    centres_aligned = copy(centres)
    if size(centres) != size(guides)
         throw("$(size(centres)) != $(size(guides)), sizes of centres and guides must be the same.")
    end
    for i in 1:size(centres, 1)
        for j in 1:size(centres, 2)
            if !guides[i, j]
                # y is common to row i
                if any(guides[i, :])
                    ys = [point[1] for point in centres[i, :]] .* guides[i, :]
                    Cy = sum(ys) / count(guides[i, :])
                else
                    Cy = centres[i, j][1]
                end
                #  x is common to column j
                if any(guides[:, j])
                    xs = [point[2] for point in centres[:, j]] .* guides[:, j]
                    Cx = sum(xs) / count(guides[:, j])
                else 
                    Cx = centres[i, j][2]
                end
                centres_aligned[i, j] = (Cy, Cx)
            end
        end
    end
    centres_aligned
end