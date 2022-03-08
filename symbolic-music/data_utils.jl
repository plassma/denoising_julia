using TFRecord
using Flux: batch
include("../utils.jl")

searchdir(path,key) = filter(x->contains(x,key), readdir(path))

function to_vector(X, limit=nothing)
    result = Matrix{Float32}[]
    for (i, x) in enumerate(X)
        reshaped = reshape(x.features.feature["inputs"].float_list.value, (512, 32))
        push!(result, reshaped)
        if !isnothing(limit) && i >= limit
            break
        end
    end
    permutedims(batch(result), [2, 1, 3])
end

function normalize!(x)
    x_max = maximum(x)
    x_min = minimum(x)

    x .-= x_min
    x ./= (x_max - x_min)
    x .*= 2
    x .-= 1
end

function get_dataset(path= "/media/matthias/Data/preprocessed_clean_encoded/"; normalize=true, limit=nothing)
    train_ds = TFRecord.read(path .* searchdir(path, "train"))
    test_ds = TFRecord.read(path .* searchdir(path, "eval"))
    train_x = to_vector(train_ds, limit)
    test_x = to_vector(test_ds, limit)
    
    if normalize
        normalize!(train_x)
        normalize!(test_x)
    end
    println("Loaded data. |train| = $(size(train_x)[end]), |test| = $(size(test_x)[end])")
    train_x, test_x
end