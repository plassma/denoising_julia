using TFRecord
include("../utils.jl")

searchdir(path,key) = filter(x->contains(x,key), readdir(path))

function to_vector(X, data_shape, limit=nothing)
    result = Matrix{Float32}[]
    for (i, x) in enumerate(X)
        reshaped = reshape(x.features.feature["inputs"].float_list.value, data_shape)
        push!(result, reshaped)
        
        if !isnothing(limit) && i > limit
            break
        end
    end
    return reshape(vcat(result...), (data_shape..., :))
end

function normalize!(x)

    x_max = maximum(x)
    x_min = minimum(x)

    x .-= x_min
    x ./= (x_max - x_min)
    x .*= 2
    x .-= 1
end

function get_dataset(path= "/media/matthias/Data/preprocessed_clean_encoded/", data_shape=(32, 512), normalize=true; limit=nothing)
    train_ds = TFRecord.read(path .* searchdir(path, "train"))
    eval_ds = TFRecord.read(path .* searchdir(path, "eval"))

    train_x = to_vector(train_ds, data_shape, limit)
    eval_x = to_vector(eval_ds, data_shape, limit)
    
    if normalize
        normalize!(train_x)
        normalize!(eval_x)
    end

    return train_x, eval_x
end