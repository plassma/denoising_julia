using Flux: @functor

struct SinusoidalPosEmb #todo: move this to common
    emb::AbstractArray{Float32}
end

@functor SinusoidalPosEmb

SinusoidalPosEmb(dim::Int64) = SinusoidalPosEmb(ℯ .^ ((0:(dim÷2-1)) * -(log(10000) / (dim ÷ 2 - 1))))

function (s::SinusoidalPosEmb)(x)
    emb = x' .* s.emb
    return cat(sin.(emb), cos.(emb), dims = 1)
end