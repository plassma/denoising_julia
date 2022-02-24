using Flux: @functor
using Flux

struct SinusoidalPosEmb #todo: move this to common
    emb::AbstractArray{Float32}
    scale::Float32
end

@functor SinusoidalPosEmb

SinusoidalPosEmb(dim::Int64, scale::Float32=1f0) = dim % 2 == 0 ? SinusoidalPosEmb(ℯ .^ ((0:(dim÷2-1)) * -(log(10000) / (dim ÷ 2 - 1))), scale) : AssertionError("dim must be even")

function (s::SinusoidalPosEmb)(x)
    emb = x' * s.emb .* s.scale
    return cat(sin.(emb), cos.(emb), dims = 1)
end

struct DenseFiLM
    chain::Chain
    scale::Dense
    shift::Dense
end

@functor DenseFiLM

DenseFiLM(dim, out_dim) = DenseFiLM(Chain(SinusoidalPosEmb(dim, 5000f0), Dense(dim, dim * 4, swish), Dense(dim * 4, dim * 4, swish)), Dense(dim * 4, out_dim), Dense(dim * 4, out_dim))

function (df::DenseFiLM)(t)
    t = df.chain(t)
    return df.scale(t), df.shift(t)
end

featurewise_affine(x, scale, shift) = scale * x + shift

struct DenseResBlock
    chain::Chain
    shortcut
end

@functor DenseResBlock

DenseResBlock(in_dim::Int64, out_dim::Int64) = DenseResBlock(Chain(BatchNorm(in_dim), Dense(in_dim, out_dim), BatchNorm(out_dim), Dense(out_dim, out_dim)), in_dim == out_dim ? identity : Dense(in_dim, out_dim))

function (self::DenseResBlock)(in, scale, shift)
    x = self.chain[1](in)
    x = featurewise_affine.(x, scale, shift)
    x = swish.(x)
    x = self.chain[2:3](x)
    x = featurewise_affine.(x, scale, shift)
    x = swish.(x)
    x = self.chain[4](x)
    return x + self.shortcut(in)
end

struct DenseDDPM

end