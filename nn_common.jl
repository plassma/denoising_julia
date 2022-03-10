using Flux
using Flux: @functor
using OMEinsum, SymEngine

struct SinusoidalPosEmb #todo: move this to common
    emb::AbstractArray{Float32}
end

@functor SinusoidalPosEmb

SinusoidalPosEmb(dim::Int64) = dim % 2 == 0 ? SinusoidalPosEmb(ℯ .^ ((0:(dim÷2-1)) * -(log(10000) / (dim ÷ 2 - 1)))) : AssertionError("dim must be even")

function (s::SinusoidalPosEmb)(x)
    emb = x' .* s.emb
    return cat(sin.(emb), cos.(emb), dims = 1)
end

struct Dense2D
    dense::Dense
end

@functor Dense2D

Dense2D(in, out, σ=identity; bias=true) = Dense2D(Dense(in, out, σ; bias = bias))

function (self::Dense2D)(x)
    d1, d2, B = size(x) #32, 512, batch_size
    x = reshape(x, d2, d1 * B)
    result = self.dense(x)
    return reshape(result, d1, :, B)
end

struct SelfAttention
    scale::Float32
    heads::Int64
    qkv::Tuple
    out::Dense2D
end

@functor SelfAttention

function SelfAttention(dim, heads=8, dim_head=dim ÷ heads)
    scale = dim_head ^ -0.5
    hidden_dim = dim_head * heads
    SelfAttention(scale, heads, Tuple(Dense2D(dim, hidden_dim; bias=false) for i = 1:3), Dense2D(hidden_dim, dim))
end

function (self::SelfAttention)(x)
    shape = size(x)
    if length(size(x)) == 4
        x = reshape(x, :, shape[3:end]...)
    end
    n, _, B = size(x)

    q, k, v = (reshape(self.qkv[i](x), n, :, self.heads, B) .* self.scale for i = 1:3)

    k = softmax(k, dims=1)
    
    context = ein"n d h b, n e h b -> e d h b"(k, v)

    out = ein"e d h b, n d h b -> n e h b"(context, q)
    out = reshape(out, n, :, B)

    return reshape(self.out(out), shape...)
end
