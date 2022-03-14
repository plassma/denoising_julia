using Flux
using Statistics
using ProgressBars
using Flux: @functor

function dbg_dump_var(var, name)
    println("$name: ($(typeof(var))) [$(size(var))]")
end

function extract(input, t, shape)
    return reshape(input[t], (repeat([1], length(shape) - 1)..., :))
end

struct GaussianDiffusionModel
    num_timesteps::Int64
    data_shape

    device
    denoise_fn

    betas::Array{Float32, 1}
    alphas_cumprod::Array{Float32, 1}
    alphas_cumprod_prev::Array{Float32, 1}

    sqrt_alphas_cumprod::AbstractArray{Float32, 1}
    sqrt_one_minus_alphas_cumprod::AbstractArray{Float32, 1}
    log_one_minus_alphas_cumprod::AbstractArray{Float32, 1}
    sqrt_recip_alphas_cumprod::AbstractArray{Float32, 1}
    sqrt_recipm1_alphas_cumprod::AbstractArray{Float32, 1}
    posterior_variance::AbstractArray{Float32, 1}
    posterior_log_variance_clipped::AbstractArray{Float32, 1}
    posterior_mean_coef1::AbstractArray{Float32, 1}
    posterior_mean_coef2::AbstractArray{Float32, 1}

    GaussianDiffusionModel(model, betas, num_timesteps, image_size, device = gpu) = gaussian_diffusion_model(model, betas, num_timesteps, image_size, device)
    GaussianDiffusionModel(args...) = new(args...)

end

@functor GaussianDiffusionModel

function gaussian_diffusion_model(model, betas, num_timesteps, data_shape, device)
    alphas = 1 .- betas
    alphas_cumprod = cumprod(alphas)
    alphas_cumprod_prev = [1, (alphas_cumprod[1:end - 1] |> cpu)...] |> device#need to move to cpu as scalar indexing is not supported on gpu

    sqrt_alphas_cumprod = sqrt.(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = sqrt.(1 .- alphas_cumprod)
    log_one_minus_alphas_cumprod = log.(1 .- alphas_cumprod)
    sqrt_recip_alphas_cumprod = 1 ./ sqrt.(alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = sqrt.(1 ./ alphas_cumprod .- 1)

    posterior_variance = betas .* (1 .- alphas_cumprod_prev) ./ (1 .- alphas_cumprod)
    posterior_log_variance_clipped = log.(max.(posterior_variance, 1e-20))

    posterior_mean_coef1 = betas .* sqrt.(alphas_cumprod_prev) ./ (1 .- alphas_cumprod)
    posterior_mean_coef2 = (1 .- alphas_cumprod_prev) .* sqrt.(alphas) ./ (1 .- alphas_cumprod)

    return GaussianDiffusionModel(num_timesteps, data_shape, device, model, betas, alphas_cumprod, alphas_cumprod_prev,
    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, log_one_minus_alphas_cumprod,
    sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod,
    posterior_variance, posterior_log_variance_clipped, posterior_mean_coef1,
    posterior_mean_coef2)
end

function q_mean_variance(gdm::GaussianDiffusionModel, x_0, t)
    mean = extract(gdm.sqrt_alphas_cumprod, t, size(x_0)) .* x_0
    variance = extract(1 .- gdm.alphas_cumprod, t, size(x_0))
    log_variance = extract(gdm.log_one_minus_alphas_cumprod, t, size(x_0))

    return mean, variance, log_variance
end

function predict_start_from_noise(gdm::GaussianDiffusionModel, x_t, t, noise)
    return extract(gdm.sqrt_recip_alphas_cumprod, t, size(x_t)) .* x_t -
        extract(gdm.sqrt_recipm1_alphas_cumprod, t, size(x_t)) .* noise
end

function q_posterior(gdm::GaussianDiffusionModel, x_start, x_t, t)
    posterior_mean = extract(gdm.posterior_mean_coef1, t, size(x_t)) .* x_start +
        extract(gdm.posterior_mean_coef2, t, size(x_t)) .* x_t
    posterior_variance = extract(gdm.posterior_variance, t, size(x_t))
    posterior_log_variance_clipped = extract(gdm.posterior_log_variance_clipped, t, size(x_t))
    return posterior_mean, posterior_variance, posterior_log_variance_clipped
end

function p_mean_variance(gdm::GaussianDiffusionModel, x, t, clip_denoised)
    x_recon = predict_start_from_noise(gdm, x, t, gdm.denoise_fn(x, t))

    if clip_denoised
        x_recon = clamp.(x_recon, -1, 1)
    end

    model_mean, posterior_variance, posterior_log_variance = q_posterior(gdm, x_recon, x, t)
    return model_mean, posterior_variance, posterior_log_variance
end

function p_sample(gdm::GaussianDiffusionModel, x, t, clip_denoised=true)
    model_mean, _, model_log_variance = p_mean_variance(gdm, x, t, clip_denoised)
    
    nonzero_mask = reshape(1 .- (t .== 0.0), fill(1, length(size(x))-1)..., size(x)[end])
    noise = randn(Float32,size(x)) |> gdm.device

    sample = model_mean + nonzero_mask .* convert(AbstractArray{Float32}, ℯ.^(0.5 .* model_log_variance)) .* noise
    return sample
end

function p_sample_loop(gdm::GaussianDiffusionModel, shape)
    
    img = randn(Float32, shape) |> gdm.device
    iter = ProgressBar(gdm.num_timesteps:-1:1)
    set_description(iter, "Sampling...")
    for i in iter
        img = p_sample(gdm, img, fill(i, shape[end]) |> gdm.device, true)
    end
    return img
end

function sample(gdm::GaussianDiffusionModel, batch_size=16)
    return p_sample_loop(gdm, (gdm.data_shape..., batch_size))
end

function q_sample(gdm::GaussianDiffusionModel, x_start, t, noise=nothing)
    if isnothing(noise)
        noise = randn(Float32, size(x_start)) |> gdm.device
    end

    return extract(gdm.sqrt_alphas_cumprod, t, size(x_start)) .* x_start .+
           extract(gdm.sqrt_one_minus_alphas_cumprod, t, size(x_start)) .* noise
end

function p_lossess(gdm::GaussianDiffusionModel, x_start, t, noise=nothing)

    if isnothing(noise)
        noise = randn(Float32,size(x_start)) |> gdm.device
    end

    x_noisy = q_sample(gdm, x_start, t, noise)
    x_recon = gdm.denoise_fn(x_noisy, t)

    loss = mean((noise - x_recon).^2)

    return loss
end

function (gdm::GaussianDiffusionModel)(x_start)
    B = size(x_start)[end]
    t = rand(1:gdm.num_timesteps, B) |> gdm.device
    return p_lossess(gdm, x_start, t)
end