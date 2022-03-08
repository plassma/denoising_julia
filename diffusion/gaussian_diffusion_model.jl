using Flux
using Statistics
using ProgressBars
using Flux: @functor

function dbg_dump_var(var, name)
    println("$name: ($(typeof(var))) [$(size(var))]")
end

function make_beta_schedule(n_timestep, s = 0.008)
    n_timestep +=1
    x = range(0, n_timestep, n_timestep)
    alphas_cumprod = cos.(((x ./ n_timestep) .+ s) ./ (1 .+ s) .* pi .* 0.5) .^ 2
    alphas_cumprod ./= alphas_cumprod[1]
    betas = 1 .- (alphas_cumprod[2:end] ./ alphas_cumprod[1:end - 1])
    return clamp.(betas, 0, 0.999)
end

function extract(input, t, shape)
    return reshape(input[t], (repeat([1], length(shape) - 1)..., :))
end

approx_standard_normal_cdf(x) = 0.5 .* (1.0 .+ tanh.(sqrt(2.0 / pi) .* (x .+ 0.044715 .* x.^3)))

function discretized_gaussian_log_likelihood(x, means, log_scales)
    centered_x = x - means
    inv_stdv = ℯ.^-log_scales
    plus_in = inv_stdv .* (centered_x .+ 1 / 255)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x .+ 1 / 255)
    cdf_min = approx_standard_normal_cdf(min_in)

    log_cdf_plus = log.(clamp.(cdf_plus, 1e-12))
    log_one_minus_cdf_min = log.(clamp(1 .- cdf_min, 1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = ifelse.(x .< -0.999, log_cdf_plus, ifelse.(x .> 0.999, log_one_minus_cdf_min,
                                                log(clamp(cdf_delta, 1e-12))))
    return log_probs
end

struct GaussianDiffusionModel
    num_timesteps::Int64
    data_shape

    device
    denoise_fn #todo: type this

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

function interpolate(gdm::GaussianDiffusionModel, x1, x2, t=nothing, lam=0.5)
    if isnothing(t)
        t = gdm.num_timesteps -1
    end

    t_batched = fill(t, size(x1, 1))
    f(x) = q_sample(gdm, x, t_batched)
    xt1, xt2 = map(f, (x1, x2))

    img = (1 - lam) * xt1 + lam * xt2

    for i = gdm.num_timesteps-1:-1:0
        img = p_sample(gdm, img, fill(i, size(x1, 1)), true)
    end

end

function q_sample(gdm::GaussianDiffusionModel, x_start, t, noise=nothing)
    if isnothing(noise)
        noise = randn(Float32, size(x_start)) |> gdm.device
    end

    return extract(gdm.sqrt_alphas_cumprod, t, size(x_start)) .* x_start .+ #todo: remove all unecessary broadcasting operators
           extract(gdm.sqrt_one_minus_alphas_cumprod, t, size(x_start)) .* noise
end

function p_lossess(gdm::GaussianDiffusionModel, x_start, t, noise=nothing)

    if isnothing(noise)
        noise = randn(Float32,size(x_start)) |> gdm.device
    end

    x_noisy = q_sample(gdm, x_start, t, noise)
    x_recon = gdm.denoise_fn(x_noisy, t)

    loss = mean((noise - x_recon).^2) #todo: loss types?

    return loss
end

function (gdm::GaussianDiffusionModel)(x_start)
    B = size(x_start)[end]
    t = rand(1:gdm.num_timesteps, B) |> gdm.device
    return p_lossess(gdm, x_start, t)
end