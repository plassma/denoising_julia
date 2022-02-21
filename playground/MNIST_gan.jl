using MLDatasets: MNIST
using Flux.Data: DataLoader
using Flux
using CUDA
using Zygote
using Plots
using BSON: @save

pyplot()
Plots.PyPlotBackend()

lr_g = 2e-4          # Learning rate of the generator network
lr_d = 2e-4          # Learning rate of the discriminator network
batch_size = 128    # batch size
num_epochs = 1000   # Number of epochs to train for
output_period = 10 # Period length for plots of generator samples
n_features = 28 * 28# Number of pixels in each sample of the MNIST dataset
latent_dim = 100    # Dimension of latent space
opt_dscr = ADAM(lr_d)# Optimizer for the discriminator
opt_gen = ADAM(lr_g) # Optimizer for the generator

 # Load the dataset
 train_x, _ = MNIST.traindata(Float32);
 # This dataset has pixel values ∈ [0:1]. Map these to [-1:1]
 train_x = 2f0 * reshape(train_x, 28, 28, 1, :) .- 1f0 |>gpu;
 # DataLoader allows to access data batch-wise and handles shuffling.
 train_loader = DataLoader(train_x, batchsize=batch_size, shuffle=true)

function my_activation(x)
    return leakyrelu(x, 0.2f0)
end

 discriminator = Chain(Dense(n_features, 1024, my_activation),
                        Dropout(0.3),
                        Dense(1024, 512, my_activation),
                        Dropout(0.3),
                        Dense(512, 256, my_activation),
                        Dropout(0.3),
                        Dense(256, 1, sigmoid)) |> gpu

generator = Chain(Dense(latent_dim, 256, my_activation),
                    Dense(256, 512, my_activation),
                    Dense(512, 1024, my_activation),
                    Dense(1024, n_features, tanh)) |> gpu

function train_dscr!(discriminator, real_data, fake_data)
    this_batch = size(real_data)[end] # Number of samples in the batch
    # Concatenate real and fake data into one big vector
    all_data = hcat(real_data, fake_data)

    # Target vector for predictions: 1 for real data, 0 for fake data.
    all_target = [ones(eltype(real_data), 1, this_batch) zeros(eltype(fake_data), 1, this_batch)] |> gpu;

    ps = Flux.params(discriminator)
    loss, pullback = Zygote.pullback(ps) do
        preds = discriminator(all_data)
        loss = Flux.Losses.binarycrossentropy(preds, all_target)
    end
    # To get the gradients we evaluate the pullback with 1.0 as a seed gradient.
    grads = pullback(1f0)

    # Update the parameters of the discriminator with the gradients we calculated above
    Flux.update!(opt_dscr, Flux.params(discriminator), grads)
    
    return loss 
end

function train_gen!(discriminator, generator)
  # Sample noise
  noise = randn(latent_dim, batch_size) |> gpu;

  # Define parameters and get the pullback
  ps = Flux.params(generator)
  # Evaluate the loss function while calculating the pullback. We get the loss for free
  loss, back = Zygote.pullback(ps) do
      preds = discriminator(generator(noise));
      loss = Flux.Losses.binarycrossentropy(preds, 1.) 
  end
  # Evaluate the pullback with a seed-gradient of 1.0 to get the gradients for
  # the parameters of the generator
  grads = back(1.0f0)
  Flux.update!(opt_gen, Flux.params(generator), grads)
  return loss
end

lossvec_gen = zeros(num_epochs)
lossvec_dscr = zeros(num_epochs)

for n in 1:num_epochs
    loss_sum_gen = 0.0f0
    loss_sum_dscr = 0.0f0

    for x in train_loader
        # - Flatten the images from 28x28xbatchsize to 784xbatchsize
        real_data = flatten(x);

        # Train the discriminator
        noise = randn(latent_dim, size(x)[end]) |> gpu
        fake_data = generator(noise)
        loss_dscr = train_dscr!(discriminator, real_data, fake_data)
        loss_sum_dscr += loss_dscr

        # Train the generator
        loss_gen = train_gen!(discriminator, generator)
        loss_sum_gen += loss_gen
    end

    # Add the per-sample loss of the generator and discriminator
    lossvec_gen[n] = loss_sum_gen / size(train_x)[end]
    lossvec_dscr[n] = loss_sum_dscr / size(train_x)[end]

    if n % output_period == 0
        @show n
        noise = randn(latent_dim, 4) |> gpu
        fake_data = reshape(generator(noise), 28, 4*28) |> cpu
        p = heatmap(fake_data, c = :greys)
        display(p)
        @save "playground/MNIST_GAN_" * string(n) * ".bson" generator discriminator
    end
    println("finished epoch ", n)
end

@save "playground/MNIST_GAN.bson" generator discriminator