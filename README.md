# Symbolic Music Diffusion in Julia
This project aims to be a standalone implementation of Denoising Diffusion Probabilistic Models (DDPMs) in Julia, and is based on [Ho et al. 2020](https://arxiv.org/abs/2006.11239).
The application of DDPMs to the domain of Symbolic Music Generation is based on [Mittal et al. 2021](https://archives.ismir.net/ismir2021/paper/000058.pdf).
This README will provide a short summary of the theoretical background of DDPMs, and as well as instructions on how to use this implementation.
## Theoretical Background
DDPMs are generative models that in 2020 were shown to be able to outperform other types of generative models such as GANs or VAEs in some image generation problems.
Unlike the latter two, DDPMs do not generate a sample in one forward pass, but rather in <img src="https://render.githubusercontent.com/render/math?math=T"> refinement steps.
Each refinement step can be interpreted as subtracting a small amount of gaussian noise.
The gaussian noise that should be subtracted can be learned by starting with data from the desired distribution, iteratively applying gaussian noise (diffusion steps), and training a neural network on the inversion of that process.

Each step in the forward diffusion process, a small amount of gaussian noise is added to the latent vector <img src="https://render.githubusercontent.com/render/math?math=x_{t-1}"> resulting in the next latent vector <img src="https://render.githubusercontent.com/render/math?math=x_t">:
	<img src="https://render.githubusercontent.com/render/math?math=x_t = \sqrt{1 - \beta_t}x_{t-1} + \sqrt{\beta_t}\mathcal{N}(0, I)">
or written as Markov transition probability:
	<img src="https://render.githubusercontent.com/render/math?math=q(x_t\mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_t I)">
$\beta_t$ is a scalar coefficient that determines the proportion of $x_t$ that is replaced by random noise. Usually a noisier sample allows for a larger update (constrained by the reverse diffusion process), so that $0 < \beta_t < \beta_{t+1} < 1$.
Although the $\beta_t$s and $\sigma_t$s could be trainable parameters, they can be set to fixed constants in practice.
The reverse diffusion process can be formulated as:
$$p_\theta(\textbf{x}_{t-1}\mid \textbf{x}_t) = \mathcal{N}(\textbf{x}_{t-1};-\boldsymbol{\mu}_\theta(x_t, t), \boldsymbol{\Sigma}_\theta(\textbf{x}_t, t))$$where $\theta$ is a function (typically a neural network) that predicts the parameters of the gaussian noise to be subtracted.
![forward and backward diffusion process](https://github.com/plassma/denoising_julia/raw/master/doc/fwd_bwd_diffusion.png)
The graphic above illustrates the forward and backward diffusion process, and was obtained from [Ho et al. 2020](https://arxiv.org/abs/2006.11239).

### Algorithms

Using $\alpha_t := 1-\beta_t$, $\alpha_t = \prod_{i=1}^t\alpha_i$ and the reparametrization trick, $x_t$ can be obtained in a single timestep requiring conditioning on $x_0$ only:
$$x_t = \sqrt{\bar{\alpha_t}}x_0 + \sqrt{1 - \bar{\alpha_t}}\mathcal {N}(0, \textbf{I})$$

The used loss function is:

$$L(\theta) := \mathbb{E}_{t, x_0, \epsilon}\Big[
\lVert\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon, t)\rVert^2\Big]$$
where $\epsilon \sim \mathcal N (\textbf{0, I})$.
This loss function can be derived from the evidence lower bound as shown in [Ho et al. 2020](https://arxiv.org/abs/2006.11239), and can also be interpreted intuitively: At timestep $t$, model $\theta$ should predict the noise $\epsilon$ which was used to distort the $x_t$ it received as input. The model also receives the timestep $t$ as an additional input (it is also possible to provide the model with $\sqrt{(\bar{\alpha_t})}$ instead of $t$ as time condition).

With this background knowledge, we are now able to formulate the training and sampling algorithms:
#### Training
<code>**repeat:**
&nbsp;$x_0 \sim q(\textbf{x}_0)$
&nbsp;$t \sim \text{Uniform}([1...T])$
&nbsp;$\epsilon \sim \mathcal{N}(\mathbf{0, I})$
&nbsp;**gradient step on**
&nbsp;&nbsp;$\nabla_\theta\lVert\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon, t)\rVert^2$
**until converged**
</code>
#### Sampling
<code>$x_T \sim \mathcal N(\textbf{0, I})$
**for**  $t = T:1$
&nbsp;$\mathbf z \sim N(\textbf{0, I})$
&nbsp;$\mathbf x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Big( \mathbf x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha_t}}}\mathbf{\mu_\theta(x_t, t)}\Big) + \sigma_t\mathbf z$
**end**
**return** $\mathbf{x_0}$
</code>


## Application to Symbolic Music
DDPMs can for example directly be applied to greyscale images, which is illustrated in the implementation using the MNIST dataset.
The discrete domain of symbolic music is an obstacle in employing Diffusion Models, which can be overcome by using the MusicVAE from [Roberts et al. 2018](http://proceedings.mlr.press/v80/roberts18a.html) and encoding discrete symbolic music tokens into continuous latent vectors.
In their implementation, [Mittal et al. 2021](https://archives.ismir.net/ismir2021/paper/000058.pdf) use pieces fixed to a length of 64 bars and encode them with the MusicVAE, 2 bars in each latent vector of 512 values, resulting in a matrix of 32x512 continuous values.
![forward and backward diffusion process](https://github.com/plassma/denoising_julia/raw/master/doc/symbolic-music-diffusion.png)
The graphic above was obtained from [Mittal et al. 2021](https://archives.ismir.net/ismir2021/paper/000058.pdf) and illustrates how the diffusion model operates on the melody tokens of discrete symbolic music.

## Implementation
My implementation of the Gaussian Diffusion Model was inspired by [lucidrains implementation](https://github.com/lucidrains/denoising-diffusion-pytorch).
As already mentioned in the background section, $\beta_t$ and $\sigma_t$ are set to time dependent constants, where $\sigma_t = \sqrt{\frac{\beta_t(1-\bar{\alpha}_{t - 1})}{(1 - \bar{\alpha}_t)}}$ and thus increases with increasing timesteps.
The core implementation consists of a callable struct named `GaussianDiffusionModel`. The struct requires a denoising function which must be a differentiable Flux model that maps from a data point $x$ and a time condition $t$ to a data point of the same shape, representing the mean of the gaussian noise to be subtracted at denoising step $t$. The other parameters of the `GaussianDiffusionModel` are a $\beta$ schedule (array of $T$ $\beta$s), the shape of the data to be diffused/sampled and the device to perform calculations on (cpu/gpu).
Additionally, a struct named `Trainer` is available. This struct takes the `GaussianDiffusionModel`to be trained, and a training- and an optional test-dataloader. The learning rate and number of epochs must also be specified. In addition, it is possible to provide a function which handles (plot/save/...) samples that can be generated after each epoch. An early-stopping criterion can also be specified.
For training and sampling, the `train!` and `sample` function are available.

### Usage
The following code illustrates a minimalist example for training. Training the diffusion model saves the GaussianDiffusionModel to `./results/{timestamp}/diffusion_model_{000epoch}.bson` after each epoch.

```julia
using .DiffusionModels
timesteps = 1000
device = gpu
data_shape = (...)
model = Model(...) |> device
betas = make_betas(...) |> device
diffusion = GaussianDiffusionModel(model, betas, data_shape, device)
train_loader = DataLoader(...)
test_loader = DataLoader(...)
trainer = Trainer(diffusion, train_loader, test_loader, 1e-3, 10)
train!(trainer)
```

After training, sampling should be straightforward:
```julia
using .DiffusionModels
device = gpu
batch_size = 64
@load  "results/{timestamp}/diffusion_model_1000.bson" model
model = model |> device
samples = sample(model, batch_size)
```

### MNIST Diffusion
The MNIST set was chosen as a form of validation for the `GaussianDiffusionModel`, it is very easy to access,  the diffusion model is applicable to the dataset with only very little preprocessing, and (intermediate) results can easily be verified visually.
The denoising model used in this example usage is a variant of a `UNet` ([Ronneberger 15](https://arxiv.org/pdf/1505.04597.pdf)) and gradually downsamples the input image to smaller images with an increasing number of channels.
In the middle of the `UNet`, a `LinearAttentionLayer` transforms the small featuremaps, before they are gradually upsampled to their original shape again.
### Symbolic Music Diffusion

The implementation of symbolic music diffusion relies greatly on [Mittal et al.'s implementation](https://github.com/magenta/symbolic-music-diffusion) of symbolic music generation.
symbolic-music/data_utils.jl is able to process data encoded and transformed according to [Mittal et al.'s implementation](https://github.com/magenta/symbolic-music-diffusion).
(Note that in the default configuration of [Mittal et al.'s implementation](https://github.com/magenta/symbolic-music-diffusion), only $\frac{42}{512}$ latent values per 2 bars are used to speed up the training process. During sampling, the remaining $470$ latent values are sampled randomly.)
Once prepared, the preprocessed .tfrecrod data can be read and used for training with the `get_dataset(path; normalize=true, limit=nothing)` function from symbolic-music/data_utils.jl.
The training process only differs very little from the minimalist example above, and is implemented in symbolic-music/diffuse_MVAE_latents_*.jl.
For the diffusion of the MVAE latents, the following two models were implemented:

#### DenseDDPM
The `DenseDDPM` model consists of standard layers like `Dense` and `BatchNorm`, but also two noteworthy types of layers: the `DenseFiLM` and the `DenseResBlock`.
The `DenseFiLM` uses a sinusoidal embedding of the timestep $t$ to produce featurewise linear modulations (affine transformations).
The `DenseResBlock` then applies these modulations to the latent matrix and also has a skip connection to allow a better flow of gradients.

#### TransformerDDPM
The `TransformerDDPM` is a `DenseDDPM` preceded by some `SelfAttentionLayer`.
In addition to the conditioning on the timestep $t$ using affine transformations, the temporal position of each of the 32 latent vectors is considered in this architecture. This is achieved by adding the sinusoidal position (1:32) encoding of each vector to itself.
The `SelfAttentionLayer`s recognize and amplify important information in the input matrix, before passing it to the `DenseDDPM' layers.

### Audio Sampling
After training, latent samples can be generated using symbolic-music/sample_MVAE_latents.jl.
To decode those latent samples (saved in .npz format), the original script from [Mittal et al.'s implementation](https://github.com/magenta/symbolic-music-diffusion) can be used.
 
## Evaluation

During training, evaluation is performed after each epoch, if an evaluation dataset is provided.
After training,  a plot of the training and evaluation losses is saved to the results folder.

![forward and backward diffusion process](https://github.com/plassma/denoising_julia/raw/master/doc/training.png)

[Mittal et al. 2021](https://archives.ismir.net/ismir2021/paper/000058.pdf) define a metric for evaluating the long term temporal consistency and variance of decoded musical pieces.
Considering time windows with a length of 4 measures, the parameters for the gaussian distribution (mean and variance) of both pitch and duration of notes occurring in those windows are calculated.
Then the overlap areas (OAs) between adjacent windows are calculated.
Temporal consistency and variance are defined as follows:
$$Consistency = \max\Big(0, 1 - \frac{\mid\mu_{OA} - \mu_{GT}\mid}{\mu_{GT}}\Big)\\
    Variance = \max\Big(0, 1 - \frac{\mid\sigma_{OA}^2 - \sigma_{GT}^2\mid}{\sigma_{GT}^2}\Big)$$
As these metrics only make sense when a GT is available for comparison (i.e. only parts of a piece have been replaced/infilled using diffusion), and infilling is not implemented, they can not be applied in this implementation.
The OAs without comparison to a GT are of no significance either, as a trivial piece only consisting of one note and one pitch would already score an OA of 100%.
