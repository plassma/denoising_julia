using MLDatasets, Plots, Flux, CUDA
using BSON: @save

pyplot()
Plots.PyPlotBackend()


# load full training set
using Flux:onehotbatch

train_x, train_y = MNIST.traindata()
train_y = onehotbatch(train_y, 0:9) |> gpu
train_x = reshape(train_x, 28*28, :) |> gpu
# load full test set
test_x, test_y  = MNIST.testdata()
test_y = onehotbatch(test_y, 0:9) |> gpu
test_x = reshape(test_x, 28*28, :) |> gpu

println(size(train_y))
println(train_y[:, 1])

m = Chain(
  Dense(28 * 28, 128, relu),
  Dense(128, 10), softmax) |> gpu

using Flux:crossentropy,throttle
loss(X, y) = crossentropy(m(X), y) 
opt = ADAM()

progress = () -> @show(loss(test_x, test_y)) # callback to show loss
using Flux:@epochs
@epochs 100 Flux.train!(loss, params(m),[(train_x,train_y)], opt, cb = throttle(progress, 10))

@save "playground/minimal_MNIST_model.bson" m

for i in 1:20
  img = test_x[:, i]

  val, idx = findmax(m(img))
  tit = "Number " * string(idx - 1)

  img = reshape(img, 28, 28)
  img = reverse(permutedims(img, [2, 1]), dims=1)
  display(heatmap(img, c = :greys, title=tit))
end