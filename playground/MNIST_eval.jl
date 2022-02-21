using MLDatasets, Plots, Flux, CUDA
using BSON: @load

train_x, train_y = MNIST.traindata()
train_x = reshape(train_x, 28*28, :) |> gpu
# load full test set
test_x, test_y  = MNIST.testdata()
test_x = reshape(test_x, 28*28, :) |> gpu
@load "playground/minimal_MNIST_model.bson" m
m = m |> gpu

for i in 1:20
    img = test_x[:, i]
  
    val, idx = findmax(m(img))
    tit = "Number " * string(idx - 1)
  
    img = reshape(img, 28, 28)
    img = reverse(permutedims(img, [2, 1]), dims=1)
    display(heatmap(img, c = :greys, title=tit))
  end