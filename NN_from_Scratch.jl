# These warmup exercises
using MLDatasets
using Plots
using Random
using LinearAlgebra

# Read in the training data (the "feature" vectors)
train = MNIST(split=:train).features
# Display one of the points as an image
Gray.(train[:,:,1]')
# Reshape the data into a more convenient form, images as vectors
train = permutedims(train,(3,1,2))
train = reshape(train,(60000,28*28))
# Get the labels for the training data
labels = MNIST(split=:train).targets

# First, try a linear fit (least squares) model. 
x = train\labels
test = MNIST(split=:test).features
test = reshape(permutedims(test,(3,1,2)),(10_000,28*28))
testLabels = MNIST(split=:test).targets
# Calculate predictions for all of the test set (round to get to integers)
preds = round.(Int,test*x)
println("Accuracy(%): ",sum(preds.==testLabels)/length(testLabels)*100)
# Accuracy is better than random guessing

# Now try a k-NN model with k=1
trial = test[1,:]
dists=[norm(trial-train[i,:]) for i in 1:60_000]
idx = argmin(dists)
labels[idx]

# Make a function that we can use to sweep over many predictions
""" Return label of nearest neighbor to test point """
function findNNlabel(testPoint,trainData,trainLabels)
    dists = [norm(testpoint - i) for i ∈ eachrow(trainData)]
    idx = argmin(dists)
    return labels[idx]
end

# 100 predictions takes about 10 seconds. This approach has great accuracy but is slow. 
# It would take about 20 mins to make predictions on the entire test set. 
nPreds = 300
@time preds = [findNNlabel(i,train,labels) for i ∈ eachrow(test[1:nPreds,:])]

println("Accuracy(%): ",sum(preds.==testLabels[1:length(preds)])/nPreds*100|>(x->round(x,digits=1)))

# Why does the k-NN model work so well? The numbers must be well separated in the 784-dimensional space
# Make a T-SNE plot to show that. T-SNE plots take awhile. About 20 secs/1000 data points here
# Check out these related sites: 
# https://github.com/lejon/TSne.jl
# https://distill.pub/2016/misread-tsne/
using TSne
@time Y = tsne(train[1:1000,:], 2, 50, 1000, 20.0)
scatter(Y[:,1], Y[:,2], marker=(2,3,"*",stroke(0)), color=Int.(labels[1:size(Y,1)]),
        aspect_ratio=1,legend=:none)
