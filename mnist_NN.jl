using MLDatasets
using Images
using Random
using Plots

train = MNIST(split=:train).features
Gray.(train[:,:,1]')
test = MNIST(split=:test).features

data = cat(test,train,dims=3)
labels= cat(MNIST(split=:test).targets,MNIST(split=:train).targets,dims=1)

reorder = shuffle(1:70_000)
labels=labels[reorder]
data = data[:,:,reorder]
trainLabels, testLabels = labels[1:60_000],labels[60_001:end]

train = data[:,:,1:60_000]
test = data[:,:,60_001:end]
e
function showImg(arr)
    return Gray.(arr[:,:]')
end

dm=reduce(vcat, [vec(train[:,:,i]) for i in 1:size(train,3)]')
tm=reduce(vcat, [vec(test[:,:,i]) for i in 1:size(test,3)]')

c = dm\trainLabels
plot(c)

dm*c
yFit = round.(Int,dm*c)
sum(yFit.==trainLabels)/60_000 
yPred = round.(Int,tm*c)
sum(yPred.==testLabels)/10_000

# Linear fit is not so excellent. Better than random guessing but not great.

# Let's try a nearest neighbor approach
function findNNlabel(testPoint,trainData,labels)
    dists = [norm(i-testPoint) for i ∈ eachrow(trainData)]
    idx = argmin(dists)
    return labels[idx]
end

@time pred = [findNNlabel(tm[i,:],dm,labels) for i ∈ 1:10_000]
sum(pred.==testLabels)/length(pred)

play = permutedims(train[:,:,1:2500],(3,1,2)) 
play = reshape(play,2500, 28*28)
Y = tsne(play,2,50,1000,20.0)
scatter(Y[:,1], Y[:,2], marker=(2,2,"*",stroke(0)), 
color=Int.(labels[1:size(Y,1)]),aspect_ratio=1,legend=:none)




# Let's make a NN: Input layer with 784 nodes, fully connected with two hidden layers, 10 neurons each, 10 outputs.
function initWeights()
    w1 = rand(10,784).-0.5
    b1 = rand(10).-0.5
    w2 = rand(10,10).-0.5
    b2 = rand(10).-0.5
    return w1,b1,w2,b2
end

""" Convert vector of outputs to probability vector """
function softmax(z)
    return exp.(z)/sum(exp.(z))
end

""" Apply nonlinear mapping to vector. Rectified linear unit. """
function reLU(x)
    return max(0.0,x)
end

# Illustrate the steps
A0 = reshape(train[:,:,1],28*28)
z1 = w1*A0+b1
A1 = reLU.(z1)

z2 = w2*A1+b2
A2 = softmax(z2)
finalAns = argmax(A2)-1

Gray.(train[:,:,1]')
labels[1]

""" Given weights and biases for a network, make predictions for input features """
function forwardPropogation(features,w1,b1,w2,b2)
    z1 = w1*features.+b1
    A1 = reLU.(z1)
    z2 = w2*A1.+b2
    A2 = [softmax(i) for i ∈ eachcol(z2)]
    return argmax.(A2).-1
end


function one_hot(label)
    idx = label+1
    v = zeros(Int,10)
    v[idx] = 1
    return v
end

function backprop(z1, A1, z2, w1, w2, feat, label)
    n = 1/length(label) # Use this to normalize sums
    one_hot_label = one_hot(label)
    dz2 = A2 - one_hot_label
    dw2 = n*dz2*A2'
    db2 = n*sum(dz2)
    dz1 = w2'*dz2.*(z1 > 0)
    dw1 = n*dz1*feat
    db1 = n*sum(dz1)
    return dw1, db1, dw2, db2
end

"""
    Update to parameters to improve predions. α is the learning rate
"""
function updateParameters(z1,b1,z2,b2,dz1,db1,dz2,db2,α=0.1)
    z1 = z1 - α*dz1
    b1 = b1 - α*db1
    z2 = z2 - α*dz2
    b2 = b2 - α*db2
    return z1,b1,z2,b2
end

function gradientDescent(features,labels,nIters,α=0.1)
    w1,b1,w2,b2 = initWeights()
    for it ∈ 1:nIters
        z1, A1, z2, A2 = forwardPropogation(train,w1,b1,w2,b2)
        dw1, db1, dw2, db2 = backPropogation(train, features, z1, A1, z2, A2)
        w1, b1, w2, b2 = updateParameters(w1, b1, w2, b2, dw1, db1, dw2, db2)
        if !mod(it,50)
            println("Iteration: ",it)
            println("Accuracy:  ",get_accuracy(get_predictions(),labels))
        end
    end
    return w1, b1, w2, b2
end




