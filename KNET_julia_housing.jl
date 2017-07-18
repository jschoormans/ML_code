using FileIO
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
rawdata = readdlm(download(url))
x = rawdata[:,1:13]'
x = (x .- mean(x,2)) ./ std(x,2)
y = rawdata[:,14:14]'
w = Any[ 0.1*randn(1,13), 0 ]

using Knet


predict(w,x) = w[1]*x .+ w[2]

loss(w,x,y) = sumabs2(y - predict(w,x)) / size(y,2)
lossgradient = grad(loss)


for i=1:25; train(w, [(x,y)]); println(loss(w,x,y)); end
println(w)
