#CLASSIFICATION

function loss(w,x,ygold)
    ypred = predict(w,x)
    ynorm = ypred .- log(sum(exp(ypred),1))
    -sum(ygold .* ynorm) / size(ygold,2)
end

function accuracy(w, data)
    ncorrect = ninstance = 0
    for (x, ygold) in data
        ypred = predict(w,x)
        ncorrect += sum(ygold .* (ypred .== maximum(ypred,1)))
        ninstance += size(ygold,2)
    end
    return ncorrect/ninstance
end

function train(w, data; lr=.1)
    for (x,y) in data
        dw = lossgradient(w, x, y)
        for i in 1:length(w)
            w[i] -= lr * dw[i]
        end
    end
    return w
end

predict(w,x) = w[1]*x .+ w[2]

loss(w,x,y) = sumabs2(y - predict(w,x)) / size(y,2)

using Knet

lossgradient = grad(loss)
