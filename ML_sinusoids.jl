using PyPlot
using Knet

n=1000

x=linspace(0,50*pi,n)
y=(x)+3*sin(x)
y_n=y+rand(n)-0.5
plot(x,y)
plot(x,y_n)

 #TRAIN NEURAL NETWORK TO FIND THE ORIGINAL SIGNAL
 #w = Any[ 0.1*randn(1,13), 0 ]

x=x'
using Knet

 function train(w, data; lr=.0000001)
    for (x,y) in data
        dw = lossgradient(w, x, y)
        for i in 1:length(w)
            w[i] -= lr * dw[i]
        end
    end
    return w
end

#predict(w,x) = w[1]*x .+ w[2]
function predict(w,x)
return w[1].*x .+ w[2].*sin(x)
end
w =  Any[ 0.1*randn(1,1), 0 ]

loss(w,x,y) = sumabs2(y - predict(w,x))

lossgradient = grad(loss)


 for i=1:2500; train(w, [(x,y)]); println("iter: ",i,"| loss:",loss(w,x,y)); end
 println(w)

plot(x,predict(w,x))
