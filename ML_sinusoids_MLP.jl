using PyPlot
using Knet

n=1000

x_o=linspace(0,50*pi,n)
y_o=(x_o)+3*sin(x_o)
y_n=y_o+rand(n)-0.5
x_n=x_o+rand(n)-0.5
x_n2=x_o+rand(n)-0.5

plot(x_o,y_o)
plot(x_o,y_n)
plot(x_n,y_n)

 #TRAIN NEURAL NETWORK TO FIND THE ORIGINAL SIGNAL
 #w = Any[ 0.1*randn(1,13), 0 ]

x=x_n
y=y_n
using Knet

 function train(w, data; lr=.1)
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
    for i=1:2:length(w)-2
        x = max(0, w[i]*x .+ w[i+1])
    end
    return w[end-1]*x .+ w[end]
end


# MLP WITH TWO HIDDEN LAYERS 
w = Any[ -0.1+0.2*rand(Float32,64,n), zeros(Float32,64,1),zeros(Float32,64,64),zeros(Float32,64,1),
         -0.1+0.2*rand(Float32,n,64),  zeros(Float32,n,1) ]


loss(w,x,y) = sumabs2(y - predict(w,x))

lossgradient = grad(loss)


 for i=1:500; train(w, [(x,y)]); println("iter: ",i,"| loss:",loss(w,x,y)); end

Figure(2)
plot(x_o,y_o)
plot(x,predict(w,x))
plot(x_o,predict(w,x_n2))
