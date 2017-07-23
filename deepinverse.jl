#DEEP INVERSE: DEEP LEARNING FOR CS

using Knet
using PyPlot
n=1000;

function sparsesignal(n,frac)
  x=rand(n).>frac
  x=float(x)
return x
end
mask=rand(n).>0.6

function us_Fmeas(x,n,mask)
  y0=fft(x)
  yu=y0.*(mask)
  y=ifft(yu)
  return y
end

x=sparsesignal(n,0.98)
y=us_Fmeas(x,n,mask)
plot(x)
plot(abs(y))





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
        x = max(0, abs(w[i]*x .+ w[i+1])) #added abs here because max cannot deal with complex
    end
    return w[end-1]*x .+ w[end]
end

# MLP WITH TWO HIDDEN LAYERS
w = Any[ -0.1+0.2*rand(Float32,64,n), zeros(Float32,64,1),
          zeros(Float32,64,64),zeros(Float32,64,1),
          zeros(Float32,4,64),zeros(Float32,4,1),
          zeros(Float32,64,4),zeros(Float32,64,1),
         -0.1+0.2*rand(Float32,n,64),  zeros(Float32,n,1) ]


#=
function predict(w,x0)
    x1 = pool(max(0, conv4(w[1],x0) .+ w[2]))
    x2 = pool(max(0, conv4(w[3],x1) .+ w[4]))
    x3 = max(0, w[5]*mat(x2) .+ w[6])
    return w[7]*x3 .+ w[8]
end

w = Any[ -0.1+0.2*rand(Float64,5,5,1,n),  zeros(Float64,1,1,n,1),
         -0.1+0.2*rand(Float64,5,5,n,50), zeros(Float64,1,1,50,1),
         -0.1+0.2*rand(Float64,500,800),   zeros(Float64,500,1),
         -0.1+0.2*rand(Float64,n,500),    zeros(Float64,n,1) ]
=#
predict(w,x)

loss(w,x,y) = sumabs2(y - predict(w,x)) #l1?

lossgradient = grad(loss)

function makesignals
  for batchsize=1:5
    if batchsize==1
      x=sparsesignal(n,0.98)
      y=us_Fmeas(x,n,mask)
    else
      x0=sparsesignal(n,0.98)
      y0=us_Fmeas(x0,n,mask)
      x=cat(2,x,x0)
      y=cat(2,y,y0)
    end
  end
return x
end


 for i=1:1000;
   [x,y]=makesignals
   for j=1:5
   train(w, [(y,x)]); println("iter: ",i,"| loss:",loss(w,y,x));  #CHANGED y and x (y is measurement, x true val)
    end
 end


Figure(2)
plot(x)
xp=predict(w,x);
plot(xp)
