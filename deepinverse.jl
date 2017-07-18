#DEEP INVERSE: DEEP LEARNING FOR CS



function sparsesignal(n,frac)
  x=rand(n).>frac
  x=float(x)
return x
end
mask=rand(n).>0.9

function us_Fmeas(n,mask)
  y0=fft(x)
  yu=y0.*(mask)
  y=ifft(yu)
  return y
end

n=1000;
x=sparsesignal(n,0.98)
y=us_Fmeas(n,0.9)
plot(x)
plot(abs(y))





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
        x = max(0, abs(w[i]*x .+ w[i+1])) #added abs here because max cannot deal with complex
    end
    return w[end-1]*x .+ w[end]
end


# MLP WITH TWO HIDDEN LAYERS
w = Any[ -0.1+0.2*rand(Float32,64,n), zeros(Float32,64,1),zeros(Float32,64,64),zeros(Float32,64,1),
         -0.1+0.2*rand(Float32,n,64),  zeros(Float32,n,1) ]


loss(w,x,y) = sumabs2(y - predict(w,x))

lossgradient = grad(loss)

for epoch=1:10
 for i=1:100;
   train(w, [(y,x)]); println("iter: ",i,"| loss:",loss(w,y,x));  #CHANGED y and x (y is measurement, x true val)
 end
 x=sparsesignal(n,0.98)
 y=us_Fmeas(n,mask)
 end

Figure(2)
plot(x)
plot(predict(w,y))
