import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10,10,50)
y = np.sin(x)

w = [0.5, -1.2, 0.3]
b = [0.0, 0.2, -0.1]
w1 = [[0.2, -0.5, 0.3],
      [1.0, -0.7, 0.8],
      [0.4, 0.6, -0.9]]
b1 = [0.0, 0.2, -0.1]
w2 = [[0.9, -0.2, 0.4],
      [-0.6, 0.3, 0.7],
      [0.1, 0.5, -0.8]]
b2 = [0.05, -0.05, 0.1]
w3 = [[0.9, -0.2, 0.4],
      [-0.6, 0.3, 0.7],
      [0.1, 0.5, -0.8]]
b3 = [0.0, 0.1, -0.1]
w4 = [0.3, -0.7, 0.5]
b4 = 0.0

def forward(x, w, b, w1, b1, w2, b2, w3, b3, w4, b4):
    layerdata1=[]
    for i in range(len(w)):
        layerdata1.append(np.tanh(w[i]*x+b[i]))

    layerdata2=[]
    for j in range(len(w1)):
        store=0
        for k in range(len(layerdata1)):
            store+=layerdata1[k]*w1[j][k]
        store+=b1[j]
        layerdata2.append(np.tanh(store))

    layerdata3=[]
    for l in range(len(w2)):
        store=0
        for m in range(len(layerdata2)):
            store+=layerdata2[m]*w2[l][m]
        store+=b2[l]
        layerdata3.append(np.tanh(store))
    layerdata4=[]
    for n in range(len(w3)):
        store=0
        for o in range(len(layerdata3)):
            store+=layerdata3[o]*w3[n][o]
        store+=b3[n]
        layerdata4.append(np.tanh(store))

    output=0
    for p in range(len(w4)):
        output+=layerdata4[p]*w4[p]
    output+=b4
    return output,layerdata1,layerdata2,layerdata3,layerdata4

def train(x, y, w, b, w1, b1, w2, b2, w3, b3, w4, b4, lr=0.01, epochs=1000):
    for _ in range(epochs):
        output, laydat1, laydat2, laydat3, laydat4 = forward(x, w, b, w1, b1, w2, b2, w3, b3, w4, b4)
        dydx=0
        for i in range(len(w)):
            dydx+=w[i]*(1-laydat1[i]**2)

        residual=dydx+output-(np.sin(x)+np.cos(x))
        dLphysdy=2*residual
        Loss=(residual**2)+0.5*(output-y)**2
        dLdatady=output-y
        lamphys=0.8
        lamdat=1-lamphys

        dLout=lamdat*dLdatady+lamphys*dLphysdy

        dLlaydat4=[dLout*w4[p]*(1-laydat4[p]**2) for p in range(len(w4))]

        dLlaydat3=[]
        for o in range(len(laydat3)):
            s=0
            for n in range(len(laydat4)):
                s+=dLlaydat4[n]*w3[n][o]
            dLlaydat3.append(s*(1-laydat3[o]**2))

        dLlaydat2=[]
        for o in range(len(laydat2)):
            s=0
            for n in range(len(laydat3)):
                s+=dLlaydat3[n]*w2[n][o]
            dLlaydat2.append(s*(1-laydat2[o]**2))

        dLlaydat1=[]
        for o in range(len(laydat1)):
            s=0
            for n in range(len(laydat2)):
                s+=dLlaydat2[n] * w1[n][o]
            dLlaydat1.append(s*(1-laydat1[o]**2))

        for i in range(len(w)):
            w[i]-=lr*dLlaydat1[i]*x
            b[i]-=lr*dLlaydat1[i]

        for i in range(len(w1)):
            for j in range(len(w1[0])):
                w1[i][j]-=lr*dLlaydat2[i]*laydat1[j]
            b1[i]-=lr*dLlaydat2[i]

        for i in range(len(w2)):
            for j in range(len(w2[0])):
                w2[i][j]-=lr*dLlaydat3[i]*laydat2[j]
            b2[i]-=lr*dLlaydat3[i]

        for i in range(len(w3)):
            for j in range(len(w3[0])):
                w3[i][j]-=lr*dLlaydat4[i]*laydat3[j]
            b3[i]-=lr*dLlaydat4[i]

        for i in range(len(w4)):
            w4[i]-=lr*dLout*laydat4[i]
        b4-=lr*dLout

    return w, b, w1, b1, w2, b2, w3, b3, w4, b4, Loss

for m in range(1000):
    for xi, yi in zip(x, y):
        w, b, w1, b1, w2, b2, w3, b3, w4, b4, Loss = train(xi, yi,w, b, w1, b1, w2, b2, w3, b3, w4, b4)


ypred = []
xvaluesforplot = np.linspace(-10,10,20)

for xnew in xvaluesforplot:
    ypred.append(float(forward(xnew, w, b, w1, b1, w2, b2, w3, b3, w4, b4)[0]))

plt.scatter(xvaluesforplot, ypred, color='red')
plt.plot(x,y)
plt.show()
