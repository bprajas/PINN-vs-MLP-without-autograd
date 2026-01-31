import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0.01,1.5,50)
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
    d1=[]
    for i in range(len(w)):
        z=w[i]*x+b[i]
        h=np.tanh(z)
        layerdata1.append(h)
        d1.append((1-h*h)*w[i])

    layerdata2=[]
    d2=[]
    for j in range(len(w1)):
        z=0
        dzdx=0
        for i in range(len(layerdata1)):
            z+=layerdata1[i]*w1[j][i]
            dzdx+=d1[i]*w1[j][i]
        z+=b1[j]
        h=np.tanh(z)
        layerdata2.append(h)
        d2.append((1-h*h)*dzdx)

    layerdata3=[]
    d3=[]
    for k in range(len(w2)):
        z=0
        dzdx=0
        for j in range(len(layerdata2)):
            z+=layerdata2[j]*w2[k][j]
            dzdx+=d2[j]*w2[k][j]
        z+=b2[k]
        h=np.tanh(z)
        layerdata3.append(h)
        d3.append((1-h*h)*dzdx)

    layerdata4=[]
    d4=[]
    for l in range(len(w3)):
        z=0
        dzdx=0
        for k in range(len(layerdata3)):
            z+=layerdata3[k]*w3[l][k]
            dzdx+=d3[k]*w3[l][k]
        z+=b3[l]
        h=np.tanh(z)
        layerdata4.append(h)
        d4.append((1-h*h)*dzdx)

    output=0
    dy_dx=0
    for p in range(len(w4)):
        output+=layerdata4[p]*w4[p]
        dy_dx+=w4[p]*d4[p]
    output+=b4

    return output,dy_dx,layerdata1,layerdata2,layerdata3,layerdata4

def train(x, y, w, b, w1, b1, w2, b2, w3, b3, w4, b4, lr=0.001):
    output,dy_dx,laydat1,laydat2,laydat3,laydat4 = forward(
        x, w, b, w1, b1, w2, b2, w3, b3, w4, b4
    )

    target_dy = np.sin(x)+np.cos(x)
    residual = (y+dy_dx) - target_dy

    dL_out = (output-y)
    dL_dydx = 0.6*2*residual

    dL_laydat4=[]
    for i in range(len(laydat4)):
        dL_laydat4.append(
            dL_out*w4[i]*(1-laydat4[i]**2)
        )

    dL_laydat3=[]
    for o in range(len(laydat3)):
        s=0
        for n in range(len(laydat4)):
            s+=dL_laydat4[n]*w3[n][o]
        dL_laydat3.append(s*(1-laydat3[o]**2))

    dL_laydat2=[]
    for o in range(len(laydat2)):
        s=0
        for n in range(len(laydat3)):
            s+=dL_laydat3[n]*w2[n][o]
        dL_laydat2.append(s*(1-laydat2[o]**2))

    dL_laydat1=[]
    for o in range(len(laydat1)):
        s=0
        for n in range(len(laydat2)):
            s+=dL_laydat2[n]*w1[n][o]
        dL_laydat1.append(s*(1-laydat1[o]**2))

    for i in range(len(w)):
        w[i]-=lr*dL_laydat1[i]*x
        b[i]-=lr*dL_laydat1[i]

    for i in range(len(w1)):
        for j in range(len(w1[0])):
            w1[i][j]-=lr*dL_laydat2[i]*laydat1[j]
        b1[i]-=lr*dL_laydat2[i]

    for i in range(len(w2)):
        for j in range(len(w2[0])):
            w2[i][j]-=lr*dL_laydat3[i]*laydat2[j]
        b2[i]-=lr*dL_laydat3[i]

    for i in range(len(w3)):
        for j in range(len(w3[0])):
            w3[i][j]-=lr*dL_laydat4[i]*laydat3[j]
        b3[i]-=lr*dL_laydat4[i]

    for i in range(len(w4)):
        w4[i]-=lr*dL_out*laydat4[i]
    b4-=lr*dL_out

    return w,b,w1,b1,w2,b2,w3,b3,w4,b4

for _ in range(12000):
    for xi, yi in zip(x,y):
        w,b,w1,b1,w2,b2,w3,b3,w4,b4 = train(
            xi, yi, w,b,w1,b1,w2,b2,w3,b3,w4,b4
        )

y_pred=[]
x_values_for_plot=np.linspace(0.05,1.5,50)

for x_new in x_values_for_plot:
    y_pred.append(float(forward(x_new,w,b,w1,b1,w2,b2,w3,b3,w4,b4)[0]))

plt.scatter(x_values_for_plot,y_pred)
plt.plot(x,y)
plt.show()
