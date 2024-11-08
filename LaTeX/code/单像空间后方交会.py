import numpy as np

#摄影机主距、内方位元素、相片比例尺
fk=153.24
m=50000.0
x0=0.0
y0=0.0
center=np.matrix([x0,y0])

#地面点坐标
mat_ground=np.matrix([[36589.41,25273.32,2195.17],
                  [37631.08,31324.51,728.69],
                  [39100.97,24934.98,2386.50],
                  [40426.54,30319.81,757.31]])
#像点坐标
mat_photo=np.matrix([[-86.15,-68.99],
                  [-53.40,82.21],
                  [-14.78,-76.63],
                  [10.46,64.43]])

#外方位元素
PhotoCenter=np.sum(mat_ground,axis=0)/4 #计算的是地面点的中心

PhotoCenter[0,2]=fk*m/1000.0 #摄影机的高度
H=PhotoCenter[0,2]

print('PhotoCenter:',PhotoCenter)

#外方位角元素phi omega kapa
phi=omega=ka=0 #初始化
Angle=np.matrix([phi,omega,ka])
#设置改正数组，Xs，Ys，Zs，phi,omega,kapa
Iteration=0 #迭代次数
while True:
    Iteration+=1
    print('这是第',Iteration,'次迭代~')
    #计算旋转矩阵
    R=np.matrix([[np.cos(phi)*np.cos(ka)-np.sin(phi)*np.sin(omega)*np.sin(ka)    , np.cos(omega)*np.sin(ka)    , np.sin(phi)*np.cos(ka)+np.cos(phi)*np.sin(omega)*np.sin(ka)],
                [-np.cos(phi)*np.sin(ka)-np.sin(phi)*np.sin(omega)*np.cos(ka)   , np.cos(omega)*np.cos(ka)    , -np.sin(phi)*np.sin(ka)+np.cos(phi)*np.sin(omega)*np.cos(ka)],
                [-np.sin(phi)*np.cos(omega)                                     , -np.sin(omega)              , np.cos(phi)*np.cos(omega)]])

    a1, b1, c1 = R[0, 0], R[0, 1], R[0, 2]
    a2, b2, c2 = R[1, 0], R[1, 1], R[1, 2]
    a3, b3, c3 = R[2, 0], R[2, 1], R[2, 2]

    Ra=np.matrix([[a1,b1,c1],
                 [a2,b2,c2],
                 [a3,b3,c3]])
    print('Ra:', Ra)

    #常数项lx，ly
    lxy=np.zeros_like(mat_photo)
    #Zb
    Zb=np.zeros((mat_ground.shape[0],1))
    #A系数矩阵
    A=np.zeros((2*mat_ground.shape[0],6))

    #共线条件方程
    for i in range(mat_photo.shape[0]):
        lx=mat_photo[i,0]+fk*(a1 * (mat_ground[i, 0] - PhotoCenter[0, 0]) + b1 * (mat_ground[i, 1] - PhotoCenter[0, 1]) + c1 * (
                        mat_ground[i, 2] - PhotoCenter[0, 2])) / (
                        a3 * (mat_ground[i, 0] - PhotoCenter[0, 0]) + b3 * (mat_ground[i, 1] - PhotoCenter[0, 1]) + c3 * (
                            mat_ground[i, 2] - PhotoCenter[0, 2]))
        ly=mat_photo[i,1]+fk*(a2 * (mat_ground[i, 0] - PhotoCenter[0, 0]) + b2 * (mat_ground[i, 1] - PhotoCenter[0, 1]) + c2 * (
                        mat_ground[i, 2] - PhotoCenter[0, 2])) / (
                         a3 * (mat_ground[i, 0] - PhotoCenter[0, 0]) + b3 * (mat_ground[i, 1] - PhotoCenter[0, 1]) + c3 * (
                             mat_ground[i, 2] - PhotoCenter[0, 2]))

        lxy[i,0]=lx
        lxy[i,1]=ly
        x=mat_photo[i,0]
        y=mat_photo[i,1]

        Zb[i,0]=a3*(mat_ground[i,0]-PhotoCenter[0,0])+b3*(mat_ground[i,1]-PhotoCenter[0,1])+c3*(mat_ground[i,2]-PhotoCenter[0,2])
        A[2 * i, 0] = (a1 * fk + a3 * x) / Zb[i, 0]
        A[2 * i, 1] = (b1 * fk + b3 * x) / Zb[i, 0]
        A[2 * i, 2] = (c1 * fk + c3 * x) / Zb[i, 0]
        A[2 * i, 3] = y * np.sin(omega) - (x * (x * np.cos(ka) - y * np.sin(ka)) / fk + fk * np.cos(ka)) * np.cos(omega)
        A[2 * i, 4] = -fk * np.sin(ka) - x * (x * np.sin(ka) + y * np.cos(ka)) / fk
        A[2 * i, 5] = y
        A[2 * i + 1, 0] = (a2 * fk + a3 * y) / Zb[i, 0]
        A[2 * i + 1, 1] = (b2 * fk + b3 * y) / Zb[i, 0]
        A[2 * i + 1, 2] = (c2 * fk + c3 * y) / Zb[i, 0]
        A[2 * i + 1, 3] = -x * np.sin(omega) - (y * (x * np.cos(ka) - y * np.sin(ka) )/ fk - fk * np.sin(ka)) * np.cos(omega)
        A[2 * i + 1, 4] = -fk * np.cos(ka) - y * (x * np.sin(ka) + y * np.cos(ka)) / fk
        A[2 * i + 1, 5] = -x

    l=np.matrix(lxy).reshape(8,1) #求转置
    print('Zb:',Zb)
    print('A:',A)
    print('l:',l)
    A=np.matrix(A)#将A转换为矩阵
    #最小二乘法求解
    mat1=((A.T*A).I*A.T*l)
    delta=mat1
    print('delta:',delta)
    dphi, domega, dka = delta[3, 0], delta[4, 0], delta[5, 0]
    phi += dphi
    omega += domega
    ka += dka
    PhotoCenter[0, 0] += delta[0, 0]
    PhotoCenter[0, 1] += delta[1, 0]
    PhotoCenter[0, 2] += delta[2, 0]
    H=PhotoCenter[0,2]
    print('PhotoCenter:',PhotoCenter)
    print('phi:',phi)
    print('omega:',omega)
    print('ka:',ka)
    #迭代停止条件
    if np.abs(delta).max()< 3e-5:
        print('delta:', delta)
        print('dphi', dphi)
        print('domega', domega)
        print('dka', dka)
        break

print('phi:',phi)
print('omega:',omega)
print('ka:',ka)
print('PhotoCenter:',PhotoCenter)
# 计算残差向量v
v = l - A * delta
#残差的中误差计算
n = 2 * mat_ground.shape[0]  #观测方程的数量
u = 6  #未知数的数量
sigma = np.sqrt((v.T * v)[0, 0] / (n - u))  #中误差计算

print('残差向量v:',v)
print('中误差sigma:',sigma)
