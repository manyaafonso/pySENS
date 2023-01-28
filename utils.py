# Autores: Manya V Afonso, original MATLAB version by: Manya V Afonso, José M Bioucas Dias, Mário A T Figueiredo
# Instituto de Telecomunicações, Instituto Superior Técnico, Lisboa, Portugal

import numpy as np
import pywt


def myNorm(x):
    return np.sqrt( (x**2).sum() )

def myNorm_l1(x):
    return np.sum( np.abs(x) )

def addGaussianNoise(x, SNRdB):
    Ps = (x**2).sum()/x.size
    sigma = np.sqrt( ((x-x.mean())**2).sum()/( x.size*(10**(SNRdB/10))  ) )
    y = x + sigma * np.random.normal(size=x.shape)
    return y

def myISNR(x0,y,x):
    ISNR = 10*np.log10( ((x0-y)**2).sum()/((x0-x)**2).sum()  )
    return ISNR

def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

def gaussian_kernel(size, sigma=1, verbose=False):
 
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_1D = kernel_1D/kernel_1D.sum()
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
 
    kernel_2D *= 1.0 / kernel_2D.max()
 
    if verbose:
        plt.imshow(kernel_2D, interpolation='none',cmap='gray')
        plt.title("Kernel")
        plt.show()
 
    return kernel_2D


def hard(x,T):
    return np.multiply(x, np.abs(x)>=T )

def soft(x,T):
    if np.sum(np.abs(T)) ==0:
        y = x
    else:
        z = np.abs(x) - T
        y = np.multiply( z, z>0 )
        y = np.multiply( np.divide( y, (y+T) ) , x)
    return (y)

def TVdiffs(u):
    tmp = np.row_stack( (u[-1,:],u[:-1,:]) )
    dux = u - tmp

    tmp = np.column_stack( (u[:,-1],u[:,:-1]) )
    duy = u - tmp
    
    return dux, duy

def myNeighbors(u,i,j):
    nb = np.array([ u[i-1,j], u[i,j-1], u[i+1,j], u[i,j+1] ])
    return nb

def myNeighborsTensor(u):
    tmp1 = np.row_stack( (u[-1,:],u[:-1,:]) )
    tmp2 = np.column_stack( (u[:,-1],u[:,:-1]) )
    tmp3 = np.row_stack( (u[1:,:],u[0,:]) )
    tmp4 = np.column_stack( (u[:,1:],u[:,0]) )
    
    nbMat = np.array( (tmp1, tmp2, tmp3, tmp4) )
    return nbMat
    
def DivergenceIm(p1,p2):
    z = p2[:,1:-1] - p2[:,:-2]
    v = np.column_stack((p2[:,0],z,-p2[:,-1]))
    
    z = p1[1:-1,:] - p1[:-2,:]
    u = np.row_stack((p1[0,:],z,-p1[-1,:]))

    divp = v + u
    return divp

def GradientIm(u):
    z = u[1:, :] - u[:-1,:]
    dux = np.row_stack((z,np.zeros(z.shape[1])))

    z = u[:, 1:] - u[:,:-1]
    duy = np.column_stack((z,np.zeros(z.shape[0])))
    
    return dux, duy

def chambolle_prox_TV_stop(g,alpha,MaxIter=10,tol=1e-2,px=np.array([0]),py=np.array([0])):
    if (px.shape != g.shape)or(py.shape != g.shape):
        px=np.zeros(g.shape)
        py=np.zeros(g.shape)
    tau = 0.249
    cont = True
    k = 0

    while cont:
        k = k + 1
        divp = DivergenceIm(px,py)
        u = divp - g/alpha
        upx,upy = GradientIm(u)
        tmp = np.sqrt(upx**2 + upy**2)
        err = np.sqrt( ((tmp*px-upx)**2 + (tmp*py-upy)**2).sum() )
        px = (px + tau * upx)/(1 + tau * tmp)
        py = (py + tau * upy)/(1 + tau * tmp)
        cont = (k<MaxIter) and (err>tol)

    f = g - alpha*DivergenceIm(px,py)
    return f,px,py

def denoiseTV(g,alpha,MaxIter=10,tol=1e-2,px=np.array([0]),py=np.array([0])):
    f,_,_ = chambolle_prox_TV_stop(g,alpha,MaxIter,tol,px,py)
    return f
    
def TVnorm(x):
    dh, dv = GradientIm(x)
    J = (np.sqrt(dh**2 + dv**2)).sum()
    return J

def TVnorm_ani(x):
    dh, dv = GradientIm(x)
    J = (np.sqrt(dh**2 + dv**2)).sum()
    return J

def cshift(x,L):
    N = len(x)
    y = np.zeros(x.shape)
    
    if (L == 0):
        return x
    
    if (L>0):
        y[L:] = x[:N-L]
        y[:L] = x[N-L:N]
    else:
        L = -L
        y[:N-L] = x[L:]
        y[N-L:N] = x[:L]
    
    return y

# transformada de wavelet, para trabalhar com matrizes.

def wrapper_swt2(img, wavelet_str, num_levels):
    coeffs = pywt.swt2(img, wavelet_str, num_levels)
    list_mat = []

    for ind_level in range(0,num_levels):
        list_mat.append( coeffs[ind_level][0] )
        for ch in range(0,3):
            list_mat.append( coeffs[ind_level][1][ch] )

    X = np.hstack([ list_mat ] )
    X = X.reshape(X.shape[0]*X.shape[1],X.shape[2])
    return X

def wrapper_iswt2(X, wavelet_str, num_levels):
    
    rows = int( X.shape[0]/(4*num_levels) )
    cols = X.shape[1]
    coeffs = [ () for _ in range(num_levels) ]

    for level in range(0, num_levels):
        cl = X[ 4*rows*level:4*rows*(level+1), : ]
        current_tuple = ( cl[:rows,:], ( cl[rows:2*rows,:], cl[2*rows:3*rows,:], cl[3*rows:4*rows,:]  ) )
        coeffs[level] = current_tuple

    y = pywt.iswt2(coeffs, wavelet_str)
    return y

def cshift(x,L):
    N = len(x)
    y = np.zeros(x.shape)
    
    if (L == 0):
        return x
    
    if (L>0):
        y[L:] = x[:N-L]
        y[:L] = x[N-L:N]#(N-L+1:N);
    else:
        L = -L
        y[:N-L] = x[L:]
        y[N-L:N] = x[:L]
    
    return y

def make_blur_kernel(kernel_length, img_size, blur_type=0):
    h_1d = np.zeros(img_size,)
    if blur_type == 0:
        # uniforme
        h_1d[:kernel_length] = 1
    else:
        # gaussiana
        h_1d[:kernel_length] = np.divide( np.ones(kernel_length,),\
                                         (np.array( range(0,kernel_length) )-((kernel_length-1)/2.0) )**2+1 )
    
    h_1d = h_1d/h_1d.sum()
    h_1d = cshift(h_1d,int(-(kernel_length-1)/2) )
    h_blur = np.outer(h_1d.transpose(), h_1d)
    return h_blur