# Authors: Manya V Afonso, original MATLAB version by: Manya V Afonso, João Miguel R Sanches
# Instituto de Sistemas e Robótica, Instituto Superior Técnico, Lisboa, Portugal

import numpy as np
import time
from utils import *

class blindInpaint(object):
    
    # inicialização e verificação dos dados observados e operadores.
    def __init__(self, y=None, model='gaussian', tau1=1, tau2=1,\
                 bias=1e-3, cont_tau1 = None, cont_tau2 = None, tol=1e-4,thr_mask=0.5,\
                 isTVinitialization=0, TViters=5):
        
        if (y is None):
            raise ValueError('input y needs to be specified.')
        else:
            self.y = y
        
        if ( model is None ):
            self.model = 'gaussian'
        else:
            self.model = model
            
        self.isTVinitialization = isTVinitialization
        if (tau1 is None):
            self.tau1 = 1
        else:
            self.tau1 = tau1
        if ( tau2 is None ):
            self.tau2 = 1
        else:
            self.tau2 = tau2
        if ( bias is None ):
            self.bias = 1e-3
        else:
            self.bias = bias
        if ( thr_mask is None ):
            self.thr_mask = 0.5
        else:
            self.thr_mask = thr_mask
            
        self.tol = tol
        self.TViters = TViters
        
        if (cont_tau1 is None):
            self.cont_tau1 = lambda x: x
        else:
            self.cont_tau1 = cont_tau1
            
        if (cont_tau2 is None):
            self.cont_tau2 = lambda x: x
        else:
            self.cont_tau2 = cont_tau2
        self.verbose = 0
        self.maxiters = 500
        self.chambolleit = TViters
        self.last_estimate = self.y
        self.stopCriterion = 1

        #fim da função init
        
    # função para empacotar os modelos estatísticos
    #def predict()
        
        
    # função para aplicar o algoritmo de blind inpainting dado o vector observado e os restantes parâmetros já definidos,
    # admitindo o modelo Gaussiano
    def blindInpaint_gaussian(self, y, tau1, tau2,\
                    bias=1e-3,\
                    x_true = None,\
                    mask_true = None,\
                    maxiters = 500,\
                    chambolleit = 5,\
                    verbose = 0,\
                    stopcriterion = 1,\
                    cont_tau1 = None,\
                    cont_tau2 = None,\
                    tol=1e-4,\
                    thr_mask=0.5):
        compute_errx = 0
        compute_errm = 0

        if not(y is None):
            self.y = y
        if not(tau1 is None):
            self.tau1 = tau1
        if not(tau2 is None):
            self.tau2 = tau2
        
        if not(bias is None):
            self.bias = bias
        if not(maxiters is None):
            self.maxiters = maxiters
        if not(chambolleit is None):
            self.chambolleit = chambolleit
        if not(tau2 is None):
            self.tau2 = tau2
        if not(stopcriterion is None):
            self.stopcriterion = stopcriterion
        if not(tol is None):
            self.tol = tol
        if not(thr_mask is None):
            self.thr_mask = thr_mask
        
        if not(cont_tau1 is None):
            self.cont_tau1 = cont_tau1
        if not(cont_tau2 is None):
            self.cont_tau2 = cont_tau2
            
        if not( x_true is None ):
            compute_errx = 1

        if not( mask_true is None ):
            compute_errm = 1

        M,N = y.shape

        # adicionar o valor de 'bias' para garantir que o logaritmo não seja aplicado sobre 0
        self.y = self.y + self.bias
        g = np.log(self.y)

        # o relógio começa a contar a partir daqui
        t0 = time.time()
        times = [ 0 ]

        u = np.zeros( (g.shape) )
        v = np.ones( (M,N) )*(np.log10(bias)-1)

        x_hat = np.exp(u)
        x_hat_prev = x_hat
        mask_est = np.exp(v)

        if compute_errx:
            err_x = [ myNorm(x_true-x_hat)**2/(M*N) ]

        if compute_errm:
            err_mask = [ myNorm_l1(mask_true-mask_est)/(M*N) ]

        obj = [ myNorm(g-u-v)**2 + tau1*TVnorm(u) + tau2*np.sum(v!=0) ]


        for it in range(0, maxiters):

            u,_,_ = chambolle_prox_TV_stop(g-v,self.tau1,self.chambolleit)
            v = hard(g-u,self.tau2)

            x_hat = np.exp(u)
            mask_est = np.exp(v)

            times.append( time.time()-t0 )
            obj.append( myNorm(g-u-v)**2 + self.tau1*TVnorm(u) + self.tau2*np.sum(v!=0) )

            if compute_errx:
                err_x.append( myNorm(x_true -x_hat )**2/(M*N) )

            if compute_errm:
                err_mask.append( myNorm_l1(mask_true-mask_est)/(M*N) )

            if (it>0):
                if stopcriterion==1:
                    criterion = np.abs( (obj[-1]-obj[-2])/obj[-2] )
                elif stopcriterion==2:
                    criterion = myNorm(x_hat-x_hat_prev)/myNorm(x_hat)
                elif stopcriterion==3:
                    criterion = obj[-1]
                else:
                    raise ValueError('Invalid stopping criterion!')

                if criterion < tol:
                    if verbose:
                        print('Convergence reached.')
                    break
                # fim do bloco das condições de paragem

            x_hat_prev = x_hat

            if callable(self.cont_tau1):
                self.tau1 = self.cont_tau1(self.tau1)
            if callable(cont_tau2):
                self.tau2 = self.cont_tau2(self.tau2)

            # fim do processo iterativo

        # conversão para valores reais, para evitar componentes imaginários devido ás operações logaritmicas
        x_hat = np.real(x_hat)

        # os pixeis da máscara têm valores entre 0 e 1, aplicamos o limiar para serem valores binarios.
        mask_est = (mask_est>self.thr_mask)

        if compute_errx and compute_errm:
            return x_hat, mask_est, obj, times, err_x, err_mask
        elif compute_errx:
            return x_hat, mask_est, obj, times, err_x
        elif compute_errm:
            return x_hat, mask_est, obj, times, err_mask
        else:
            return x_hat, mask_est, obj, times

        # fim da função