# Python implementation for the Split Augmented Lagrangian and Shrinkage Algorithm (SALSA)
# Authors: Manya V Afonso, original MATLAB version by: Manya V Afonso, José M Bioucas Dias, Mário A T Figueiredo
# Instituto de Telecomunicações, Instituto Superior Técnico, Lisboa, Portugal

import numpy as np
import time
from utils import TVnorm, chambolle_prox_TV_stop, soft

# Classe para criar um objecto para empacotar os dados observados, operador linear/matriz do modelo de observação, e restantes parâmetros.
# TO DO: generalizar para definir outros algoritmos C-SALSA, CoRAL, etc no mesmo objecto?

class SALSA(object):

    # inicialização e verificação dos dados observados e operadores.
    def __init__(self, y=None, A=None, AT=None, invLS=None, tau=1, mu=0.1, Phi=None, Psi=None, P=None, PT=None,\
                 isTVinitialization=0, TViters=5):
        
        if (y is None):
            raise ValueError('input y needs to be specified.')
        else:
            self.y = y
            
        self.psi_ok = 0
        self.Psi = Psi
        self.isinvLS = 0
        self.definedP = 0
        self.definedPT = 0
        self.definedA = 0
        self.definedAT = 0

        if not( invLS is None ):
            self.isinvLS = 1
            self.invLS = invLS

        if not( P is None ):
            self.definedP = 1
            self.P = P

        if not( PT is None ):
            self.definedPT = 1
            self.PT = PT

        if not( A is None ):
            self.definedA = 1
            self.A = A
            
        if not( AT is None ):
            self.definedAT = 1
            self.AT = AT
            
        if not( Psi is None ):
            self.Psi = Psi
            
        if not( Phi is None ):
            self.Phi = Phi
        
        self.isTVinitialization = isTVinitialization
        if (tau is None):
            self.tau = tau
        else:
            self.tau = 1
        if ( mu is None ):
            self.mu = mu
        else:
            self.mu = 0.1
            
        self.x_init = 2
        self.tolA = 1e-3
        self.verbose = 0
        self.MaxIt = 500
        self.TViters = TViters
        self.compute_mse = 0
        self.numA = 0
        self.numAt = 0
        self.last_estimate = self.y
        self.stopCriterion = 1

        # a verificar se os parâmetros P e PT foram ambos definidos, e que as dimensionalidades são compatíveis.
        if np.logical_xor( self.definedP,self.definedPT ):
            raise ValueError('If you give P you must also give PT, and vice versa.')
            
        if (self.definedP == 0):
            self.P = lambda x: x
            self.PT = lambda x: x
            
        # se a A for uma função, AT também devia ser definida.
        if ( callable(self.A) and ( not( callable(self.AT) ) ) ):
            raise ValueError('The function handle for transpose of A is missing')
            
        # se a A for uma matrix, definimos funções para multiplicações por A ou AT, para compatibilidade com o caso de serem funções.
        if not( callable(self.A) ):

            if not( (self.y.shape[0] == self.A.shape[0] ) and ( self.y.shape[1] == 1) ):
                raise ValueError('For a MxN observation matrix A, the measurement vector y must be a length M vector.')

            matA = A
            matAT = np.transpose(A)
            self.AT = lambda x: matAT*x
            self.A = lambda x: matA*x
    
        # a partir daqui A e AT são funções.

        self.ATy = self.AT(self.y)
        self.numAt = self.numAt + 1

        # a verificar se o operador invLS foi definido.
        if callable(self.A):

            if ( not(self.isinvLS) or not( callable(self.invLS) ) ):
                raise ValueError('(A^T A + \mu I)^(-1) must be specified as a function handle.')
            else:
                dummy = self.invLS(self.ATy, self.mu)
                if not( (dummy.shape[0]==self.ATy.shape[0]) and ( dummy.shape[1]==self.ATy.shape[1] ) ):
                    raise ValueError('Specified function handle for solving the LS step does not seem compatible with the specified A and AT.')
        else: # A foi definida como uma matrix
            M,N = matA.shape
            if ( M > N ):
                ATA = np.transpose(matA)*matA 
                self.invLS = lambda x, mu: ( np.linalg.inv( ATA+mu*np.identity(N) ) )*x
            else:
                AAT = matA*np.transpose(matA)
                self.invLS = lambda x, mu: ( AT*np.linalg.inv(mu*np.identity(M)+AAT)*A )*x
            

        # Se a Psi foi definida, verificar se é uma função com dimensionalidade compatível.
        if callable(self.Psi):
            if self.isTVinitialization:
                print('Warning: user specified Phi and Psi will not be used as TV with initialization flag has been set to 1.')
            else:
                try:
                    dummy = Psi(PT(self.ATy),self.tau);
                    self.psi_ok = 1
                except:
                    raise ValueError('Something is wrong with function handle for psi')
        else: # se não foi definida, usamos o soft threshold
            self.Psi = lambda x,tau: soft(x,tau)

        # Se a Psi foi definida, precisamos também da Phi
        if (self.psi_ok == 1):
            try:
                self.Phi
            except:
                raise ValueError('If you give Psi you must also give Phi')
            else:
                # a Phi é definida
                if callable(self.Phi):
                    try:  # a verificar a dimensionalidade da Phi
                        dummy = self.Phi( self.PT(self.ATy) )
                    except:
                        raise ValueError('Something is wrong with function handle for phi')
                else:
                    raise ValueError('Phi does not seem to be a valid function handle')
        else:  # caso não foram definidas nem uma nem outra, utilizamos a norma l1.
            if not self.isTVinitialization:
                self.Phi = lambda x: np.sum( np.abs(x) )
            else:
                self.Phi = lambda x: TVnorm(x)
        #fim da função init

    # função para aplicar o algoritmo SALSA dado o vector observado e os restantes parâmetros já definidos.
    def predict(self, y=None, tau=1, mu = 0.1, stopCriterion=1, x_true=None, x_init=2, MaxIt=100, tolA=0.01, verbose=0):
        
        if not(stopCriterion is None):
            if not(stopCriterion in [1,2,3] ):
                raise ValueError('Unknown stopping criterion')
            self.stopCriterion = stopCriterion    
        
        if not( x_true is None ):
            self.compute_mse = 1
        else:
            self.compute_mse = 0
    
        if not(y is None):
            if y.shape == self.y.shape:
                self.y = y
                self.ATy = self.AT(y)
            else:
                raise ValueError('y dimensions are incompatible.')
                
        if not(tau is None):
            self.tau = tau
        if not( mu is None ):
            self.mu = mu
        if not(MaxIt is None):
            self.MaxIt = MaxIt
        if not(tolA is None):
            self.tolA = tolA
        if not(verbose is None):
            self.verbose = verbose
                
        if (x_init is None):
            x_init = self.x_init
            
        if np.isscalar(x_init):
            if x_init == 0: # zeros
                x = AT( np.zeros(self.y.shape) )
            elif x_init == 1: # aleatoriamente
                x = np.random.rand( self.ATy.shape )
            elif x_init == 2: # ATy
                x = self.ATy
            else: 
                raise ValueError('Unknown initialization option.')
        else:
            if not( (A(x_init)).shape == y.shape ):
                raise ValueError('Size of initial x is not compatible with A')
            else:
                x = x_init

        # se a x verdadeira foi dada, verificamos a dimensionalidade.
        if ( self.compute_mse and not( x_true.shape == x.shape )  ):
            raise ValueError('True x has incompatible size')
            
        PTx = self.PT(x)

        u = PTx
        bu = u
        threshold = self.tau/self.mu

        criterion = [1]
        distance = []
        mses = []

        # inicializar a função do custo.
        resid =  self.y-self.A(x)
        self.numA = self.numA + 1
        prev_f = 0.5*np.sum( resid**2 ) + self.tau*self.Phi(u)

        data_vector_size = np.product(x.shape)

        if self.verbose:
            print('Initial value of objective function = {:.2e}'.format(prev_f) )

        # o relógio começa a contar a partir daqui
        t0 = time.time()
        times = [ 0 ]
        objective = [ prev_f ]

        if self.compute_mse:
            mses.append( np.sum( (x-x_true)**2 )/data_vector_size )

        if self.isTVinitialization:
            pux = np.zeros(u.shape)
            puy = np.zeros(u.shape)

        for it in range(0,self.MaxIt):

            xprev = x

            if self.isTVinitialization:
                u,pux,puy = chambolle_prox_TV_stop(np.real(PTx-bu), threshold, self.TViters,\
                                                   tol=1e-2, px=pux, py=puy)
            else:
                u = self.Psi(PTx-bu, threshold)

            r = self.ATy + self.mu*self.P(u+bu)
            x = self.invLS(r, self.mu)
            PTx = self.PT(x)
            bu = bu + (u - PTx)
            resid =  y-self.A(x)
            self.numA = self.numA + 1
            objective.append( 0.5*np.sum(resid**2) + self.tau*self.Phi(u) )

            if self.compute_mse:
                err = x - x_true
                mses.append( np.sum(err**2)/data_vector_size )

            distance.append( np.sqrt( np.sum( (PTx-u)**2 ) )/np.sqrt( np.sum( PTx**2 ) + np.sum( u**2 ) ) )

            if (it>0):
                if stopCriterion == 1:
                    # diferença relativa da função de custo.
                    criterion.append( np.abs(objective[-1]-objective[-2])/objective[-2] )
                elif stopCriterion == 2:
                    # diferença relativa entre 2 consecutivos valores da estimativa
                    criterion.append( myNorm(x-xprev)/myNorm(x) )
                elif stopCriterion == 3:
                    # valor alvo da função de custo
                    criterion.append( objective[-1] )
                else:
                    if it == 1:
                        print('Unknown stopping criterion.')
                    continue

                if ( criterion[-1] < self.tolA ):
                    if self.verbose:
                        out_str = 'iter= {:d}, obj= {:.2e}, stop criterion= {:.2e}, (target= {:.2e})'.format(it,\
                                                                                                 objective[-1],\
                                                                                                 criterion[-1], tolA)
                        if self.compute_mse:
                            out_str = out_str + ', MSE = {:.2f}'.format(mses[-1])

                        print(out_str)
                    print('Convergence reached.')

                    times.append( time.time() - t0 )
                    break
            # fim do bloco das condições de paragem

            if self.verbose:
                out_str = 'iter= {:d}, obj= {:.2e}, stop criterion= {:.2e}, (target= {:.2e})'.format(it,objective[-1],\
                                                                                               criterion[-1], tolA)
                if self.compute_mse:
                    out_str = out_str + ', MSE = {:.2f}'.format(mses[-1])

                print(out_str)

            times.append( time.time() - t0 )
        # fim do processo iterativo

        if self.compute_mse:
            return x, objective, times, distance, self.numA, self.numAt, mses
        else:
            return x, objective, times, distance, self.numA, self.numAt
        
        # fim da função predict

###############################################################################################################
# Classe para criar um objecto para empacotar os dados observados, operador linear/matriz do modelo de observação, e restantes parâmetros.
# TO DO: generalizar para definir outros algoritmos C-SALSA, CoRAL, etc no mesmo objecto?

class CSALSA(object):

    # inicialização e verificação dos dados observados e operadores.
    def __init__(self, y=None, A=None, AT=None, invLS=None, mu1=0.1, mu2=0.1, sigma=0.01, Phi=None, Psi=None, P=None, PT=None,\
                 isTVinitialization=0, TViters=5, delta=1, epsilon=0):
        
        if (y is None):
            raise ValueError('input y needs to be specified.')
        else:
            self.y = y
            
        self.psi_ok = 0
        self.Psi = Psi
        self.isinvLS = 0
        self.definedP = 0
        self.definedPT = 0
        self.definedA = 0
        self.definedAT = 0

        if not( invLS is None ):
            self.isinvLS = 1
            self.invLS = invLS

        if not( P is None ):
            self.definedP = 1
            self.P = P

        if not( PT is None ):
            self.definedPT = 1
            self.PT = PT

        if not( A is None ):
            self.definedA = 1
            self.A = A
            
        if not( AT is None ):
            self.definedAT = 1
            self.AT = AT
            
        if not( Psi is None ):
            self.Psi = Psi
            
        if not( Phi is None ):
            self.Phi = Phi
        
        self.isTVinitialization = isTVinitialization
        if not(mu1 is None):
            self.mu1 = mu1
        else:
            self.mu1 = 0.1
        if not( mu2 is None ):
            self.mu2 = mu2
        else:
            self.mu2 = 0.1
        if not( sigma is None ):
            self.sigma = sigma
        else:
            self.sigma = 0.01
        if not( delta is None ):
            self.delta = delta
        else:
            self.delta = 1
        if not( epsilon is None ):
            self.epsilon = epsilon
        else:
            self.epsilon = 0
            
        self.x_init = 2
        self.tolA = 1e-3
        self.verbose = 0
        self.MaxIt = 500
        self.TViters = TViters
        self.compute_mse = 0
        self.numA = 0
        self.numAt = 0
        self.last_estimate = self.y
        self.stopCriterion = 3
        #self.delta = 1
        #self.epsilon = 0

        # a verificar se os parâmetros P e PT foram ambos definidos, e que as dimensionalidades são compatíveis.
        if np.logical_xor( self.definedP,self.definedPT ):
            raise ValueError('If you give P you must also give PT, and vice versa.')
            
        if (self.definedP == 0):
            self.P = lambda x: x
            self.PT = lambda x: x
            
        # se a A for uma função, AT também devia ser definida.
        if ( callable(self.A) and ( not( callable(self.AT) ) ) ):
            raise ValueError('The function handle for transpose of A is missing')
            
        # se a A for uma matrix, definimos funções para multiplicações por A ou AT, para compatibilidade com o caso de serem funções.
        if not( callable(self.A) ):

            if not( (self.y.shape[0] == self.A.shape[0] ) and ( self.y.shape[1] == 1) ):
                raise ValueError('For a MxN observation matrix A, the measurement vector y must be a length M vector.')

            matA = A
            matAT = np.transpose(A)
            self.AT = lambda x: matAT*x
            self.A = lambda x: matA*x
    
        # a partir daqui A e AT são funções.

        self.ATy = self.AT(self.y)
        self.numAt = self.numAt + 1

        # a verificar se o operador invLS foi definido.
        if callable(self.A):

            if ( not(self.isinvLS) or not( callable(self.invLS) ) ):
                raise ValueError('(A^T A + \mu I)^(-1) must be specified as a function handle.')
            else:
                dummy = self.invLS(self.ATy, self.mu1, self.mu2)
                if not( (dummy.shape[0]==self.ATy.shape[0]) and ( dummy.shape[1]==self.ATy.shape[1] ) ):
                    raise ValueError('Specified function handle for solving the LS step does not seem compatible with the specified A and AT.')
        else: # A foi definida como uma matrix
            M,N = matA.shape
            if ( M > N ):
                ATA = np.transpose(matA)*matA 
                self.invLS = lambda x, mu1, mu2: ( np.linalg.inv( ATA+(mu1+mu2)*np.identity(N) ) )*x
            else:
                AAT = matA*np.transpose(matA)
                self.invLS = lambda x, mu1, mu2: ( AT*np.linalg.inv( (mu1+mu2)*np.identity(M)+AAT)*A )*x
            

        # Se a Psi foi definida, verificar se é uma função com dimensionalidade compatível.
        if callable(self.Psi):
            if self.isTVinitialization:
                print('Warning: user specified Phi and Psi will not be used as TV with initialization flag has been set to 1.')
            else:
                try:
                    dummy = self.Psi(self.PT(self.ATy),self.tau);
                    self.psi_ok = 1
                except:
                    raise ValueError('Something is wrong with function handle for psi')
        else: # se não foi definida, usamos o soft threshold
            self.Psi = lambda x,tau: soft(x,tau)

        # Se a Psi foi definida, precisamos também da Phi
        if (self.psi_ok == 1):
            try:
                self.Phi
            except:
                raise ValueError('If you give Psi you must also give Phi')
            else:
                # a Phi é definida
                if callable(self.Phi):
                    try:  # a verificar a dimensionalidade da Phi
                        dummy = self.Phi( self.PT(self.ATy) )
                    except:
                        raise ValueError('Something is wrong with function handle for phi')
                else:
                    raise ValueError('Phi does not seem to be a valid function handle')
        else:  # caso não foram definidas nem uma nem outra, utilizamos a norma l1.
            if not self.isTVinitialization:
                self.Phi = lambda x: np.sum( np.abs(x) )
            else:
                self.Phi = lambda x: TVnorm(x)
        #fim da função init

    # função para aplicar o algoritmo SALSA dado o vector observado e os restantes parâmetros já definidos.
    def predict(self, y=None, mu1=0.1, mu2=0.1, sigma=0.01, delta=1, epsilon=0, stopCriterion=3, x_true=None, x_init=2, MaxIt=100, tolA=0.01, verbose=0):
        
        if not(stopCriterion is None):
            if not(stopCriterion in [1,2,3] ):
                raise ValueError('Unknown stopping criterion')
            self.stopCriterion = stopCriterion    
        
        if not( x_true is None ):
            self.compute_mse = 1
        else:
            self.compute_mse = 0
    
        if not(y is None):
            if y.shape == self.y.shape:
                self.y = y
                self.ATy = self.AT(y)
            else:
                raise ValueError('y dimensions are incompatible.')
                
        if not(mu1 is None):
            self.mu1 = mu1
        if not( mu2 is None ):
            self.mu2 = mu2
        if not( sigma is None ):
            self.sigma = sigma
        if not( delta is None ):
            self.delta = delta
        if not( epsilon is None ):
            self.epsilon = epsilon
            
        if not(MaxIt is None):
            self.MaxIt = MaxIt
        if not(tolA is None):
            self.tolA = tolA
        if not(verbose is None):
            self.verbose = verbose
                
        if (x_init is None):
            x_init = self.x_init
            
        if np.isscalar(x_init):
            if x_init == 0: # zeros
                x = AT( np.zeros(self.y.shape) )
            elif x_init == 1: # aleatoriamente
                x = np.random.rand( self.ATy.shape )
            elif x_init == 2: # ATy
                x = self.ATy
            else: 
                raise ValueError('Unknown initialization option.')
        else:
            if not( (A(x_init)).shape == y.shape ):
                raise ValueError('Size of initial x is not compatible with A')
            else:
                x = x_init

        # se a x verdadeira foi dada, verificamos a dimensionalidade.
        if ( self.compute_mse and not( x_true.shape == x.shape )  ):
            raise ValueError('True x has incompatible size')
            
        PTx = self.PT(x)

        u = np.zeros(PTx.shape)
        bu = np.zeros(u.shape)
        v = np.zeros(PTx.shape)
        bv = np.zeros(u.shape)
        
        tau = self.mu1/self.mu2
        Ny = np.product(y.shape)
        if (self.epsilon==0):
            self.epsilon = np.sqrt( Ny+8*np.sqrt(Ny) )*self.sigma

        distance1 = []
        distance2 = []
        mses = []

        # inicializar a função do custo.
        resid =  self.y-self.A(x)
        self.numA = self.numA + 1
        criterion = [ np.sqrt( np.sum( resid**2 ) ) ]
        distance1 = [ np.sqrt( np.sum( (self.A(x)-y-v)**2 ) ) ] # [ norm(Ax(:)-y(:)-v(:),2) ]
        distance2 = [ np.sqrt( np.sum( (PTx-u)**2 ) ) ] #[ norm(PTx(:)-u(:),2) ]

        data_vector_size = np.product(x.shape)

        if self.verbose:
            out_str = 'restriction 1 = {:.2e}, restriction 2 = {:.2e}, (criterion= {:.2e})'.format(distance1[-1],\
                                                                                                   distance2[-1],\
                                                                                                   criterion[-1])
            print(out_str)

        # o relógio começa a contar a partir daqui
        t0 = time.time()
        times = [ 0 ]
        objective = [ self.Phi(x) ]

        if self.compute_mse:
            mses.append( np.sum( (x-x_true)**2 )/data_vector_size )

        if self.isTVinitialization:
            pux = np.zeros(u.shape)
            puy = np.zeros(u.shape)

        for it in range(0,self.MaxIt):
            
            mu1inv = 1/self.mu1
            mu2inv = 1/self.mu2

            xprev = x
            
            r = self.mu1*self.P(u+bu) + self.mu2*self.AT(y+v+bv)
            self.numAt = self.numAt + 1
            
            x = self.invLS(r, self.mu1, self.mu2)
            PTx = self.PT(x)            

            if self.isTVinitialization:
                u,pux,puy = chambolle_prox_TV_stop(np.real(PTx-bu), mu1inv, self.TViters,\
                                                   tol=1e-2, px=pux, py=puy)
            else:
                u = self.Psi(PTx-bu, mu1inv)
                
            Ax = self.A(x)
            self.numA = self.numA + 1
            ve = Ax - y - bv
            n_ve = np.sqrt( np.sum( ve**2 ) ) #norm(ve(:));
            if n_ve <= self.epsilon:
                v = ve
            else:
                v =  ve/n_ve*self.epsilon

            bv = bv-(Ax-y-v)
            bu = bu-(PTx-u)
            
            resid =  self.y-self.A(x)
            criterion.append( np.sqrt( np.sum( resid**2 ) ) )
            distance1.append( np.sqrt( np.sum( (self.A(x)-y-v)**2 ) ) )
            distance2.append( np.sqrt( np.sum( (PTx-u)**2 ) ) )

            objective.append( self.Phi(x) )

            if self.compute_mse:
                err = x - x_true
                mses.append( np.sum(err**2)/data_vector_size )

            if (it>0):
                if stopCriterion == 1:
                    # diferença relativa da função de custo.
                    criterion.append( np.abs(objective[-1]-objective[-2])/objective[-2] )
                elif stopCriterion == 2:
                    # diferença relativa entre 2 consecutivos valores da estimativa
                    criterion.append( myNorm(x-xprev)/myNorm(x) )
                elif stopCriterion == 3:
                    # valor alvo da função de custo
                    criterion.append( objective[-1] )
                else:
                    if it == 1:
                        print('Unknown stopping criterion.')
                    continue

                if ( ( criterion[-1] < self.tolA ) and ( criterion[-1]<=self.epsilon) ):
                    if self.verbose:
                        out_str = 'restriction 1 = {:.2e}, restriction 2 = {:.2e}, (criterion= {:.2e})'.format(distance1[-1],\
                                                                                                   distance2[-1],\
                                                                                                   criterion[-1])
                        if self.compute_mse:
                            out_str = out_str + ', MSE = {:.2f}'.format(mses[-1])

                        print(out_str)
                    print('Convergence reached.')

                    times.append( time.time() - t0 )
                    break
            # fim do bloco das condições de paragem

            if self.verbose:
                
                out_str = 'restriction 1 = {:.2e}, restriction 2 = {:.2e}, (criterion= {:.2e})'.format(distance1[-1],\
                                                                                                   distance2[-1],\
                                                                                                   criterion[-1])
                
                if self.compute_mse:
                    out_str = out_str + ', MSE = {:.2f}'.format(mses[-1])

                print(out_str)

            times.append( time.time() - t0 )
        # fim do processo iterativo

        if self.compute_mse:
            return x, objective, times, criterion, distance1, distance2, self.numA, self.numAt, mses
        else:
            return x, objective, times, criterion, distance1, distance2, self.numA, self.numAt
        
        # fim da função predict

        
##############################################################################################################

# Classe para criar um objecto para empacotar os dados observados, operador linear/matriz do modelo de observação, e restantes parâmetros.
# TO DO: generalizar para definir outros algoritmos C-SALSA, CoRAL, etc no mesmo objecto?

class CoRAL(object):

    # inicialização e verificação dos dados observados e operadores.
    def __init__(self, y=None, A=None, AT=None, invLS=None, tau1=1, tau2=1,\
                 mu1=0.1, mu2=0.1, Phi1=None, Psi1=None, Phi2=None, Psi2=None,\
                 P1=None, P1T=None, P2=None, P2T=None, W=None, WT=None,\
                 isTVinitialization1=0, TViters1=5,\
                 isTVinitialization2=0, TViters2=5 ):
        
        if (y is None):
            raise ValueError('input y needs to be specified.')
        else:
            self.y = y
            
        self.psi1_ok = 0
        self.Psi1 = Psi1
        self.psi2_ok = 0
        self.Psi2 = Psi2
        
        self.isinvLS = 0
        self.definedP1 = 0
        self.definedP1T = 0
        self.definedP2 = 0
        self.definedP2T = 0
        self.definedA = 0
        self.definedAT = 0
        self.definedW = 0
        self.definedWT = 0

        if not( invLS is None ):
            self.isinvLS = 1
            self.invLS = invLS

        if not( P1 is None ):
            self.definedP1 = 1
            self.P1 = P1
        if not( P1T is None ):
            self.definedP1T = 1
            self.P1T = P1T
        if not( P2 is None ):
            self.definedP2 = 1
            self.P2 = P2
        if not( P2T is None ):
            self.definedP2T = 1
            self.P2T = P2T
        if not( W is None ):
            self.definedW = 1
            self.W = W
        if not( WT is None ):
            self.definedWT = 1
            self.WT = WT

        if not( A is None ):
            self.definedA = 1
            self.A = A
            
        if not( AT is None ):
            self.definedAT = 1
            self.AT = AT
            
        if not( Psi1 is None ):
            self.Psi1 = Psi1
        if not( Phi1 is None ):
            self.Phi1 = Phi1
        if not( Psi2 is None ):
            self.Psi2 = Psi2
        if not( Phi2 is None ):
            self.Phi2 = Phi2
        
        self.isTVinitialization1 = isTVinitialization1
        self.isTVinitialization2 = isTVinitialization2
        if (tau1 is None):
            self.tau1 = tau1
        else:
            self.tau1 = 1
        if (tau2 is None):
            self.tau2 = tau2
        else:
            self.tau2 = 1
        if ( mu1 is None ):
            self.mu1 = mu1
        else:
            self.mu1 = 0.1
        if ( mu2 is None ):
            self.mu2 = mu2
        else:
            self.mu2 = 0.1
            
        self.x_init = 2
        self.tolA = 1e-3
        self.verbose = 0
        self.MaxIt = 500
        self.TViters1 = TViters1
        self.TViters2 = TViters2
        self.compute_mse = 0
        self.numA = 0
        self.numAt = 0
        self.last_estimate = self.y
        self.stopCriterion = 1

        # a verificar se os parâmetros P e PT foram ambos definidos, e que as dimensionalidades são compatíveis.
        if np.logical_xor( self.definedP1,self.definedP1T ):
            raise ValueError('If you give P1 you must also give PT1, and vice versa.')
            
        if (self.definedP1 == 0):
            self.P1 = lambda x: x
            self.P1T = lambda x: x
            
        if np.logical_xor( self.definedP2,self.definedP2T ):
            raise ValueError('If you give P2 you must also give PT2, and vice versa.')
            
        if (self.definedP2 == 0):
            self.P2 = lambda x: x
            self.P2T = lambda x: x
            
        # se a A for uma função, AT também devia ser definida.
        if ( callable(self.A) and ( not( callable(self.AT) ) ) ):
            raise ValueError('The function handle for transpose of A is missing')
            
        # se a A for uma matrix, definimos funções para multiplicações por A ou AT, para compatibilidade com o caso de serem funções.
        if not( callable(self.A) ):

            if not( (self.y.shape[0] == self.A.shape[0] ) and ( self.y.shape[1] == 1) ):
                raise ValueError('For a MxN observation matrix A, the measurement vector y must be a length M vector.')

            matA = A
            matAT = np.transpose(A)
            self.AT = lambda x: matAT*x
            self.A = lambda x: matA*x
    
        # a partir daqui A e AT são funções.

        self.ATy = self.AT(self.y)
        self.numAt = self.numAt + 1

        # a verificar se o operador invLS foi definido.
        if callable(self.A):

            if ( not(self.isinvLS) or not( callable(self.invLS) ) ):
                raise ValueError('(A^T A + \mu I)^(-1) must be specified as a function handle.')
            else:
                dummy = self.invLS(self.ATy, self.mu1, self.mu2)
                if not( (dummy.shape[0]==self.ATy.shape[0]) and ( dummy.shape[1]==self.ATy.shape[1] ) ):
                    raise ValueError('Specified function handle for solving the LS step does not seem compatible with the specified A and AT.')
        else: # A foi definida como uma matrix
            M,N = matA.shape
            if ( M > N ):
                ATA = np.transpose(matA)*matA 
                self.invLS = lambda x, mu1, mu2: ( np.linalg.inv( ATA+ (mu1+mu2)*np.identity(N) ) )*x
            else:
                AAT = matA*np.transpose(matA)
                self.invLS = lambda x, mu1, mu2: ( AT*np.linalg.inv( (mu1+mu2)*np.identity(M)+AAT)*A )*x
            

        # Se a Psi foi definida, verificar se é uma função com dimensionalidade compatível.
        if callable(self.Psi1):
            if self.isTVinitialization1:
                print('Warning: user specified Phi1 and Psi1 will not be used as TV with initialization flag has been set to 1.')
            else:
                try:
                    dummy = Psi1(P1T(self.ATy),self.tau1);
                    self.psi1_ok = 1
                except:
                    raise ValueError('Something is wrong with function handle for psi')
        else: # se não foi definida, usamos o soft threshold
            self.Psi1 = lambda x,tau: soft(x,tau)

        # Se a Psi foi definida, precisamos também da Phi
        if (self.psi1_ok == 1):
            try:
                self.Phi1
            except:
                raise ValueError('If you give Psi you must also give Phi')
            else:
                # a Phi é definida
                if callable(self.Phi1):
                    try:  # a verificar a dimensionalidade da Phi
                        dummy = self.Phi1( self.P1T(self.ATy) )
                    except:
                        raise ValueError('Something is wrong with function handle for phi')
                else:
                    raise ValueError('Phi does not seem to be a valid function handle')
        else:  # caso não foram definidas nem uma nem outra, utilizamos a norma l1.
            if not self.isTVinitialization1:
                self.Phi1 = lambda x: np.sum( np.abs(x) )
            else:
                self.Phi1 = lambda x: TVnorm(x)
        
        if callable(self.Psi2):
            if self.isTVinitialization2:
                print('Warning: user specified Phi2 and Psi2 will not be used as TV with initialization flag has been set to 1.')
            else:
                try:
                    dummy = Psi2(P2T(self.ATy),self.tau2);
                    self.psi2_ok = 1
                except:
                    raise ValueError('Something is wrong with function handle for psi')
        else: # se não foi definida, usamos o soft threshold
            self.Psi2 = lambda x,tau: soft(x,tau)

        # Se a Psi foi definida, precisamos também da Phi
        if (self.psi2_ok == 1):
            try:
                self.Phi2
            except:
                raise ValueError('If you give Psi you must also give Phi')
            else:
                # a Phi é definida
                if callable(self.Phi2):
                    try:  # a verificar a dimensionalidade da Phi
                        dummy = self.Phi2( self.P2T(self.ATy) )
                    except:
                        raise ValueError('Something is wrong with function handle for phi')
                else:
                    raise ValueError('Phi does not seem to be a valid function handle')
        else:  # caso não foram definidas nem uma nem outra, utilizamos a norma l1.
            if not self.isTVinitialization2:
                self.Phi2 = lambda x: np.sum( np.abs(x) )
            else:
                self.Phi2 = lambda x: TVnorm(x)
        
        #fim da função init

    # função para aplicar o algoritmo SALSA dado o vector observado e os restantes parâmetros já definidos.
    def predict(self, y=None, tau1=1, tau2=1, mu1 = 0.1, mu2 = 0.1,\
                stopCriterion=1, x_true=None, x_init=2, MaxIt=100, tolA=0.01, verbose=0):
        
        if not(stopCriterion is None):
            if not(stopCriterion in [1,2,3] ):
                raise ValueError('Unknown stopping criterion')
            self.stopCriterion = stopCriterion    
        
        if not( x_true is None ):
            self.compute_mse = 1
        else:
            self.compute_mse = 0
    
        if not(y is None):
            if y.shape == self.y.shape:
                self.y = y
                self.ATy = self.AT(y)
            else:
                raise ValueError('y dimensions are incompatible.')
                
        if not(tau1 is None):
            self.tau1 = tau1
        if not(tau2 is None):
            self.tau2 = tau2
        if not( mu1 is None ):
            self.mu1 = mu1
        if not( mu2 is None ):
            self.mu2 = mu2
        if not(MaxIt is None):
            self.MaxIt = MaxIt
        if not(tolA is None):
            self.tolA = tolA
        if not(verbose is None):
            self.verbose = verbose
                
        if (x_init is None):
            x_init = self.x_init
            
        if np.isscalar(x_init):
            if x_init == 0: # zeros
                x = AT( np.zeros(self.y.shape) )
            elif x_init == 1: # aleatoriamente
                x = np.random.rand( self.ATy.shape )
            elif x_init == 2: # ATy
                x = self.ATy
            else: 
                raise ValueError('Unknown initialization option.')
        else:
            if not( (A(x_init)).shape == y.shape ):
                raise ValueError('Size of initial x is not compatible with A')
            else:
                x = x_init

        # se a x verdadeira foi dada, verificamos a dimensionalidade.
        if ( self.compute_mse and not( x_true.shape == x.shape )  ):
            raise ValueError('True x has incompatible size')
            
        P1Tx = self.P1T(x)
        P2Tx = self.P2T(x)
        
        u = P1Tx
        bu = u
        mu1inv = 1/self.mu1
        threshold1 = self.tau1/self.mu1
        v = P2Tx
        bv = v
        mu2inv = 1/self.mu2
        threshold2 = self.tau2/self.mu2
        
        criterion = [1]
        distance1 = []
        distance2 = []
        mses = []

        # inicializar a função do custo.
        resid =  self.y-self.A(x)
        self.numA = self.numA + 1
        prev_f = 0.5*np.sum( resid**2 ) + self.tau1*self.Phi1(u) + self.tau2*self.Phi2(v)

        data_vector_size = np.product(x.shape)

        if self.verbose:
            print('Initial value of objective function = {:.2e}'.format(prev_f) )

        # o relógio começa a contar a partir daqui
        t0 = time.time()
        times = [ 0 ]
        objective = [ prev_f ]

        if self.compute_mse:
            mses.append( np.sum( (x-x_true)**2 )/data_vector_size )

        if self.isTVinitialization1:
            pux = np.zeros(u.shape)
            puy = np.zeros(u.shape)

        if self.isTVinitialization2:
            pvx = np.zeros(v.shape)
            pvy = np.zeros(v.shape)
            
        for it in range(0,self.MaxIt):

            xprev = x

            if self.isTVinitialization1:
                u,pux,puy = chambolle_prox_TV_stop(np.real(P1Tx-bu), threshold1, self.TViters1,\
                                                   tol=1e-2, px=pux, py=puy)
            else:
                u = self.Psi1(P1Tx-bu, threshold1)

            if self.isTVinitialization2:
                v,pvx,pvy = chambolle_prox_TV_stop(np.real(P2Tx-bv), threshold2, self.TViters2,\
                                                   tol=1e-2, px=pvx, py=pvy)
            else:
                v = self.Psi2(P2Tx-bv, threshold2)
                
            r = self.ATy + self.mu1*self.P1(u+bu) + self.mu2*self.P2(v+bv)
            x = self.invLS(r, self.mu1, self.mu2)
            
            P1Tx = self.P1T(x)
            bu = bu + (u - P1Tx)
            P2Tx = self.P2T(x)
            bv = bv + (v - P2Tx)
            
            resid =  y-self.A(x)
            self.numA = self.numA + 1
            objective.append( 0.5*np.sum(resid**2) + self.tau1*self.Phi1(u) + self.tau2*self.Phi2(v) )

            if self.compute_mse:
                err = x - x_true
                mses.append( np.sum(err**2)/data_vector_size )

            distance1.append( np.sqrt( np.sum( (P1Tx-u)**2 ) )/np.sqrt( np.sum( P1Tx**2 ) + np.sum( u**2 ) ) )
            distance2.append( np.sqrt( np.sum( (P2Tx-v)**2 ) )/np.sqrt( np.sum( P2Tx**2 ) + np.sum( v**2 ) ) )

            if (it>0):
                if stopCriterion == 1:
                    # diferença relativa da função de custo.
                    criterion.append( np.abs(objective[-1]-objective[-2])/objective[-2] )
                elif stopCriterion == 2:
                    # diferença relativa entre 2 consecutivos valores da estimativa
                    criterion.append( myNorm(x-xprev)/myNorm(x) )
                elif stopCriterion == 3:
                    # valor alvo da função de custo
                    criterion.append( objective[-1] )
                else:
                    if it == 1:
                        print('Unknown stopping criterion.')
                    continue

                if ( criterion[-1] < self.tolA ):
                    if self.verbose:
                        out_str = 'iter= {:d}, obj= {:.2e}, stop criterion= {:.2e}, (target= {:.2e})'.format(it,\
                                                                                                 objective[-1],\
                                                                                                 criterion[-1], tolA)
                        if self.compute_mse:
                            out_str = out_str + ', MSE = {:.2f}'.format(mses[-1])

                        print(out_str)
                    print('Convergence reached.')

                    times.append( time.time() - t0 )
                    break
            # fim do bloco das condições de paragem

            if self.verbose:
                out_str = 'iter= {:d}, obj= {:.2e}, stop criterion= {:.2e}, (target= {:.2e})'.format(it,objective[-1],\
                                                                                               criterion[-1], tolA)
                if self.compute_mse:
                    out_str = out_str + ', MSE = {:.2f}'.format(mses[-1])

                print(out_str)

            times.append( time.time() - t0 )
        # fim do processo iterativo

        if self.compute_mse:
            return x, objective, times, distance1, distance2, self.numA, self.numAt, mses
        else:
            return x, objective, times, distance1, distance2, self.numA, self.numAt
        
        # fim da função predict

###############################################################################################################