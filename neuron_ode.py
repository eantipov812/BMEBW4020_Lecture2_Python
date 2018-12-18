import numpy as np 

def hodgkin_huxley(t,I_ext,channels=[True,True,True],output='V',initV=-10.0,initx=[0.,0.,1.0]):
    """Hodgkin Huxley simulation 
    
    Notes
    ------
    The Hodgkin Huxley model has 3 ion channels and 3 internal states
    channels = [K, Na, Leak(R)]
    states = [n, m, h]
    
    The indexing of the file reflects that
    
    
    Example
    --------
    ``` python
    >>> import neuron_ode as ode
    >>> import numpy as np
    >>> t = np.arange(0.0,0.02,1e-6)
    >>> V,a,b,x,gnhm,I = ode.hodgkin_huxley(t,np.ones_like(t)*100,output='Full')
    ```
    
    Parameters
    ------------
    t: ndarray of float
        time vector in seconds
    I_ext: ndarray of float
        input current with same size as t
    channels: ndarray of bool
        determins if [K,Na,Leak] channels are active
    output: string
        output mode 
    initV: float
        initial value for voltage
    initx: ndarray-like of float of shape (3,)
        initial values for state variables
    """
    dt = t[1]-t[0]
    ddt = 1e3*dt
    
    # Reverse potentials for K, Na, R (mV)
    E = np.array([   -12., 115., 10.613])
    
    # Channel conductances (mmho/cm^2) [mho -> ohm^{-1}]
    gmax = np.array([ 36.0, 120.0, 0.3])
    
    # Initial states [n, m, h]
    x = np.zeros((len(t),3))
    x[0,:] = np.array(initx)

    # Initialize membrane voltage:
    V  = np.zeros_like(t)
    V[0] = initV
    I = np.zeros((len(t),3))

    # Alpha and beta variables
    a = np.zeros((len(t),3))
    b = np.zeros((len(t),3))

    # Channel conductances [K,Na,R]
    gnmh = np.zeros((len(t),3))
    gnmh[:,2] = gmax[2] #constant no need to be in a loop

    # Perform numerical integration of the ODEs:
    for i in range(1,len(t)):
        old_V = V[i-1]
        a[i,0] = (10.-old_V)/(100.*(np.exp((10.-old_V)/10.)-1.))
        a[i,1] = (25.-old_V)/(10.*(np.exp((25.-old_V)/10.)-1.))
        a[i,2] = 0.07*np.exp(-old_V/20.)

        b[i,0] = 0.125*np.exp(-old_V/80.)
        b[i,1] = 4.*np.exp(-old_V/18.)
        b[i,2] = 1./(np.exp((30.-old_V)/10.)+1.)
    
        x[i,:] = x[i-1,:] + ddt*(a[i,:]*(1.-x[i-1,:]) - b[i,:]*x[i-1,:])
        #x[i,:] = np.clip(x[i,:],0.0,1.0)
        
        gnmh[i,0] = gmax[0]*x[i,0]**4
        gnmh[i,1] = gmax[1]*(x[i,1]**3)*x[i,2]

        # Update the ionic currents and membrane voltage:
        I[i,:] = gnmh[i,:]*(old_V-E) * channels   # nA, channels are used to mask currents
        V[i] = old_V + ddt*(I_ext[i]-np.sum(I[i,:]))

    if output=='Full':
        return V,a,b,x,gnmh,I
    elif output=='V':
        return V
    else:
        raise NotImplementedError

def hodgkin_huxley_rinzel(t,I_ext,channels=[True,True,True],output='V',initV=-10.0,initx=[0.]):
    """Rinzel Reduced Hodgkin Huxley 
    
    Notes
    ------
    The Rinzel Reduced Hodgkin Huxley model has 3 ion channels and 1' internal states
    channels = [K, Na, Leak]
    states = [R] 
    note that the state R does not refere to Leak
    
    Example
    --------
    ``` python
    >>> import neuron_ode as ode
    >>> import numpy as np
    >>> t = np.arange(0.0,0.02,1e-6)
    >>> V,a,b,x,gnhm,I = ode.hodgkin_huxley_rinzel(t,np.ones_like(t)*100,output='Full')
    ```
    
    Parameters
    ------------
    t: ndarray of float
        time vector in seconds
    I_ext: ndarray of float
        input current with same size as t
    channels: ndarray of bool
        determins if [K,Na,Leak] channels are active
    output: string
        output mode 
    initV: float
        initial value for voltage
    initx: ndarray-like of float of shape (1,)
        initial values for state variables
    """
    dt = t[1]-t[0]
    ddt = 1e3*dt
    
    # Reverse potentials for K, Na, Leak (mV)
    E = np.array([   -12., 115., 10.613])
    # Channel conductances (mmho/cm^2) [mho -> ohm^{-1}]
    gmax = np.array([ 36.0,120.0,0.300])

    # Initial states [R]
    x = np.zeros((len(t),1))
    x[0,:] = initx[0]

    # Initialize membrane voltage:
    V  = np.zeros_like(t)
    V[0] = initV
    
    # Initialize Ion channel currents [K,Na,Leak]
    I = np.zeros((len(t),3))

    # Alpha and beta variables
    a = np.zeros((len(t),3))
    b = np.zeros((len(t),3))

    # Channel conductances [K,Na,Leak]
    gnmh = np.zeros((len(t),3))
    gnmh[:,2] = gmax[2] #constant no need to be in a loop

    # Determine the slope S of the Rinzel approximation (see [1]):
    h_0 = 0.07/(0.07+1/(np.exp(3.)+1.))
    n_0 = 0.1/(np.exp(1.)-1.)/(0.1/(np.exp(1.)-1.) + 0.125)
    S = (1-h_0)/n_0

    # Perform numerical integration of the ODEs:
    for i in range(1,len(t)):
        old_V = V[i-1]
        
        a[i,0] = 0.01*(10.-old_V)/(np.exp((10.-old_V)/10.)-1.)
        a[i,1] = 0.1*(25.-old_V)/(np.exp((25.-old_V)/10.)-1.)
        a[i,2] = 0.07*np.exp(-old_V/20.)

        b[i,0] = 0.125*np.exp(-old_V/80.)
        b[i,1] = 4.*np.exp(-old_V/18.)
        b[i,2] = 1./(np.exp((30.-old_V)/10.)+1.)
        
        n_infty = a[i,0]/(a[i,0] + b[i,0])
        m_infty = a[i,1]/(a[i,1] + b[i,1])
        h_infty = a[i,2]/(a[i,2] + b[i,2])
        
        R_infty = S/(1.+S**2)*(n_infty + S*(1-h_infty))
        tau_R = 1. + 5.0*np.exp(-(old_V-10.)**2/(55.**2))
        
        # Use exponential Euler numerical integration method to compute W:
        # Recall that for exponential Euler, if dy/dt=a-b*y, 
        # y(t_k+1)=y_tk*D + a/b*(1-D), where D=exp(-b*dt)    
        _a = 3.0*R_infty/tau_R;
        _b = 3.0/tau_R;
        D = np.exp(-_b*ddt);    
        x[i,:] = x[i-1,:]*D+(_a/_b)*(1-D)
        
        
        gnmh[i,0] = gmax[0] * ((x[i-1]/S)**4)
        gnmh[i,1] = gmax[1] * (m_infty**3)*(1-x[i-1,:])

        # update the ionic currents and membrane voltage:
        I[i,:] = gnmh[i,:]*(old_V-E)
        V[i] = old_V + ddt*(I_ext[i]-np.sum(I[i,:]))

    if output=='Full':
        return V,a,b,x,gnmh,I
    elif output=='V':
        return V
    else:
        raise NotImplementedError
        
def hodgkin_huxley_wilson(t,I_ext,channels=[True,True,True],output='V',initV=-0.7,initx=[0.088]):
    """Wilson Reduced Hodgkin Huxley 
    
    Notes
    ------
    The Rinzel Reduced Hodgkin Huxley model has 2 ion channels and 1 internal states
    channels = [Na, Leak]
    states = [R] 
    note that the state R does not refere to Leak
    
    Example
    --------
    ``` python
    >>> import neuron_ode as ode
    >>> import numpy as np
    >>> t = np.arange(0.0,0.02,1e-6)
    >>> V,a,b,x,gnhm,I = ode.hodgkin_huxley_wilson(t,np.ones_like(t)*100,output='Full')
    ```

    Parameters
    ------------
    t: ndarray of float
        time vector in seconds
    I_ext: ndarray of float
        input current with same size as t
    channels: ndarray of bool
        determins if [K,Na,Leak] channels are active
    output: string
        output mode 
    initV: float
        initial value for voltage
    initx: ndarray-like of float of shape (1,)
        initial values for state variables
    """
    dt = t[1]-t[0]
    ddt = 1e3*dt
    
    # Reverse potentials for Na, K (mV)
    E = np.array([ 55.0, -92.0])
    # Channel conductances (mmho/cm^2) [mho -> ohm^{-1}]
    gmax = np.array([ 1.0, 26.0])

    # Initial states [R]
    x = np.zeros((len(t),1))
    x[0,:] = initx[0]
    
    # time constant for state R
    tau_R = 1.9
    
    # Capacitance 
    C = 1.2
    
    # Initialize membrane voltage:
    V  = np.zeros_like(t)
    V[0] = initV
    
    # Initialize Ion channel currents [Na,Leak]
    I = np.zeros((len(t),2))

    # Channel conductances [Na,Leak]
    gnmh = np.zeros((len(t),2))

    # Perform numerical integration of the ODEs:
    for i in range(1,len(t)):
        old_V = V[i-1]
        R_infty = 0.0135*old_V+1.03
        
        # Use exponential Euler numerical integration method to compute W:
        # Recall that for exponential Euler, if dy/dt=a-b*y, 
        # y(t_k+1)=y_tk*D + a/b*(1-D), where D=exp(-b*dt)    
          
        x[i,:] = x[i-1,:] + ddt*(R_infty-x[i-1,:])/tau_R
        
        gnmh[i,0] = gmax[0]*(17.81+47.71e-2*old_V+32.63e-4*(old_V**2))
        gnmh[i,1] = gmax[1]*x[i-1,:]

        # update the ionic currents and membrane voltage:
        I[i,:] = gnmh[i,:]*(old_V-E)
        V[i] = old_V + ddt/C*(I_ext[i]-np.sum(I[i,:]))

    if output=='Full':
        return V,None,None,x,gnmh,I
    elif output=='V':
        return V
    else:
        raise NotImplementedError

