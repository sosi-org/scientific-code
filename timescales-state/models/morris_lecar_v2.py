import sympy
sympy.init_printing()

def model2():
    # Morris Lacar
    # Based on https://github.com/jonmarty/Morris-Lecar/blob/master/MorrisLecar.ipynb
    # variation 2
    # https://en.wikipedia.org/wiki/Morris%E2%80%93Lecar_model

    t = sympy.symbols("t")
    V, N, = sympy.symbols( "V, N")

    # '(\w+)':   $1,
    #tanh = sympy.tanh
    #cosh = sympy.cosh
    #exp = sympy.exp
    #half12 = sympy.One()/2
    half12 = sympy.core.numbers.One()/2

    def tanh(x):
        return x - (x**3)/3
    def cosh(x):
        return 1+(x**2)/2
    #def tanh(x):
    #    return x - (x**3)/3 + 2*(x**5)/15
    #def cosh(x):
    #    return 1+(x**2)/2 +(x**4)/24

    def sm1(r):
        return half12 * (1 + tanh(r))

    def exp(x):
        return 1 + x + (x**2)/2 # + (x**3)/6

    def sm2(x):
        return 1 / (1 + exp(-2*x))


    '''
    C, # Capacitance of membrane
    V_1, # Tuning parameters for steady state and time constant
    phi, # reference frequency
    V_L, # Equilibrum potentials for ion channels
    g_Ca, # leak, conductances through membrane for each ion
    V, # Membrane potential
    N, # Recovery variance
    '''
    C, V_1, V_2, V_3, V_4, phi, V_L, V_Ca, V_K, g_Ca, g_K, g_L, I \
        = sympy.symbols("""
        C, V_1, V_2, V_3, V_4, phi, V_L, V_Ca, V_K, g_Ca, g_K, g_L, I
      """)

    # Define functions
    M_ss = sm2((V - V_1) / V_2)
    N_ss = sm2((V - V_3) / V_4)
    T_N = 1 / (phi * cosh((V - V_3) / (2 * V_4)))

    # Define differential equations
    # dV(I)
    dV = (I - g_L * (V - V_L) - g_Ca *
                       M_ss * (V - V_Ca) - g_K * N * (V - V_K)) / C

    dN = (N_ss - N) / T_N

    # Equations for the input of each channel
    L = - g_L * (V - V_L)
    Ca = - g_Ca * M_ss * (V - V_Ca)
    K = - g_K * N * (V - V_K)

    # dynamics vars
    dyn_vars = (V, N)
    t = t
    # dynamics
    dyn_derivs =(dV, dN)
    model_params =  (C, V_1, V_2, V_3, V_4, phi, V_L, V_Ca, V_K, g_Ca, g_K, g_L)
    model_inputs =  (I,)

    model = dyn_vars, t, dyn_derivs, model_params, model_inputs
    return model

scenarios = [ #params =
  {
    # (\w+) =        '$1':
    'C':  6.69810502993, # Capacitance of membrane
    'V_1':  30, # Tuning parameters for steady state and time constant
    'V_2':  15,
    'V_3':  0,
    'V_4':  30,
    'phi':  0.025, # reference frequency
    'V_L':  -50, # Equilibrum potentials for ion channels
    'V_Ca':  100,
    'V_K':  -70,
    'g_Ca':  1.1, # leak, conductances through membrane for each ion
    'g_K':  2,
    'g_L':  0.5,

    'I': 0.0 + i * 0.4,

    # does not work
    #'initial': {
    #    'V':  -52.14, # Membrane potential
    #    'N':  0.02, # Recovery variance
    #}
  } for i in range(3)]

initial_values = [
    {
        'V':  -52.14, # Membrane potential
        'N':  0.02, # Recovery variance
    }
    ] * len(scenarios)

def get_ranges(sc):
    # todo: xrange with regards to square_nc
    #xrange = (-1*5, 1*5)
    #xrange = (-1, 1)
    #yrange = [(1/sc['b'])*(x-sc['a']) for x in xrange]
    print('todo: ranges')
    #return (xrange, yrange)
    # todo: rename v,w
    # todo: symbolic

    mins = [-1,-100]
    maxs = [1,100]
    xrange = (mins[0], maxs[0])
    yrange = (mins[1], maxs[1])
    return (xrange, yrange)

def get_wide_ranges(sc):
    xrange = (-500.0, 500.0)
    yrange = (-500.0, 500.0)
    return (xrange, yrange)

# label for plot figure
pname = 'ML:jonmarty'
