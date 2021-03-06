import sympy
sympy.init_printing()

doulcier=True


'''
    doulcier's variation of FitzHugh-Nagumo model

    doulcier's glossary:
        displacement:  perturbation
        autonomous system:  When `t` (float): Time is Not used => autonomous system
        Membrane potential:  v
        Recovery variable:  w
'''

REAL_ONLY = False # whether try to remove complex roots at the symbolic calculations


scenarios = [
    {"a":-.3, "b":1.4, "tau":20, "I":0},
    {"a":-.3, "b":1.4, "tau":20, "I":0.23},
    {"a":-.3, "b":1.4, "tau":20, "I":0.5}
]


initial_values = [ {'v':  0.0, 'w':  0.0, } ] * len(scenarios)

def model1():
    # Define variable as symbols for sympy
    v, w = sympy.symbols("v, w", real=REAL_ONLY)
    t = sympy.symbols("t", real=REAL_ONLY)
    a, b, tau, I = sympy.symbols("a, b, tau, I")

    # Symbolic expression of the system
    # doulcier's vriation'
    dvdt = v - v**3  - w + I
    dwdt = (v - a - b * w)/tau


    # dynamics vars, dynamics
    (dyn_vars, t, dyn_derivs, model_params, model_inputs) = (v,w), t, (dvdt,dwdt), (a, b, tau), (I,)
    return dyn_vars, t, dyn_derivs, model_params, model_inputs

# model1 = lambda : models(doulcier=True)

# phase plane panel range
# PPrange
# phpp = {'xrange': , 'yrange':,}
def get_ranges(sc):
    # todo: xrange with regards to square_nc
    #xrange = (-1*5, 1*5)
    xrange = (-1, 1)
    yrange = [(1/sc['b'])*(x-sc['a']) for x in xrange]

    # todo: rename v,w
    # todo: symbolic
    print(xrange, yrange)

    return (xrange, yrange)

def get_wide_ranges(sc):
    xrange = (-50.0, 50.0)
    yrange = (-50.0, 50.0)
    return (xrange, yrange)

# label for plot figure
pname = 'FNM:doulcier'
