import sympy
sympy.init_printing()

'''
    sohale 's variation of FitzHugh-Nagumo model
    based on the formula on https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model
'''


# doulcier's glossary:
# displacement:  perturbation
# autonomous system:  When `t` (float): Time is Not used => autonomous system
# Membrane potential:  v
# Recovery variable:  w


REAL_ONLY = False # whether try to remove complex roots at the symbolic calculations


scenarios = [
    {"a":0.7, "b":0.8, "tau":12.5, "I":0.23},
    {"a":0.7, "b":0.8, "tau":12.5, "I":0.5},
    {"a":0.7, "b":0.8, "tau":12.5, "I":0.7},
]

initial_values = [ {'v':  0.0, 'w':  0.0, } ] * len(scenarios)

def model1():
    # Define variable as symbols for sympy
    v, w = sympy.symbols("v, w", real=REAL_ONLY)
    t = sympy.symbols("t", real=REAL_ONLY)
    a, b, tau, I = sympy.symbols("a, b, tau, I")

    # Symbolic expression of the system
    # sohail
    dvdt = v - v**3 /3.0 - w + I
    dwdt = (v + a - b * w)/tau

    # dynamics vars, dynamics
    (dyn_vars, t, dyn_derivs, model_params, model_inputs) = (v,w), t, (dvdt,dwdt), (a, b, tau), (I,)
    return dyn_vars, t, dyn_derivs, model_params, model_inputs


def get_ranges(sc):
    # todo: xrange with regards to square_nc
    #xrange = (-1.0*5, 1.0*5)
    #xrange = (-1, 1)
    # todo: cte
    #yrange = [-0.5, 0.9285714285714286]
    #yrange = [-0.5, 1.0]
    #yrange = [-1.5, 1.5]
    xrange = (-2.0, 2.0)
    yrange = (-2.0, 2.0)

    return (xrange, yrange)

def get_wide_ranges(sc):
    xrange = (-50.0, 50.0)
    yrange = (-50.0, 50.0)
    return (xrange, yrange)

# label for plot figure
pname = 'FNM:@sohale'
