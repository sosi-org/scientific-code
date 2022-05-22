import sympy
sympy.init_printing()

doulcier=True


# doulcier's glossary:
# displacement:  perturbation
# autonomous system:  When `t` (float): Time is Not used => autonomous system
# Membrane potential:  v
# Recovery variable:  w


REAL_ONLY = False # whether try to remove complex roots at the symbolic calculations


if doulcier:
    scenarios = [
        {"a":-.3, "b":1.4, "tau":20, "I":0},
        {"a":-.3, "b":1.4, "tau":20, "I":0.23},
        {"a":-.3, "b":1.4, "tau":20, "I":0.5}
    ]
else:
    scenarios = [
        {"a":0.7, "b":0.8, "tau":12.5, "I":0.23},
        {"a":0.7, "b":0.8, "tau":12.5, "I":0.5},
        {"a":0.7, "b":0.8, "tau":12.5, "I":0.7},
    ]

initial_values = [ {'v':  0.0, 'w':  0.0, } ] * len(scenarios)

def models(doulcier):
    # Define variable as symbols for sympy
    v, w = sympy.symbols("v, w", real=REAL_ONLY)
    t = sympy.symbols("t", real=REAL_ONLY)
    a, b, tau, I = sympy.symbols("a, b, tau, I")

    # Symbolic expression of the system
    if doulcier:
        dvdt = v - v**3  - w + I
        dwdt = (v - a - b * w)/tau
    else:
        # sohail
        dvdt = v - v**3 /3.0 - w + I
        dwdt = (v + a - b * w)/tau

    # dynamics vars, dynamics
    (dyn_vars, t, dyn_derivs, model_params, model_inputs) = (v,w), t, (dvdt,dwdt), (a, b, tau), (I,)
    return dyn_vars, t, dyn_derivs, model_params, model_inputs

model1 = lambda : models(doulcier=True)

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

    return (xrange, yrange)

if doulcier:
  pname = 'doulcier'
else:
  pname = '@sohale'
