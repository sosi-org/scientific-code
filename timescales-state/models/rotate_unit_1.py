import sympy
sympy.init_printing()

'''
unit circle
'''

REAL_ONLY = False # whether try to remove complex roots at the symbolic calculations


scenarios = [
    {"tau":20, "I":0},
    {"tau":20, "I":0.23},
    {"tau":20, "I":0.5}
]


initial_values = [ {'v':  0.0, 'w':  0.0, } ] * len(scenarios)

def model1():
    # Define variable as symbols for sympy
    v, w = sympy.symbols("v, w", real=REAL_ONLY)
    t = sympy.symbols("t", real=REAL_ONLY)
    tau, I = sympy.symbols("tau, I")

    # Symbolic expression of the system
    # doulcier's vriation'
    dvdt = -w + I
    dwdt = v/tau


    # dynamics vars, dynamics
    (dyn_vars, t, dyn_derivs, model_params, model_inputs) = (v,w), t, (dvdt,dwdt), (tau,), (I,)
    return dyn_vars, t, dyn_derivs, model_params, model_inputs


def get_ranges(sc):

    xrange = (-2, 2)
    yrange = (-2, 2)
    return (xrange, yrange)

def get_wide_ranges(sc):
    xrange = (-2.0, 2.0)
    yrange = (-2.0, 2.0)
    return (xrange, yrange)

# label for plot figure
pname = 'UnitCrcle'
