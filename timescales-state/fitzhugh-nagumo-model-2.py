# Based on fitzhugh-nagumo-model-1.py
# Now refactoring to use sympy for arbitrary models

# variuos forms of FNM?

# from wonderful page of Guilhem Doulcier:
# https://www.normalesup.org/~doulcier/teaching/modeling/excitable_systems.html
# Based on Guilhem Doulcier's work


from functools import partial
import numpy as np
import scipy.integrate
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches #used to write custom legends
import sympy
sympy.init_printing()

doulcier = False

# doulcier's glossary:
# displacement:  perturbation
# autonomous system:  When `t` (float): Time is Not used => autonomous system
# Membrane potential:  v
# Recovery variable:  w


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

def model1():
    # Define variable as symbols for sympy
    v, w = sympy.symbols("v, w")
    t = sympy.symbols("t")
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

(dyn_vars, t, dyn_derivs, model_params, model_inputs) = model1()
#(v,w), t, (dvdt,dwdt), (a, b, tau), (I,) = (dyn_vars, t, dyn_derivs, model_params, model_inputs)
print((dyn_vars, t, dyn_derivs, model_params, model_inputs))
# _model = model1()


UPSAMPLEx=5
# UPSAMPLEx=1

SIMU_TIME=200
SIMU_STEPS1=1500*UPSAMPLEx
SIMU_STEPS2=1000*UPSAMPLEx

time_span = np.linspace(0, SIMU_TIME, num=SIMU_STEPS1)

# is it used?
#model_lamb = sympy.lambdify((*dyn_vars, *model_params, *model_inputs), dyn_derivs, dummify=False)



def fitzhugh_nagumo_partial(**model_params_and_inputs):
    # print('@FNM', model_params_and_inputs)

    dyn_vars2 = tuple([deriv_elem.subs(model_params_and_inputs) for deriv_elem in dyn_derivs])
    model_lamb = sympy.lambdify((*dyn_vars,t), dyn_vars2, dummify=False)
    ndim = len(dyn_derivs)

    def p(x, t):
        # either ((2,n), (n,))  or ((2,), ())
        # is_scalar =
        #print(x.shape, 'x.shape')
        assert x.shape[0] == ndim
        x0x1 = tuple(x) # x[0,:],x[1,:]
        y = model_lamb(*x0x1, t)
        assert len(y) == 2
        #print(y[0].shape)
        dvdw = np.array(y)
        #print(dvdw.shape)
        assert len(dvdw.shape) == 2 or dvdw.shape==(2,) 
        assert dvdw.shape == dvdw.shape == (ndim,) or (ndim, x.shape[1])
        # either (2,n)  or (2,)
        return dvdw
    return p

# ic: Array containing the value of y for each desired time in t, with the initial value y0 in the first row.

def get_displacement(
        param, dmax=0.5,
        time_span=np.linspace(0, SIMU_TIME, SIMU_STEPS2),
        number=20
  ):
    # We start from the resting point...
    ic = scipy.integrate.odeint(
          fitzhugh_nagumo_partial(**param),
          y0=[0,0],
          #t= np.linspace(0,SIMU_STEPS2-1, SIMU_STEPS2)
          t= np.linspace(0,SIMU_TIME*UPSAMPLEx, SIMU_STEPS2)
      )[-1]
    # and do some displacement of the potential.
    veocities = []
    traj = []
    for displacement in np.linspace(0, dmax, number):
        traj.append(
          scipy.integrate.odeint(
            fitzhugh_nagumo_partial(**param),
            y0=ic+np.array([displacement,0]),
            #full_output=True, adds a second entry to output
            t=time_span
        ))
        solu = traj[-1]  # v, w = sol.T
        vel_eval = fitzhugh_nagumo_partial(**param)(solu.T, time_span)

        # vel_eval.shape: # (2, 1500)
        veocities.append(vel_eval)
    return traj, veocities

# Do the numerical integration.
trajectories = {} # We store the trajectories in a dictionnary, it is easier to recover them.
velo_s = {}
for i,param in enumerate(scenarios):
    trajectories[i], velo_s[i] \
        = get_displacement(param, number=4, time_span=time_span, dmax=0.5)

if doulcier:
  pname = 'doulcier'
else:
  pname = '@sohale'

# Draw the trajectories.
fig, ax = plt.subplots(1, len(scenarios), figsize=(5*len(scenarios),5))
for i,param in enumerate(scenarios):
        ax[i].set(xlabel='Time', ylabel='v, w',
                     title='{:<8} {}'.format(pname, param))
        for j in range(len(trajectories[i])):
            v = ax[i].plot(time_span,trajectories[i][j][:,0], color='C0')
            w = ax[i].plot(time_span,trajectories[i][j][:,1], color='C1', alpha=.5)
        ax[i].legend([v[0],w[0]],['v','w'])
plt.tight_layout()


def minmax(simulation):
    mins = np.min(simulation, axis=1)
    maxs = np.max(simulation, axis=1)
    ranges = np.array([mins, maxs ]).T
    print('ranges:', ranges)
    mm1 = np.max(mins, axis=0)
    mm2 = np.min(maxs, axis=0)
    return (mm1, mm2, ranges)

fig, ax = plt.subplots(1, len(scenarios), figsize=(5*len(scenarios),5))
# i = simulation/experiment (with different parameters)
for i,param in enumerate(scenarios):
    velocities = velo_s[i]
    # j = repeats = 4
    for j in range(len(velocities)):
        simulation = velocities[j].copy()  # [2,1500]
        # fake rescaling W:
        simulation[1,:] = simulation[1,:] * param['tau']
        d_v = simulation[0,:]
        d_w = simulation[1,:]
        (mm1,mm2, ranges) = minmax(simulation)
        ax[i].plot(np.array([mm1,mm2]), np.array([mm1,mm2]), 'r--', alpha=.2)
        ax[i].plot(d_v, d_w, 'k-')
    np.set_printoptions(precision=2)
    ax[i].set(xlabel=r'$\dot{V}$', ylabel=r'$\dot{W}$',
        title='v:{}, w:{}\n {:<8}'.format(ranges[0][:], ranges[1][:], pname));

# Second derivative
fig, ax = plt.subplots(1, len(scenarios), figsize=(5*len(scenarios),5))
# i = simulation/experiment (with different parameters)
for i,param in enumerate(scenarios):
    velocities = velo_s[i]
    # j = repeats = 4
    for j in range(len(velocities)):
        simulation = velocities[j]  # [2,1500]
        d_v0 = simulation[0,:]
        d_w0 = simulation[1,:]
        d_t = time_span
        d2_v = np.diff(d_v0)/np.diff(time_span)
        d2_w = np.diff(d_w0)/np.diff(time_span)
        (mm1,mm2, ranges) = minmax(np.array([d2_v, d2_w]))
        ax[i].plot(np.array([mm1,mm2]), np.array([mm1,mm2]), 'r--', alpha=.2)
        ax[i].plot(d2_v, d2_w, 'k-')
    np.set_printoptions(precision=2)
    ax[i].set(xlabel=r'$\ddot{V}$', ylabel=r'$\ddot2{W}$',
        title='ACCELERATION\nv:{}, w:{}\n {:<8}'.format(ranges[0][:], ranges[1][:], pname));

# symbolic

def jacobian_fitznagumo(v, w, a, b, tau, I):
    """ Jacobian matrix of the ODE system modeling Fitzhugh-Nagumo's excitable system
    Args
    ====
        a,b (float): Parameters
        tau (float): Recovery timescale.
    Return: np.array 2x2"""
    return np.array([[- 3 * v**2 + 1 , -1],
                       [1/tau, -b/tau]])

# Symbolic computation of the Jacobian using sympy...

# Symbolic expression of the matrix
sys = sympy.Matrix(dyn_derivs)
var = sympy.Matrix(dyn_vars)
jac = sys.jacobian(var)

# You can convert jac to a function:
jacobian_fitznagumo_symbolic = sympy.lambdify((*dyn_vars, *model_params, *model_inputs), jac, dummify=False)

#jacobian_fitznagumo = jacobian_fitznagumo_symbolic
sympy.pprint(jac, use_unicode=True)
# https://docs.sympy.org/latest/tutorial/printing.html
print()
print(sympy.pretty(jac))

plt.show()
#brew install graphviz
