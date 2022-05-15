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

doulcier = True
REAL_ONLY = False # whether try to remove complex roots at the symbolic calculations

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

# Magical Indices (for readability)
# Consts:  For magical numbers
# Not every 0 is the same. Not every 1 is the same.
# Now you can search all 0s in that capacity.
# Naming mth eindices. Use unicode to make these shorter.
# It is just about "naming" aand labelling, not about configurating.
# It is not for eliminatig magical numbers. It has a higher purpose.
#
# # "model" data structure
#  i.e. (dyn_vars, t, dyn_derivs, model_params, model_inputs)
M_VARS_I = 0  # dynamics vars
M_DYN_I = 2  # dynamics
M_TIME_I = 1
M_PARAMS_I = 3
M_EXOGIPUTS_I = 4 # exogenous inputs

# The io format, for (return value of) `fitzhugh_nagumo_partial()`
I_LAMB_NPARRAY = 0 #M_SYMBOLIC_.I
I_LAMB_ASLIST = 1 # M_LAMBDA_.I

# for data structure for "integration"/ode results
SIM_I_NDIM = 0
SIM_I_BINS = 1   # dimention for "bin"s if simulation and integration
                # means "across bins"
                # was S IM_I_BIN

# Try to avoid the following
# Avoid such hardcoded indexing though
# every usage will be a technical debt when we generalise to higher dimensional equations
# will need to be parametrised in proper variables (not consts)
_VAR_I_V = 0
_VAR_I_W = 1
_NDIM_VW = 2  # "across dims". try to avoid

# (_NDIM_VW, 1500) is more readable than (2, 1500)


_model = model1()

UPSAMPLEx=5
# UPSAMPLEx=1

SIMU_TIME=200
SIMU_STEPS1=1500*UPSAMPLEx
SIMU_STEPS2=1000*UPSAMPLEx

time_span = np.linspace(0, SIMU_TIME, num=SIMU_STEPS1)


def fitzhugh_nagumo_partial(_model, **model_params_and_inputs):

    (dyn_vars, t, dyn_derivs, model_params, model_inputs) = _model
    #(v,w), t, (dvdt,dwdt), (a, b, tau), (I,) = (dyn_vars, t, dyn_derivs, model_params, model_inputs)
    print((dyn_vars, t, dyn_derivs, model_params, model_inputs))

    dyn_vars2 = tuple([deriv_elem.subs(model_params_and_inputs) for deriv_elem in dyn_derivs])
    model_lamb = sympy.lambdify((*dyn_vars,t), dyn_vars2, dummify=False)
    ndim = len(dyn_derivs)

    # I_LAMB_NPARRAY: receives ndarray, returns ndarray
    def p(x, t):
        # either args=((_NDIM_VW,n), (n,))  or args=((_NDIM_VW,), ())
        # is_scalar =
        #print(x.shape, 'x.shape')
        assert x.shape[SIM_I_NDIM] == ndim
        x0x1 = tuple(x) # x[0,:],x[1,:]
        y = model_lamb(*x0x1, t)
        assert len(y) == _NDIM_VW
        #print(y[0].shape)
        dvdw = np.array(y)
        #print(dvdw.shape)
        assert len(dvdw.shape) == _NDIM_VW or dvdw.shape==(_NDIM_VW,)
        assert dvdw.shape == dvdw.shape == (ndim,) or (ndim, x.shape[SIM_I_BINS])
        # either (_NDIM_VW,n)  or (_NDIM_VW,)
        return dvdw

    # I_LAMB_ASLIST: receives list, returns list
    def p3(xy_list,t):
          # print(len(xy_list))
          # print(xy_list[0].shape)
          assert len(xy_list) == ndim
          # to make suree (*xy,t) align correctly
          dvdw = model_lamb(*xy_list, t)
          assert len(dvdw) == ndim
          return dvdw

    # I_LAMB_NPARRAY, I_LAMB_ASLIST
    return p,p3

# ic: Array containing the value of y for each desired time in t, with the initial value y0 in the first row.

def get_displacement(
        param, dmax=0.5,
        time_span=np.linspace(0, SIMU_TIME, SIMU_STEPS2),
        number=20
  ):
    # We start from the resting point...
    ic = scipy.integrate.odeint(
          fitzhugh_nagumo_partial(_model, **param)[I_LAMB_NPARRAY],
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
            fitzhugh_nagumo_partial(_model, **param)[I_LAMB_NPARRAY],
            y0=ic+np.array([displacement,0]),
            #full_output=True, adds a second entry to output
            t=time_span
        ))
        solu = traj[-1]  # v, w = sol.T
        vel_eval = fitzhugh_nagumo_partial(_model, **param)[I_LAMB_NPARRAY](solu.T, time_span)

        # vel_eval.shape: # (_NDIM_VW, 1500)
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
            v = ax[i].plot(time_span,trajectories[i][j][:,_VAR_I_V], color='C0')
            w = ax[i].plot(time_span,trajectories[i][j][:,_VAR_I_W], color='C1', alpha=.5)
        ax[i].legend([v[0],w[0]],['v','w'])
plt.tight_layout()


def minmax(simulation):
    mins = np.min(simulation, axis=SIM_I_BINS)
    maxs = np.max(simulation, axis=SIM_I_BINS)
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
        # simulation[_VAR_I_W,:] = simulation[_VAR_I_W,:] * param['tau']
        d_v = simulation[_VAR_I_V,:]
        d_w = simulation[_VAR_I_W,:]
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
        d_v0 = simulation[_VAR_I_V,:]
        d_w0 = simulation[_VAR_I_W,:]
        d_t = time_span
        d2_v = np.diff(d_v0)/np.diff(time_span)
        d2_w = np.diff(d_w0)/np.diff(time_span)
        (mm1,mm2, ranges) = minmax(np.array([d2_v, d2_w]))
        ax[i].plot(np.array([mm1,mm2]), np.array([mm1,mm2]), 'r--', alpha=.2)
        ax[i].plot(d2_v, d2_w, 'k-')
    np.set_printoptions(precision=2)
    ax[i].set(xlabel=r'$\ddot{V}$', ylabel=r'$\ddot2{W}$',
        title='ACCELERATION\nv:{}, w:{}\n {:<8}'.format(ranges[0][:], ranges[1][:], pname));

##############################################


# todo: why is v defined here?
def symbolic_nullclines(_model, param):
    # symbolic (algebraic) null-clines:
    print()
    print(param)
    dv = _model[M_DYN_I][_VAR_I_V].subs(param)
    dw = _model[M_DYN_I][_VAR_I_W].subs(param)
    w = _model[M_VARS_I][_VAR_I_W]
    v = _model[M_VARS_I][_VAR_I_V]

    from sympy import Eq
    from sympy import Matrix
    from sympy import solve
    c = sympy.symbols("c", real=REAL_ONLY)

    # v - v**3  - w + I == 0
    # w == c

    nlc1 = dv  # dv=0
    #nlc1 = dw # dw=0

    # The "scan"ning line/shape (guiding shape)
    #lin1 = v-c # single root solution
    lin1 = w-c  # multiple roots
    # lin1 = v-c+w
    # lin1 = v*v-c+w*w

    print('nlc1', nlc1)
    print('lin1', lin1)

    eq3m = Matrix([nlc1, lin1])
    zero0 =  Matrix([0,0])
    # also: zero= c*0 ,  0 * dv , sympy.core.numbers.Zero
    # unexplored option: `Eq(..., evaluate=False)`
    eq3 = Eq(eq3m, zero0)
    sympy.pprint(eq3, use_unicode=True)
    print('Going to solve')
    vars_list, _solution_set = \
        solve(eq3, (v,w),force=True, set=True)
    # Alternative: sympy.solve([eq1,eq2], (w,), force=True, manual=True, set=True)
    # ([w], {(-v**3 + v,)})
    # convert set to list
    # vars_list: # (v,w)
    # len = number of roots:
    solution_sympy_list = list(_solution_set)
    # each solution is a tuple, for (v,w). has 2=ndim elements

    # todo: c -> s
    # todo: rename c to s
    # c is the free parameter for the "curve". Since it's aa null-cline, we need (exactly) one free ariable for it
    solution_lambda_list = [
        # `modules=["scipy", "numpy"]`` is essential for the correct calculation of nullclines (generalised approach). Especiaally when there are multiple roots.
        sympy.lambdify((c,), solt_vw, dummify=False, modules=["scipy", "numpy"] )
            for solt_vw in solution_sympy_list ]

    # each is a function that returns a tuple of size `ndim``

    return solution_sympy_list, solution_lambda_list



# Nullclines

square_nc = False

def plot_isocline(_model, param, ax, a, b, tau, I, color='k', style='--', opacity=.5, vmin=-5,vmax=5):
    """Plot the null iscolines of the Fitzhugh nagumo system"""
    # todo: rename v_np
    v = np.linspace(vmin,vmax,100)
    s_np = np.linspace(-10,+10, 1000)

    nc_solution_list, nc_lambdas = symbolic_nullclines(_model, param)

    ctr = 0
    for ncl in nc_lambdas:
      c = s_np
      #c = v
      vw_tuple = ncl(c) # t
      #ax.plot(c, vw_tuple, style, color=color, alpha=opacity)
      v_ = vw_tuple[_VAR_I_V]+np.random.randn(1)*0.001
      w_ = vw_tuple[_VAR_I_W]+np.random.randn(1)*0.001

      print('-----',v_.shape,v_.dtype,'  w',w_.shape, w_.dtype) # why float
      wch_ =np.logical_and(np.isreal(v_),np.isreal(v_))
      v_ = v_[wch_]
      w_ = w_[wch_]
      #v_ = np.abs(v_)
      #w_ = np.abs(w_)
      #v_ = np.real(v_)
      #w_ = np.real(w_)

      COLS = ['r','g','b', 'm']; ctr += 1
      ax.plot(v_, w_, '.', color=COLS[ctr], alpha=opacity)
      # todo: indent

    ax.plot(v, v - v**3 + I, style, color=color, alpha=opacity)
    # (v - a - b * w)/tau == 0
    # w=...
    ax.plot(v, (v - a)/b, style, color=color, alpha=opacity)
    if (square_nc):
        #ax.set_aspect('equal', adjustable='box')
        # adjustable='datalim')
        ax.set_aspect(param['tau'], adjustable='box')

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
for i, sc in enumerate(scenarios):
    plot_isocline(_model, sc, ax[i], **sc)
    ax[i].set(xlabel='v', ylabel='w',
              title='{}'.format(sc))

##############################################

def plot_vector_field(ax, param, xrange, yrange, steps=50):
    # Compute the vector field
    x = np.linspace(xrange[0], xrange[1], steps)
    y = np.linspace(yrange[0], yrange[1], steps)
    X,Y = np.meshgrid(x,y)  # each shape: (50,50)

    dxdy = fitzhugh_nagumo_partial(_model, **param)[I_LAMB_ASLIST]([X,Y], 0.0)

    dx,dy = dxdy[:2]

    # streamplot is an alternative to quiver
    # that looks nicer when your vector filed is
    # continuous.
    ax.streamplot(X,Y,dx, dy, color=(0,0,0,.1))

    ax.set(xlim=(xrange[0], xrange[1]), ylim=(yrange[0], yrange[1]))
    if (square_nc):
        #ax.set_aspect('equal', adjustable='box')
        # adjustable='datalim')
        ax.set_aspect(param['tau'], adjustable='box')

fig, ax = plt.subplots(1, 3, figsize=(20, 6))
for i, sc in enumerate(scenarios):
    # todo: xrange with regards to square_nc
    #xrange = (-1*5, 1*5)
    xrange = (-1, 1)
    yrange = [(1/sc['b'])*(x-sc['a']) for x in xrange]
    plot_vector_field(ax[i], sc, xrange, yrange)
    ax[i].set(xlabel='v', ylabel='w',
          title='{}'.format(sc))
##############################################

# symbolic and hardcoded Jacobian

def jacobian_fitznagumo(v, w, a, b, tau, I):
    """ Jacobian matrix of the ODE system modeling Fitzhugh-Nagumo's excitable system
    Args
    ====
        a,b (float): Parameters
        tau (float): Recovery timescale.
    Return: np.array 2x2"""
    return np.array([[- 3 * v**2 + 1 , -1],
                       [1/tau, -b/tau]])
    # for `doulcier` only
    # yes it matches the sym_jacobian() when I set doulcier=True

def symb_jacobian(_model):
    # Symbolic computation of the Jacobian using sympy...
    (dyn_vars, t, dyn_derivs, model_params, model_inputs) = _model

    # Symbolic expression of the matrix
    sys = sympy.Matrix(dyn_derivs)
    var = sympy.Matrix(dyn_vars)
    jac = sys.jacobian(var)

    # You can convert jac to a function:
    jacobian_fitznagumo_symbolic = sympy.lambdify((*dyn_vars, *model_params, *model_inputs), jac, dummify=False)

    sympy.pprint(jac, use_unicode=True)
    # https://docs.sympy.org/latest/tutorial/printing.html
    print()
    print(sympy.pretty(jac))

    return jacobian_fitznagumo_symbolic

jacobian_fitznagumo = symb_jacobian(_model)


# by doulcier
def stability(jacobian):
    """ Stability of the equilibrium given its associated 2x2 jacobian matrix.
    Use the eigenvalues.
    Args:
        jacobian (np.array 2x2): the jacobian matrix at the equilibrium point.
    Return:
        (string) status of equilibrium point.
    """

    eigv = np.linalg.eigvals(jacobian)

    if all(np.real(eigv)==0) and all(np.imag(eigv)!=0):
        nature = "Center"
    elif np.real(eigv)[0]*np.real(eigv)[1]<0:
        nature = "Saddle"
    else:
        stability = 'Unstable' if all(np.real(eigv)>0) else 'Stable'
        nature = stability + (' focus' if all(np.imag(eigv)!=0) else ' node')
    return nature

def stability_alt(jacobian):
    """ Stability of the equilibrium given its associated 2x2 jacobian matrix.
    Use the trace and determinant.
    Args:
        jacobian (np.array 2x2): the jacobian matrix at the equilibrium point.
    Return:
        (string) status of equilibrium point.
    """

    determinant = np.linalg.det(jacobian)
    trace = np.matrix.trace(jacobian)
    if np.isclose(trace, 0):
        nature = "Center (Hopf)"
    elif np.isclose(determinant, 0):
        nature = "Transcritical (Saddle-Node)"
    elif determinant < 0:
        nature = "Saddle"
    else:
        nature = "Stable" if trace < 0 else "Unstable"
        nature += " focus" if (trace**2 - 4 * determinant) < 0 else " node"
    return nature


# todo: write this symbollically:
def find_roots(a,b,I, tau):
    # The coeficients of the polynomial equation are:
    # 1           * v**3
    # 0           * v**2
    # - (1/b - 1) * v**1
    # - (a/b + I) * v**0
    coef = [1, 0, 1/b - 1, - a/b - I]

    # We are only interested in real roots.
    # np.isreal(x) returns True only if x is real.
    # The following line filter the list returned by np.roots
    # and only keep the real values.
    roots = [np.real(r) for r in np.roots(coef) if np.isreal(r)]

    # We store the position of the equilibrium.
    return [[r, r - r**3 + I] for r in roots]

eqnproot = {}
for i, param in enumerate(scenarios):
    eqnproot[i] = find_roots(**param)

#
eqstability = {}
for i, param in enumerate(scenarios):
    eqstability[i] = []
    for e in eqnproot[i]:
        J = jacobian_fitznagumo(e[_VAR_I_V],e[_VAR_I_W], **param)
        eqstability[i].append(stability(J))
print(eqstability)

plt.show()
#brew install graphviz
