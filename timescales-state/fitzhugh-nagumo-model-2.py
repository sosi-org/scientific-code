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

_model = model1()

UPSAMPLEx=5
# UPSAMPLEx=1

SIMU_TIME=200
SIMU_STEPS1=1500*UPSAMPLEx
SIMU_STEPS2=1000*UPSAMPLEx

time_span = np.linspace(0, SIMU_TIME, num=SIMU_STEPS1)


def fitzhugh_nagumo_partial(_model, **model_params_and_inputs):
    # print('@FNM', model_params_and_inputs)

    (dyn_vars, t, dyn_derivs, model_params, model_inputs) = _model
    #(v,w), t, (dvdt,dwdt), (a, b, tau), (I,) = (dyn_vars, t, dyn_derivs, model_params, model_inputs)
    print((dyn_vars, t, dyn_derivs, model_params, model_inputs))

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
    def p3(xy_list,t):
          # print(len(xy_list))
          # print(xy_list[0].shape)
          assert len(xy_list) == ndim
          # to make suree (*xy,t) align correctly
          dvdw = model_lamb(*xy_list, t)
          assert len(dvdw) == ndim
          return dvdw

    return p,p3

# ic: Array containing the value of y for each desired time in t, with the initial value y0 in the first row.

def get_displacement(
        param, dmax=0.5,
        time_span=np.linspace(0, SIMU_TIME, SIMU_STEPS2),
        number=20
  ):
    # We start from the resting point...
    ic = scipy.integrate.odeint(
          fitzhugh_nagumo_partial(_model, **param)[0],
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
            fitzhugh_nagumo_partial(_model, **param)[0],
            y0=ic+np.array([displacement,0]),
            #full_output=True, adds a second entry to output
            t=time_span
        ))
        solu = traj[-1]  # v, w = sol.T
        vel_eval = fitzhugh_nagumo_partial(_model, **param)[0](solu.T, time_span)

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

##############################################


# todo: why is v defined here?
def symbolic_nullclines(_model, param):
    # symbolic (algebraic) null-clines:
    print()
    # see https://github.com/sosi-org/scientific-code/blob/633224dc72692c13acc8263e21c933119eeaaf69/beeseyes/pycode/derive_formulas.py
    dv = _model[2][0].subs(param)
    w = _model[0][1]
    v = _model[0][0]
    print('dv_', dv)
    print('w', w)
    from sympy import Eq #as Equation
    # https://docs.sympy.org/latest/modules/core.html?highlight=eq#sympy.core.relational.Equality
    # https://docs.sympy.org/latest/modules/solvers/solvers.html
    if False:
        zero = 0 * dv # ? sympy.core.numbers.Zero
        eq1 = Eq(dv, zero)
        sympy.pprint(eq1, use_unicode=True)
        c = sympy.symbols("c")
        eq2 = Eq(v, c*0, evaluate=False)
        sympy.pprint(eq2, use_unicode=True)
        # _.doit()
        # simplify(_)
        #solutions = sympy.solve(eq, (a,b,t),force=True, manual=True, set=True)
        solutions = sympy.solve([eq1,eq2], (w,),force=True, manual=True, set=True)
        print('solutions', solutions)  # ([w], {(-v**3 + v,)})
        print('solutions[0]', solutions[0])
        #sympy.core.numbers.Zero
        #sol_a = solution[0][0]
        #sol_b = solution[0][1]
        #sol_t = solution[0][2]
    if True:
        from sympy import Matrix
        from sympy import solve
        c = sympy.symbols("c")
        nlc1 = dv
        lin1 = v-c
        print('nlc1', nlc1)
        print('lin1', lin1)
        eq3m = Matrix([nlc1, lin1])
        zero0 =  Matrix([0,0])
        #eq3 = Eq(matr, c*0, evaluate=False)
        # unexplored option: evaluate=False
        eq3 = Eq(eq3m, zero0)
        sympy.pprint(eq3)
        print('gonna solv')
        #solutions = solve(eq3, (w,c),force=True, manual=True, set=True)
        #solutions = solve(eq3, (w,),force=True, manual=True, set=True)
        vars_list, _solution_set = solutions = solve(eq3, (w,),force=True, set=True)
        # ([w], {(-v**3 + v,)})
        # convert set to list
        print(list(_solution_set))
        solution_list = list(_solution_set)

        # return solution_list


        # without  `set=True`: returns list of solutions?
        #[(-v**3 + v,)]

    # solve: set=True:
    # simplify=True
    # "check=True (default) will run through the symbol tags to elimate unwanted solutions. If no assumptions are included, all possible solutions will be returned"
    # why didi I choose this? "manual=True (default is False). Do not use the polys/matrix method to solve a system of equations, solve them one at a time as you might “manually.”"
    import code
    #code.interact(local=locals())
    print('bye')
    exit()
    # v - v**3  - w + I == 0
    # w=...

# todo: null-clines
def plot_isocline(_model, param, ax, a, b, tau, I, color='k', style='--', opacity=.5, vmin=-1,vmax=1):
    """Plot the null iscolines of the Fitzhugh nagumo system"""
    v = np.linspace(vmin,vmax,100)

    q= symbolic_nullclines(_model, param)
    ax.plot(v, v - v**3 + I, style, color=color, alpha=opacity)
    # (v - a - b * w)/tau == 0
    # w=...
    ax.plot(v, (v - a)/b, style, color=color, alpha=opacity)
    # todo: adapt to the generatlised symbolic _model

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
    X,Y = np.meshgrid(x,y)
    print(X.shape)
    #exit()

    # dx,dy = fitzhugh_nagumo([X,Y],0,**param)
    dxdy = fitzhugh_nagumo_partial(_model, **param)[1]([X,Y], 0)

    dx,dy = dxdy[:2]

    # streamplot is an alternative to quiver
    # that looks nicer when your vector filed is
    # continuous.
    ax.streamplot(X,Y,dx, dy, color=(0,0,0,.1))

    ax.set(xlim=(xrange[0], xrange[1]), ylim=(yrange[0], yrange[1]))

fig, ax = plt.subplots(1, 3, figsize=(20, 6))
for i, sc in enumerate(scenarios):
    xrange = (-1, 1)
    yrange = [(1/sc['b'])*(x-sc['a']) for x in xrange]
    plot_vector_field(ax[i], sc, xrange, yrange)
    ax[i].set(xlabel='v', ylabel='w',
          title='{}'.format(sc))
##############################################

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
        J = jacobian_fitznagumo(e[0],e[1], **param)
        eqstability[i].append(stability(J))
print(eqstability)

plt.show()
#brew install graphviz
