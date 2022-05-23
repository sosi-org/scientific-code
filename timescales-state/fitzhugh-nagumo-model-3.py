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

fitzhugh = True

mi = 5

if mi == 1:
    # doulcier
    from models import fitzhugh_nagumo_1_doulcier
    model1 = fitzhugh_nagumo_1_doulcier.model1
    scenarios = fitzhugh_nagumo_1_doulcier.scenarios
    initial_values = fitzhugh_nagumo_1_doulcier.initial_values
    get_ranges = fitzhugh_nagumo_1_doulcier.get_ranges
    get_wide_ranges = fitzhugh_nagumo_1_doulcier.get_wide_ranges
    pname = fitzhugh_nagumo_1_doulcier.pname

elif mi == 2:
    # sohale
    from models import fitzhugh_nagumo_2
    model1 = fitzhugh_nagumo_2.model1
    scenarios = fitzhugh_nagumo_2.scenarios
    initial_values = fitzhugh_nagumo_2.initial_values
    get_ranges = fitzhugh_nagumo_2.get_ranges
    get_wide_ranges = fitzhugh_nagumo_2.get_wide_ranges
    pname = fitzhugh_nagumo_2.pname

elif mi == 3:
    # Morris-Lecar's variation 1 by jonmarty
    from models import morris_lecar_v1
    model1 = morris_lecar_v1.model2
    scenarios = morris_lecar_v1.scenarios
    initial_values = morris_lecar_v1.initial_values
    get_ranges = morris_lecar_v1.get_ranges
    get_wide_ranges = morris_lecar_v1.get_wide_ranges
    pname = morris_lecar_v1.pname

    import yaml

elif mi == 4:
    # morris_lecar variation 2 (FAILED)
    # solver not implemented
    # FAILED
    from models import morris_lecar_v2
    model1 = morris_lecar_v2.model2
    scenarios = morris_lecar_v2.scenarios
    initial_values = morris_lecar_v2.initial_values
    get_ranges = morris_lecar_v2.get_ranges
    get_wide_ranges = morris_lecar_v2.get_wide_ranges
    pname = morris_lecar_v2.pname

elif mi == 5:
    # uc
    from models import rotate_unit_1
    model1 = rotate_unit_1.model1
    scenarios = rotate_unit_1.scenarios
    initial_values = rotate_unit_1.initial_values
    get_ranges = rotate_unit_1.get_ranges
    get_wide_ranges = rotate_unit_1.get_wide_ranges
    pname = rotate_unit_1.pname
else:
    raise Exception('model index wrong')

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
ηM_VARS = 0  # dynamics vars
ηM_DYN = 2   # The dynamics
ηM_TIME = 1
ηM_PARAMS = 3
ηM_EXOG_INPUTS = 4 # exogenous inputs. I_ext

# The io format, for (return value of) `evaluate_model_dvdw_partial()`
ηLAMB_NPARRAY = 0 #M_SYMBOLIC_.I
ηLAMB_ASLIST = 1 # M_LAMBDA_.I

# for data structure for "integration"/ode results
ηSIM_NDIM = 0
ηSIM_BINS = 1   # dimention for "bin"s if simulation and integration
                # means "across bins"
                # was S IM_I_BIN

# Try to avoid the following
# Avoid such hardcoded indexing though
# every usage will be a technical debt when we generalise to higher dimensional equations
# will need to be parametrised in proper variables (not consts)
_ηVAR_V = 0
_ηVAR_W = 1
_ηNDIM_VW = 2  # "across dims". try to avoid

# (_ηNDIM_VW, 1500) is more readable than (2, 1500)


_model = model1()

def get_free_variables(_model, params):
    #(dV,dN) = _model[ηM_DYN]; print(dV); print(dV.subs(params))
    subtsed = [dX.subs(params) for dX in _model[ηM_DYN]]
    free_variables = sympy.Tuple(*subtsed).free_symbols
    return free_variables

def show_model_info(_model, params):
    print(_model[ηM_DYN])

    free_variables = get_free_variables(_model, params)
    print('free variables:', free_variables)
    ndims1 = len(_model[ηM_VARS])
    ndims2 = len(_model[ηM_DYN])
    assert len(free_variables) == ndims1
    assert len(free_variables) == ndims2
    # The only free variables should be `_model[ηM_VARS]`, everything else must be concrete numbers

show_model_info(_model, scenarios[0])

def get_initial_array(vars_tuple, initial_values):
    # [0,0]
    # param['initial'] if 'initial' in param else [0,0]
    #initial_ = sympy.Tuple(*_model[ηM_VARS]).subs(initial_values)
    assert type(initial_values) is dict
    st = sympy.Tuple(*vars_tuple)
    print('st', st)
    print(initial_values)
    st_ = st.subs(initial_values)
    print('st_', st_)
    print('s1', vars_tuple[0].subs(initial_values))
    print('s2', vars_tuple[1].subs(initial_values))
    initial_ = list(st_)
    print('initial_', initial_)

    if False:
        V, N, = sympy.symbols( "V, N", real=REAL_ONLY)
        V, N, = sympy.symbols( "V, N")
        print('?1',V.subs(initial_values))
        print('?2',V.subs({'V':4}))


    print('vars_tuple', vars_tuple, type(vars_tuple))
    m = sympy.Matrix([*vars_tuple])
    print('m', m)
    m0 = sympy.Matrix([initial_values[vars_tuple[i].name] for i in range(len(vars_tuple))])
    print('m0', m0)
    eq = sympy.Eq(m, m0)
    print('eq', eq)

    vars_list, _solution_set = \
     sympy.solve(eq, vars_tuple, force=True, set=True)
    print(_solution_set)

    a = [initial_values[var.name] for var in vars_tuple]
    print(a)
    # import pdb
    # #  #pdb.pm()
    # pdb.set_trace(header='debug mode:')
    # exit()
    #return initial_
    return a

def testsp():
    import sympy
    X = sympy.symbols( "X")
    print('i', (1*X).subs({'X': 1.01})) # works
    X = sympy.symbols( "X", real=True) # causes problem
    print('ii', (1*X).subs({'X': 2.02}))
    X = sympy.symbols( "X", real=False) # even this
    print('iii', (1*X).subs({'X': 3.03}))
    print()
    X = sympy.symbols( "X")
    print('i', (1*X).subs({'X': 1.0}))
    #X = sympy.symbols( "X", real=True)
    print('ii', (1*X).subs({'X': 2.02}))
    #X = sympy.symbols( "X")
    print('iii', (1*X).subs('X', 3.03))
    #import pdb
    #pdb.set_trace(header='debug mode:')

testsp()


get_initial_array(_model[ηM_VARS], initial_values[0])

def check_model(_model, params, initial_values):

    ndims1 = len(_model[ηM_VARS])
    ndims2 = len(_model[ηM_DYN])
    assert ndims1 == ndims2

    free_variables = get_free_variables(_model, params)
    assert len(free_variables) == ndims1

    va = get_initial_array(_model[ηM_VARS], initial_values)
    print(va)
    #for v in va:
    #    print('>', v)
    #    assert len(v.free_symbols) == 0


check_model(_model, scenarios[0], initial_values[0])



UPSAMPLEx = 15
#UPSAMPLEx = 5
# UPSAMPLEx = 1

SIMU_TIME=1200
SIMU_STEPS1=1500*UPSAMPLEx
SIMU_STEPS2=1000*UPSAMPLEx

time_span = np.linspace(0, SIMU_TIME, num=SIMU_STEPS1)


def evaluate_model_dvdw_partial(_model, **model_params_and_inputs):
    '''
    substitute params, Lambdify, partial evaluate of dvdw(v,w)
    '''
    (dyn_vars, t, dyn_derivs, model_params, model_inputs) = _model
    #(v,w), t, (dvdt,dwdt), (a, b, tau), (I,) = (dyn_vars, t, dyn_derivs, model_params, model_inputs)
    print((dyn_vars, t, dyn_derivs, model_params, model_inputs))

    dyn_vars2 = tuple([deriv_elem.subs(model_params_and_inputs) for deriv_elem in dyn_derivs])
    model_lamb = sympy.lambdify((*dyn_vars,t), dyn_vars2, dummify=False)
    ndim = len(dyn_derivs)

    # ηLAMB_NPARRAY: receives ndarray, returns ndarray
    def p(x, t):
        # either args=((_ηNDIM_VW,n), (n,))  or args=((_ηNDIM_VW,), ())
        # is_scalar =
        #print(x.shape, 'x.shape')
        assert x.shape[ηSIM_NDIM] == ndim
        x0x1 = tuple(x) # x[0,:],x[1,:]
        y = model_lamb(*x0x1, t)
        assert len(y) == _ηNDIM_VW
        #print(y[0].shape)
        dvdw = np.array(y)
        #print(dvdw.shape)
        assert len(dvdw.shape) == _ηNDIM_VW or dvdw.shape==(_ηNDIM_VW,)
        assert dvdw.shape == dvdw.shape == (ndim,) or (ndim, x.shape[ηSIM_BINS])
        # either (_ηNDIM_VW,n)  or (_ηNDIM_VW,)
        return dvdw

    # ηLAMB_ASLIST: receives list, returns list
    def p3(xy_list,t):
          # print(len(xy_list))
          # print(xy_list[0].shape)
          assert len(xy_list) == ndim
          # to make suree (*xy,t) align correctly
          dvdw = model_lamb(*xy_list, t)
          assert len(dvdw) == ndim
          return dvdw

    # ηLAMB_NPARRAY, ηLAMB_ASLIST
    return p,p3

# ic: Array containing the value of y for each desired time in t, with the initial value y0 in the first row.

def simulate_perturbed(
        param, initial,
        dmax=0.5,
        time_span=np.linspace(0, SIMU_TIME, SIMU_STEPS2),
        number=20
  ):
    # We start from the resting point...
    ic = scipy.integrate.odeint(
          evaluate_model_dvdw_partial(_model, **param)[ηLAMB_NPARRAY],
          y0=initial,
          #t= np.linspace(0,SIMU_STEPS2-1, SIMU_STEPS2)
          t= np.linspace(0,SIMU_TIME*UPSAMPLEx, SIMU_STEPS2)
      )[-1]
    # and do some displacement of the potential.
    veocities = []
    traj = []
    for displacement in np.linspace(0, dmax, number):
        traj.append(
          scipy.integrate.odeint(
            evaluate_model_dvdw_partial(_model, **param)[ηLAMB_NPARRAY],
            y0=ic+np.array([displacement,0]),
            #full_output=True, adds a second entry to output
            t=time_span
        ))
        solu = traj[-1]  # v, w = sol.T
        vel_eval = evaluate_model_dvdw_partial(_model, **param)[ηLAMB_NPARRAY](solu.T, time_span)

        # vel_eval.shape: # (_ηNDIM_VW, 1500)
        veocities.append(vel_eval)
    return traj, veocities

# Do the numerical integration.
trajectories = {} # We store the trajectories in a dictionnary, it is easier to recover them.
velo_s = {}
for i,param in enumerate(scenarios):
    initial_ = get_initial_array(_model[ηM_VARS], initial_values[i])

    trajectories[i], velo_s[i] \
        = simulate_perturbed(param, initial_, number=4, time_span=time_span, dmax=0.5)


# Draw the trajectories.
fig, ax = plt.subplots(1, len(scenarios), figsize=(5*len(scenarios),5))
for i,param in enumerate(scenarios):
        ax[i].set(xlabel='Time', ylabel='v, w',
                     title='{:<8} {}'.format(pname, param))
        for j in range(len(trajectories[i])):
            v = ax[i].plot(time_span,trajectories[i][j][:,_ηVAR_V], color='C0')
            w = ax[i].plot(time_span,trajectories[i][j][:,_ηVAR_W], color='C1', alpha=.5)
        ax[i].legend([v[0],w[0]],['v','w'])
plt.tight_layout()


def minmax(simulation):
    z0 = np.zeros((simulation.shape[ηSIM_NDIM],1), dtype=simulation.dtype)
    z1 = np.ones((simulation.shape[ηSIM_NDIM],1), dtype=simulation.dtype) * 0.0000001

    simulation = np.concatenate((simulation, z0, z1), axis=ηSIM_BINS)
    simulation = simulation[:, ~(np.isnan(simulation).any(axis=ηSIM_NDIM))]
    simulation = simulation[:, (np.isfinite(simulation).any(axis=ηSIM_NDIM))]
    assert simulation.shape[ηSIM_BINS] > 0

    mins = np.min(simulation, axis=ηSIM_BINS)
    maxs = np.max(simulation, axis=ηSIM_BINS)
    effective_ranges = np.array([mins, maxs ]).T
    print('effective_ranges:', effective_ranges)
    sqmin = np.max(mins, axis=0)
    sqmax = np.min(maxs, axis=0)
    return (sqmin, sqmax, effective_ranges)

# plot First derivative
fig, ax = plt.subplots(1, len(scenarios), figsize=(5*len(scenarios),5))
# i = simulation/experiment (with different parameters)
for i,param in enumerate(scenarios):
    velocities = velo_s[i]
    # j = repeats = 4
    for j in range(len(velocities)):
        simulation = velocities[j].copy()  # [2,1500]
        # fake rescaling W:
        # simulation[_ηVAR_W,:] = simulation[_ηVAR_W,:] * param['tau']
        d_v = simulation[_ηVAR_V,:]
        d_w = simulation[_ηVAR_W,:]
        (sqmin,sqmax, effective_ranges) = minmax(simulation)
        ax[i].plot(np.array([sqmin,sqmax]), np.array([sqmin,sqmax]), 'r--', alpha=.2)
        ax[i].plot(d_v, d_w, 'k-')
    np.set_printoptions(precision=2)
    ax[i].set(xlabel=r'$\dot{V}$', ylabel=r'$\dot{W}$',
        title='v:{}, w:{}\n {:<8}'.format(effective_ranges[0][:], effective_ranges[1][:], pname));
    (xrange, yrange) = get_ranges(param)
    # ax[i].set(xlim=(xrange[0], xrange[1]), ylim=(yrange[0], yrange[1]))

# plot Second derivative
fig, ax = plt.subplots(1, len(scenarios), figsize=(5*len(scenarios),5))
# i = simulation/experiment (with different parameters)
for i,param in enumerate(scenarios):
    velocities = velo_s[i]
    # j = repeats = 4
    for j in range(len(velocities)):
        simulation = velocities[j]  # [2,1500]
        d_v0 = simulation[_ηVAR_V,:]
        d_w0 = simulation[_ηVAR_W,:]
        d_t = time_span
        d2_v = np.diff(d_v0)/np.diff(time_span)
        d2_w = np.diff(d_w0)/np.diff(time_span)
        (sqmin,sqmax, effective_ranges) = minmax(np.array([d2_v, d2_w]))
        ax[i].plot(np.array([sqmin,sqmax]), np.array([sqmin,sqmax]), 'r--', alpha=.2)
        ax[i].plot(d2_v, d2_w, 'k-')
    np.set_printoptions(precision=2)
    ax[i].set(xlabel=r'$\ddot{V}$', ylabel=r'$\ddot2{W}$',
        title='ACCELERATION\nv:{}, w:{}\n {:<8}'.format(effective_ranges[0][:], effective_ranges[1][:], pname))
    (xrange, yrange) = get_ranges(param)
    ax[i].set(xlim=(xrange[0], xrange[1]) ) #, ylim=(yrange[0], yrange[1]))

##############################################


def safe_evaluate(ncl, _s):
    # ncl is a lambdified (using numpy)
    #print('_s', _s)
    vw_tuple = ncl(_s) # t
    assert len(vw_tuple) == 2, 'only implemented for 2 dimentions'
    #print('vw_tuple', vw_tuple)
    #print()
    #ax.plot(_s, vw_tuple, style, color=color, alpha=opacity)
    print('shape@', vw_tuple[_ηVAR_V].shape)
    #print('v@', vw_tuple[_ηVAR_V])
    #print('w@', vw_tuple[_ηVAR_W])
    print('ok')
    v_ = vw_tuple[_ηVAR_V]
    w_ = vw_tuple[_ηVAR_W]
    # Sometimes v_ or w_ may be scalar:
    v_ = v_ + 0.0 * _s
    w_ = w_ + 0.0 * _s
    # why is w_ sometimes a single number?!
    assert v_.shape == w_.shape, 'mismatching shapes'

    print('-----',v_.shape,v_.dtype,'  w',w_.shape, w_.dtype) # why float

    def _isreal0(npa):
        return np.isreal(v_)
    def _report_small_imag(npa):
        #np.set_printoptions(precision=7, suppress=True)
        vv = np.imag(v_)
        vv = vv[np.absolute(vv) < 0.001 ]
        print(np.sort(vv))
        # typical value: 2.00e-15 and smaller !
        exit()

    def _isreal(npa):
        # _report_small_imag(v_)
        ε = 0.00001
        return np.absolute(np.imag(v_)) < ε
    # important lesson: never use raw `np.isreal()`. Use ε and an extra `np.real()`
    wch_ =np.logical_and(_isreal(v_),_isreal(w_))
    v_ = v_[wch_]
    w_ = w_[wch_]
    #v_ = np.abs(v_)
    #w_ = np.abs(w_)
    v_ = np.real(v_)
    w_ = np.real(w_)

    return (v_, w_)


# todo: why is v defined here?
def symbolic_nullclines(whichi, _model, param):
    # symbolic (algebraic) null-clines:
    print()
    print(param)
    # dv = _model[ηM_DYN][_ηVAR_V].subs(param)
    # dw = _model[ηM_DYN][_ηVAR_W].subs(param)
    ndim = len(_model[ηM_DYN])
    # with substitution
    dvdw_subs = [_model[ηM_DYN][i].subs(param) for i in range(ndim)]
    w  = _model[ηM_VARS][_ηVAR_W]
    v  = _model[ηM_VARS][_ηVAR_V]

    from sympy import Eq
    from sympy import Matrix
    from sympy import solve

    # _s is the free parameter for the "curve". Since it's aa null-cline, we need (exactly) one free ariable for it
    _s = sympy.symbols("s", real=False)

    # v - v**3  - w + I == 0
    # w == _s

    #nlc1 = dv  # dv=0
    #nlc1 = dw # dw=0

    # without substitution
    # nlc1 = _model[ηM_DYN][whichi]
    # with substitution
    nlc1 = dvdw_subs[whichi]

    # The "scan"ning line/shape (guiding shape)
    # easy:
    # lin1 = v-_s # single root solution
    # difficult:
    lin1 = w-_s  # multiple roots
    # lin1 = v-_s+w
    # lin1 = v*v-_s+w*w

    print('nlc1', nlc1)
    print('lin1', lin1)

    eq3m = Matrix([nlc1, lin1])
    zero0 =  Matrix([0,0])
    # also: zero= _s*0 ,  0 * dv , sympy.core.numbers.Zero
    # unexplored option: `Eq(..., evaluate=False)`
    eq3 = Eq(eq3m, zero0)
    sympy.pprint(eq3, use_unicode=True)
    print('Going to solve')
    ##################

    vars_list, _solution_set = \
        solve(eq3, (v,w),force=True, set=True)
        # sympy.nsolve

    ##################
    # Alternative: sympy.solve([eq1,eq2], (w,), force=True, manual=True, set=True)
    # ([w], {(-v**3 + v,)})
    # vars_list: # (v,w)
    # convert set to list. len = number of roots:
    solution_list_sympy = list(_solution_set)
    # each solution is a tuple, for (v,w). has 2=ndim elements

    solution_lambda_list = [
        # Lesson learned: `modules=["scipy", "numpy"]`` is essential for the correct calculation of nullclines (generalised approach). Especiaally when there are multiple roots.
        sympy.lambdify((_s,), solt_vw, dummify=False, modules=["scipy", "numpy"] )
            for solt_vw in solution_list_sympy ]

    safe_eval_lambda_list = [
        # Lesson: Double lambda trick to simulate closures, by @max-shawabkeh
        (lambda ncl:(lambda s: safe_evaluate(ncl, s)))(ncl)
            for ncl in solution_lambda_list
    ]

    # checksol()
    # check_assumptions()

    # each is a function that returns a tuple of size `ndim`

    return solution_list_sympy, solution_lambda_list, safe_eval_lambda_list

# Nullclines

square_nc = False

def plot_isocline(whichi, _model, param, ax, sc2, color='k', style='--', opacity=.5, vmin=-1,vmax=1):
    """Plot the null iscolines of the Fitzhugh nagumo system"""

    # a, b, tau, I = sc2
    nc_solution_list, nc_lambdas, safe_eval_lambdas = symbolic_nullclines(whichi, _model, param)

    print(nc_solution_list)
    ctr = 0
    #for ncl in nc_lambdas:
    for safeeval_lam in safe_eval_lambdas:
      print()
      print('root number', ctr)
      # symbolic: root number (s, -84.8528137423857*I)
      print('symbolic:', nc_solution_list[ctr])
      s_np = np.linspace(-2,+2, 1000)
      _s = s_np
      #(v_, w_) = safe_evaluate(ncl, _s)
      (v_, w_) =  safeeval_lam(_s)
      σ = 0.01 / 10
      v_ = v_ + np.random.randn(*v_.shape) * σ
      w_ = w_ + np.random.randn(*w_.shape) * σ

      COLS = [['r', 'm',]*20, [ 'g','b', ]*20 ]

      ctr += 1
      ax.plot(v_, w_, '-', color=COLS[whichi][ctr], linewidth=5, alpha=0.3)
      # todo: indent

    v_np = np.linspace(vmin,vmax,100)
    if False:
        ax.plot(v_np, v_np - v_np**3 + I, style, color=color, alpha=opacity)
        # (v - a - b * w)/tau == 0
        # w=...
        ax.plot(v_np, (v_np - a)/b, style, color=color, alpha=opacity)



fig, ax = plt.subplots(1, len(scenarios), figsize=(18, 6))
for i, sc in enumerate(scenarios):
    for whichj in range(2):
        plot_isocline(whichj, _model, sc, ax[i], sc)

    if (square_nc):
        #ax.set_aspect('equal', adjustable='box')
        # adjustable='datalim')
        ax.set_aspect(sc['tau'], adjustable='box')

    ax[i].set(xlabel='v', ylabel='w',
              title='{}'.format(sc))
    (xrange, yrange) = get_ranges(sc)
    ax[i].set(xlim=(xrange[0], xrange[1]) ) #, ylim=(yrange[0], yrange[1]))

##############################################

def plot_vector_field(ax, param, xrange, yrange, steps=50*2):
    # Compute the vector field
    x = np.linspace(xrange[0], xrange[1], steps)
    y = np.linspace(yrange[0], yrange[1], steps)
    X,Y = np.meshgrid(x,y)  # each shape: (50,50)

    dxdy = evaluate_model_dvdw_partial(_model, **param)[ηLAMB_ASLIST]([X,Y], 0.0)

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

fig, ax = plt.subplots(1, len(scenarios), figsize=(20, 6))
for i, sc in enumerate(scenarios):
    (xrange, yrange) = get_ranges(sc)
    plot_vector_field(ax[i], sc, xrange, yrange)
    ax[i].set(xlabel='v', ylabel='w',
          title='{}'.format(sc))
    ax[i].set(xlim=(xrange[0], xrange[1]), ylim=(yrange[0], yrange[1]))

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

    print(dyn_vars)
    print(model_params)
    print(model_inputs)
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

plt.show()
eqnproot = {}
for i, param in enumerate(scenarios):
    eqnproot[i] = find_roots(**param)

#
eqstability = {}
for i, param in enumerate(scenarios):
    eqstability[i] = []
    for e in eqnproot[i]:
        J = jacobian_fitznagumo(e[_ηVAR_V],e[_ηVAR_W], **param)
        eqstability[i].append(stability(J))
print(eqstability)

plt.show()
#brew install graphviz
