import sympy

# This program generates automatic c-code for gradient of
# implicit objects with complicated formulas. It generates the gradient automatically.

# from __future__ import division
import sympy
from sympy import *

sympy.init_printing()


def surface_ellipsoid():
    # Plane:
    ux, uy, uz = symbols('ux uy uz', real=true)
    vx, vy, vz = symbols('vx vy vz', real=true)
    wx, wy, wz = symbols('wx wy wz', real=true)
    x0, y0, z0 = symbols('x0 y0 z0', real=true)

    uu, vv, ww = symbols('uu vv ww', real=true)

    # Part 1: UVW coordinates

    # Sphere parameters:
    # Center
    C0 = Matrix([x0, y0, z0])
    # Plane: side vectors
    U = Matrix([ux, uy, uz])
    V = Matrix([vx, vy, vz])
    W = Matrix([wx, wy, wz])
    # UV = Matrix([U.T, V.T]).T
    UVW = Matrix([U.T, V.T, W.T]).T

    # I wish I had SDF. That would be automatically geodesic.

    # Local cordinates:
    # todo: rename: uvw
    abc = Matrix([uu, vv, ww])

    point_in_sphere_uvw1 = UVW * abc + C0
    # point_in_sphere_uvw2 = norm(UVW * abc) - 1 #  = 0
    # pair them together?
    # No, it's just one equation:
    # implicit
    ellipsoid_surface_eq = uu*uu + vv*vv + ww*ww - 1
    ellipsoid_surface_expl = point_in_sphere_uvw1

    # sympy.pprint(point_in_sphere_uvw1)
    sympy.pprint(ellipsoid_surface_eq)
    # But this is not what we need

    variables = (uu, vv, ww)
    parameters = (U, V, W, C0)
    eq = ellipsoid_surface_eq
    # world_variables = (x,y,z) # no such thing

    return {
        'vars': variables,  # local vars
        # 'x,y,z': world_variables, # local vars ---> 'expl-eq'
        'params': parameters,
        'impl-eq': eq,
        'expl-eq': ellipsoid_surface_expl,
        'unknowns': (uu, vv, ww),
        'desired_vars': [uu, vv, ww],
    }


# invalid
def surface_sphere_polar():
    # Part 2: Polar coordinates
    # Formulation 2:

    φ, θ, r = symbols('φ θ r', real=true)
    x0, y0, z0 = symbols('x0 y0 z0', real=true)
    C0 = Matrix([x0, y0, z0])

    x = r * sin(φ) * cos(θ) + x0
    y = r * sin(φ) * sin(θ) + y0
    z = r * cos(φ) + z0
    print('-------')
    sympy.pprint(x)
    sympy.pprint(y)
    sympy.pprint(z)

    # Implicit
    # Oops, not needed: This r is not that r
    polar_sphere_eq = x*x + y*y + z*z - r*r

    return {
        'vars': (φ, θ, r),
        'params': (C0),
        # Implicit. Constraints. In fugure, it can be more than one
        'impl-eq': polar_sphere_eq,
        # Explicit
        # Explicit: Cartesian (x,y,z) (in host space: aug-dims representation) (vector3D)
        'expl-eq': Matrix([x, y, z]),
        'unknowns': (φ, θ, r),
        'desired_vars': [φ, θ, r],
    }


# simplified
def surface_sphere_polar_simple():
    # Part 3: Polar coordinates
    # Formulation 3:
    # sphere (1) at 0

    φ, θ, r = symbols('φ θ r', real=true)

    x = r * sin(φ) * cos(θ)
    y = r * sin(φ) * sin(θ)
    z = r * cos(φ)
    print('-------')
    sympy.pprint(x)
    sympy.pprint(y)
    sympy.pprint(z)

    # Implicit
    # No need perhaps
    # This would be trivial (always hold. Hence, no gradiant?) (todo: check)
    # Will set r to 1. Just that. (So, not trivial then)
    polar_sphere_eq_ = x*x + y*y + z*z - 1
    polar_sphere_eq = r*r - 1

    # r will be a function of t

    return {
        'vars': (φ, θ, r),
        'params': (),
        # Implicit. Constraints. In fugure, it can be more than one
        'impl-eq': polar_sphere_eq,
        # Explicit
        # Explicit: Cartesian (x,y,z) (in host space: aug-dims representation) (vector3D)
        'expl-eq': Matrix([x, y, z]),
        'unknowns': (φ, θ, r),
        'desired_vars': [φ, θ, r],
    }

# A model has a constaint, but also "joint" iing it with another model will make the other mode (ray) a second constaint


def surface_sphere_polar_simpler_pauli():
    # complex-number case (Pauli matrix?)
    # sphere (1) at 0

    φ, θ, r = symbols('φ θ r', real=true)

    x = r * sin(φ) * cos(θ)
    y = r * sin(φ) * sin(θ)
    z = r * cos(φ)

    # Implicit
    polar_sphere_eq = r*r - 1
    # r will be a function of t

    return {
        'vars': (φ, θ, r),
        'params': (),
        'impl-eq': polar_sphere_eq,
        # Explicit
        'expl-eq': Matrix([x, y, z]),
        'unknowns': (φ, θ, r),
        'desired_vars': [φ, θ, r],
    }

# The insightful unit sphere:


def surface_simple_unit_sphere1():
    # complex-number case (Pauli matrix?)
    # sphere (1) at 0

    xx, yy, zz = symbols('xx yy zz', real=true)
    # Insightful:
    # no need for paramsm in fact (new vision: from representation theory)
    #  This means, we also can add optional (internal param) in between, but not really needed.
    # Then afterways, we can have a two-way way to corespond the points x,y,z to params (φ, θ, r), but aftger computing.
    #   Hence, the acnalytical formulas dont need to be interms of those parameters (simpler symbolical "solve"s, also simpler numerical and compute-code)
    # That can be yet-another pluggable part (lego, interoperatble, joint-able, grip: "co-consider")

    # Implicit
    unit_sphere_eq = xx*xx + yy*yy + zz*zz - 1
    # r will be a function of t

    return {
        'vars': (xx, yy, zz),
        'params': (),
        'impl-eq': unit_sphere_eq,
        # Explicit
        'expl-eq': Matrix([xx, yy, zz]),
        'unknowns': (xx, yy, zz),
        'desired_vars': [xx, yy, zz],
    }


various_forms = []

various_forms.append(surface_ellipsoid())
# invalid:
# various_forms.append(surface_sphere_polar())
various_forms.append(surface_sphere_polar_simple())
various_forms.append(surface_sphere_polar_simpler_pauli())

# various_forms.append(polar_sphere_eq)
# chosen_forms = [various_forms[2]]

chosen_forms = [surface_simple_unit_sphere1()]

# ------------------------------


def make_ray():
    # Now  ray (to cross with each):

    #  Ray:
    ox, oy, oz = symbols('ox oy oz', real=true)
    dx, dy, dz = symbols('dx dy dz', real=true)

    # Ray origin:
    O = Matrix([ox, oy, oz])
    # Ray direction:
    D = Matrix([dx, dy, dz])
    # Ray drive (t,l)
    t = symbols('t', real=true)

    # rename: rayeq ?
    ray = O + t * D
    print(ray)
    sympy.pprint(ray)

    # wow, already
    ray_thing = {
        'vars': (t),
        'params': (O, D),
        # No constraints
        'impl-eq': (),
        # Explicit: vector3D
        'expl-eq': ray,
        'unknowns': (t),
        'desired_vars': [t],
    }
    return ray_thing

# -------------------------
# Now cross ray with each:


def transform_using_gcd(soltions_tuple):
    # todo: GCD of all but not 't' is desired
    # Maybe: ((u2,v2,w2),t2), gcd = transform_using_gcd(...).
    #   (u2,v2,t2), gcd =.
    #   gcd each group: [((u2,v2,w2),gcd_uvw), ((u2,v2,w2,t2),gcd_uvwt) ] = transform_using_gcd([(u2,v2,w2),gcd_uvw), (u2,v2,w2,t2)])
    # soltuple = tuple(solutions)
    # soltuple_gcd, gcd = transform_using_gcd(tuple(solutions))
    # soltuple_gcd, gcd = transform_using_gcd(soltuple)
    # transform_using_gcd(tuple(solutions), t)
    if False:
        # Did not work with `sol_t` (in Flat Plane)
        # Maybe no need to do it for `t`?
        # Then for t, see if it cimplifies it, then use it.
        d1 = gcd(sol_uu, sol_vv)
        d2 = gcd(d1, sol_ww)
        print('d1 = ', d1)
        print('d2 = ', d2)
        d = d1
        print('gcd = ', 1/d)

        denom = 1/d
        print('a =', sol_uu * denom, '/', denom)
        print('b =', sol_vv * denom, '/', denom)
        print('t =', sol_t * denom, '/', denom)
    print('--------------------')
    assert len(soltions_tuple) == 4, "(u,v,w,t) or (x,y,z,t)"
    # these are in the represeantion space
    (u, v, w, t) = soltions_tuple
    d1 = gcd(u, v)
    print(d1)
    d2 = gcd(u, w)
    print(d2)
    d3 = gcd(v, w)
    print(d3)
    d4 = gcd(u, t)
    print(d4)

    # The gcd s turned out ot be the same.
    # So custom tailor-made gcd is god enough

    # If this (assert-as-if-condition), and not to constrain the future changes
    # for other cases I haven't thought
    # idea: assert can be apractical solution for nultiple purposes (limit changes, document, protect, reminder to change "this", etc: change THIS or cahgen THAT, or change unknown (invariance check : check somehwre, but since it's as-early-aspossible , we are hopeful that we can find it, even thought we have not separated the causes and have all constraints in one "invariant" together))
    assert d1 == d2 and d1 == d3 and d1 == d4

    gcd_d = d1
    return (u*gcd_d, v*gcd_d, w*gcd_d, t*gcd_d), gcd_d


ray_thing = make_ray()

for form_i, form in enumerate(chosen_forms):
    # set of desired vars: desired_set, unkowns, fast_vars, vars, surface_vars
    # May be any number of DoF: Local DoF. (Usually DoF equals to DoF of host space): (u,v,w) or (φ,θ,r)
    surface_vars = form['desired_vars']
    # Only (t,)  (DoF = 1)
    ray_vars = ray_thing['desired_vars']
    # desired_set_tuple = (*surface_vars, *ray_vars)
    # print('desired_set', desired_set)
    desired_vars_tuple = (*surface_vars, *ray_vars)
    print('desired_vars_tuple', desired_vars_tuple)

    # for desiredv_j, desired_var in enumerate(desired_set):
    if True:
        vars = form['vars']
        params = form['params']
        # Implicit equation for each: f(...) = 0
        # eq = form['eq']
        ieq = form['impl-eq']
        # Explicit equation for each: (x,y,z) := ...  host_eq
        exeq = form['expl-eq']
        # todo: repeated code
        rayeq = ray_thing['expl-eq']
        eq0 = exeq - rayeq
        print('equation == 0 :')
        sympy.pprint(eq0)
        # desired_vars_tuple = (desired_var,)

        # Equation
        from sympy import Eq as Equation
        # Solver

        zero0 = Matrix([0, 0, 0])
        # eq = Equation(plane - ray, 0) # Does not work. Obviously.
        # eq = Equation(eq0, zero0)
        # Unike the old one, we have two equaiotns: Since the plat plane is not enough (one equation), but we increase diention,a dn the constain it.
        # ieq = constrain
        # exeq - ray = higher-dimension repr + clams
        # eq = Equation((exeq - ray, ieq), zero0)

        # todo: repeated code
        rayeq = ray_thing['expl-eq']
        eq1 = Equation(exeq - rayeq, zero0)
        eq2 = Equation(ieq, 0)
        # 3 (higher-dim) + 1 (scalar)
        # Later on, add Symplectic: add to 36, and add constraints.

        print('Equations (2) to solve:')
        sympy.pprint(eq1, use_unicode=True)
        sympy.pprint(eq2, use_unicode=True)

        # desired_vars = (uu,vv,ww, t)
        # If I don't take all, it wil consider the other desired variables as knowns
        # all unkonwns need to be in one place
        # uknowns = ,
        print('To solve eq. set for', desired_vars_tuple)
        FAKE = False
        if FAKE:
            # fake the solution:
            solution = [desired_vars_tuple]
        else:
            # Actually Solve
            if False:
                solutions = solve((eq1, eq2), desired_vars_tuple,
                                  force=True, manual=True, set=True)

            solution = solve((eq1, eq2), desired_vars_tuple,
                             force=True, manual=True)

        print('Num solutions: ', len(solution))
        for solition_id in range(len(solution)):
            num_desired_vars = len(solution[solition_id])
            print("\n>>>>> Solution %d: has %d vars <<<<<" %
                  (solition_id, num_desired_vars))
            """
        # sphere coords
        sol_uu = solution[solition_id][0]
        sol_vv = solution[solition_id][1]
        sol_ww = solution[solition_id][2]
        # ray coord(1)
        sol_t = solution[solition_id][3]
        # sol_abt = solution[solition_id]
        """
            # rename
            solutions = []
            for j in range(num_desired_vars):
                varname = str(desired_vars_tuple[j])
                # Algebraic (symbolic) solution
                sol_v = solution[solition_id][j]

                """
          print('raw: sol_u =', sol_uu )
          print('raw: sol_v =', sol_vv )
          print('raw: sol_w =', sol_ww )
          print('raw: sol_t =', sol_t )
          """
                print('')
                print(f'raw: sol_{varname} =', sol_v)
                solutions += [sol_v]

            print('')

            print('Calculating gcd, for speed')

            # It is slow. How to save the result when done?

            print('----', len(solutions), solutions)
            soltuple_gcd, inv_gcd1 = transform_using_gcd(tuple(solutions))

            # a set of solutions (but not math-set)
            print(soltuple_gcd)
            ###################

            print('')
            for i in range(len(soltuple_gcd)):
                print(soltuple_gcd[i], '       /     ', 1/inv_gcd1)

            # Generalisation:
            # "Make efficient": by separating parts (that are re-used) hence, re-use computed values
            # Very useful for auto-diff, but on steroids
            # Separating (clive-age)
            # Shall I return (a,b) (in case of (1/b)), ro shall I return (a), (1/b)? IT seems the latter is good
            # todo (as an ossibility, no really omperative): rename: transform_using_gcd -> make_efficient_factorize_compute or make_efficient_factorize_compute_using_gcd

            # Now we  have a set of compute-efficient solutions,
            # for x,y,z
            # But we better need some paratmers
            # For taking visual represenation
            # Separate, or indide?
            #  separate: "joint" (put together) (co-consider)

            #In general, it is not multiple, but "substitute". In fact, a chain of them.

            print('')

      # todo: save/replace LaTeX for mapltotlib

exit()


denom = 1/d
print('a =', sol_a * denom, '/', denom)
print('b =', sol_b * denom, '/', denom)
print('t =', sol_t * denom, '/', denom)

'''
Outcome:
   denom =  dx*uy*vz - dx*uz*vy - dy*ux*vz + dy*uz*vx + dz*ux*vy - dz*uy*vx
   a = (dx*oy*vz - dx*oz*vy + dx*vy*z0 - dx*vz*y0 - dy*ox*vz + dy*oz*vx - dy*vx*z0 + dy*vz*x0 + dz*ox*vy - dz*oy*vx + dz*vx*y0 - dz*vy*x0) / denom
   b = (-dx*oy*uz + dx*oz*uy - dx*uy*z0 + dx*uz*y0 + dy*ox*uz - dy*oz*ux + dy*ux*z0 - dy*uz*x0 - dz*ox*uy + dz*oy*ux - dz*ux*y0 + dz*uy*x0) / denom
   t = (-ox*uy*vz + ox*uz*vy + oy*ux*vz - oy*uz*vx - oz*ux*vy + oz*uy*vx + ux*vy*z0 - ux*vz*y0 - uy*vx*z0 + uy*vz*x0 + uz*vx*y0 - uz*vy*x0) / denom
'''

# https://github.com/sohale/implisolid/blob/9c8b93005cbc5d82e171d9a1090e52b06a6c7290/js_iteration_1/symbolic_gradients.py
exit(1)
