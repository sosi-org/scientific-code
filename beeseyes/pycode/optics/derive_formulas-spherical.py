import sympy

# This program generates automatic c-code for gradient of
#implicit objects with complicated formulas. It generates the gradient automatically.

#from __future__ import division
import sympy
from sympy import *

sympy.init_printing()


# Plane:
ux, uy, uz = symbols('ux uy uz', real=true)
vx, vy, vz = symbols('vx vy vz', real=true)
wx, wy, wz = symbols('wx wy wz', real=true)
x0, y0, z0 = symbols('x0 y0 z0', real=true)


uu, vv, ww = symbols('uu vv ww', real=true)

various_forms = []
# Part 1: UVW coordinates

# Sphere parameters:
# Center
C0 = Matrix([x0,y0,z0])
# Plane: side vectors
U = Matrix([ux,uy,uz])
V = Matrix([vx,vy,vz])
W = Matrix([wx,wy,wz])
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


variables = (uu,vv,ww)
parameters = (U,V,W,C0)
eq = ellipsoid_surface_eq
# world_variables = (x,y,z) # no such thing

various_forms.append({
    'vars': variables, # local vars
    # 'x,y,z': world_variables, # local vars ---> 'expl-eq'
    'params': parameters,
    'impl-eq': eq,
    'expl-eq': ellipsoid_surface_expl,
    'unknowns': (uu,vv,ww),
    'desired_vars': [uu,vv,ww],
})

# Part 2: Polar coordinates
# Formulation 2:

φ,θ,r = symbols('φ θ r', real=true)
x = r * sin(φ) * cos(θ) + x0
y = r * sin(φ) * sin(θ) + y0
z = r * cos(φ) + z0
print('-------')
sympy.pprint(x)
sympy.pprint(y)
sympy.pprint(z)

# Implicit
polar_sphere_eq = x*x + y*y + z*z - r*r

various_forms.append({
    'vars': (φ,θ,r),
    'params': (C0),
    # Implicit. Constraints. In fugure, it can be more than one
    'impl-eq': polar_sphere_eq,
    # Explicit
    # Explicit: Cartesian (x,y,z) (in host space: aug-dims representation) (vector3D)
    'expl-eq': Matrix([x,y,z]),
    'unknowns': (φ,θ,r),
    'desired_vars': [φ,θ,r],
})

# various_forms.append(polar_sphere_eq)

# ------------------------------
# Now  ray (to cross with each):

# Ray:
ox, oy, oz = symbols('ox oy oz', real=true)
dx, dy, dz = symbols('dx dy dz', real=true)

# Ray origin:
O = Matrix([ox,oy,oz])
# Ray direction:
D = Matrix([dx,dy,dz])
# Ray drive (t,l)
t = symbols('t', real=true)

ray = O + t * D
print(ray)
sympy.pprint(ray)

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

# -------------------------
# Now cross ray with each:

chosen_forms = [various_forms[1]]
for form_i, form in enumerate(chosen_forms):
  # set of desired vars: desired_set, unkowns, fast_vars, vars, surface_vars
  # May be any number of DoF: Local DoF. (Usually DoF equals to DoF of host space): (u,v,w) or (φ,θ,r)
  surface_vars = form['desired_vars']
  # Only (t,)  (DoF = 1)
  ray_vars = ray_thing['desired_vars']
  #desired_set_tuple = (*surface_vars, *ray_vars)
  #print('desired_set', desired_set)
  desired_vars_tuple = (*surface_vars, *ray_vars)
  print('desired_vars_tuple', desired_vars_tuple)

  #for desiredv_j, desired_var in enumerate(desired_set):
  if True:
    vars = form['vars']
    params = form['params']
    # Implicit equation for each: f(...) = 0
    # eq = form['eq']
    ieq = form['impl-eq']
    # Explicit equation for each: (x,y,z) := ...  host_eq
    exeq = form['expl-eq']
    rayeq = ray_thing['expl-eq']
    eq0 = exeq - rayeq
    print('equation == 0 :')
    sympy.pprint(eq0)
    # desired_vars_tuple = (desired_var,)


    # Equation
    from sympy import Eq as Equation
    # Solver

    zero0 =  Matrix([0,0,0])
    #eq = Equation(plane - ray, 0) # Does not work. Obviously.
    # eq = Equation(eq0, zero0)
    # Unike the old one, we have two equaiotns: Since the plat plane is not enough (one equation), but we increase diention,a dn the constain it.
    # ieq = constrain
    # exeq - ray = higher-dimension repr + clams
    # eq = Equation((exeq - ray, ieq), zero0)
    eq1 = Equation(exeq - ray, zero0)
    eq2 = Equation(ieq, 0)
    # 3 (higher-dim) + 1 (scalar)
    # Later on, add Symplectic: add to 36, and add constraints.

    print('Equation2 to solve:')
    sympy.pprint(eq1, use_unicode=True)
    sympy.pprint(eq2, use_unicode=True)


    # desired_vars = (uu,vv,ww, t)
    # If I don't take all, it wil consider the other desired variables as knowns
    # all unkonwns need to be in one place
    # uknowns = ,
    print('To solve eq. set for', desired_vars_tuple)
    if False:
      # fake the solution:
      solution = [desired_vars_tuple]
    # Solve!
    if False:
      solutions = solve((eq1,eq2), desired_vars_tuple, force=True, manual=True, set=True)
    solution = solve((eq1,eq2), desired_vars_tuple, force=True, manual=True)


    print('Num solutions: ', len(solution))
    solutions = []
    for solition_id in range(len(solution)):
        num_desired_vars = len(solution[solition_id])
        print("Solution %d: has %d vars" % (solition_id, num_desired_vars))
        """
        # sphere coords
        sol_uu = solution[solition_id][0]
        sol_vv = solution[solition_id][1]
        sol_ww = solution[solition_id][2]
        # ray coord(1)
        sol_t = solution[solition_id][3]
        # sol_abt = solution[solition_id]
        """
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
          print(f'raw: sol_{varname} =', sol_v )
          solutions += [sol_v]
    print('Calculating gcd, for speed')
    if False:
        d1 = gcd(sol_uu, sol_vv)  # Did not work with `sol_t` (in Flat Plane)
        d2 = gcd(d1, sol_ww)
        print('d1 = ', d1)
        print('d2 = ', d2)
        d = d1
        print('gcd = ', 1/d)

        denom = 1/d
        print('a =', sol_uu * denom  , '/', denom)
        print('b =', sol_vv * denom  , '/', denom)
        print('t =', sol_t * denom  , '/', denom)
    print('--------------------')
    # It is slow. How to save the result when done?

    # todo: save/replace LaTeX for mapltotlib

exit()


denom = 1/d
print('a =', sol_a * denom  , '/', denom)
print('b =', sol_b * denom  , '/', denom)
print('t =', sol_t * denom  , '/', denom)

'''
Outcome:
   denom =  dx*uy*vz - dx*uz*vy - dy*ux*vz + dy*uz*vx + dz*ux*vy - dz*uy*vx
   a = (dx*oy*vz - dx*oz*vy + dx*vy*z0 - dx*vz*y0 - dy*ox*vz + dy*oz*vx - dy*vx*z0 + dy*vz*x0 + dz*ox*vy - dz*oy*vx + dz*vx*y0 - dz*vy*x0) / denom
   b = (-dx*oy*uz + dx*oz*uy - dx*uy*z0 + dx*uz*y0 + dy*ox*uz - dy*oz*ux + dy*ux*z0 - dy*uz*x0 - dz*ox*uy + dz*oy*ux - dz*ux*y0 + dz*uy*x0) / denom
   t = (-ox*uy*vz + ox*uz*vy + oy*ux*vz - oy*uz*vx - oz*ux*vy + oz*uy*vx + ux*vy*z0 - ux*vz*y0 - uy*vx*z0 + uy*vz*x0 + uz*vx*y0 - uz*vy*x0) / denom
'''

# https://github.com/sohale/implisolid/blob/9c8b93005cbc5d82e171d9a1090e52b06a6c7290/js_iteration_1/symbolic_gradients.py
exit(1)
