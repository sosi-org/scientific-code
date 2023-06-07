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


a, b, c = symbols('a b c', real=true)

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
abc = Matrix([a,b,c])

point_in_sphere_uvw1 = UVW * abc + C0
# point_in_sphere_uvw2 = norm(UVW * abc) - 1 #  = 0
# pair them together?
# No, it's just one equation:
# implicit
ellipsoid_surface_eq = a*a+b*b+c*c - 1
ellipsoid_surface_expl = point_in_sphere_uvw1

# sympy.pprint(point_in_sphere_uvw1)
sympy.pprint(ellipsoid_surface_eq)
# But this is not what we need


variables = (a,b,c)
parameters = (U,V,W,C0)
eq = ellipsoid_surface_eq
# world_variables = (x,y,z) # no such thing

various_forms.append({
    'vars': variables, # local vars
    # 'x,y,z': world_variables, # local vars ---> 'expl-eq'
    'params': parameters,
    'impl-eq': eq,
    'expl-eq': ellipsoid_surface_expl,
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
    'impl-eq': polar_sphere_eq,
    'expl-eq': Matrix([x,y,z]),
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

# -------------------------
# Now cross ray with each:

for form_i, form in enumerate(various_forms):
    vars = form['vars']
    params = form['params']
    # Implicit equation for each: f(...) = 0
    # eq = form['eq']
    ieq = form['impl-eq']
    # Explicit equation for each: (x,y,z) := ...
    exeq = form['expl-eq']
    eq0 = exeq - ray
    print('equation == 0 :')
    sympy.pprint(eq0)
exit()
'''
⎡a⋅ux + b⋅vx - dx⋅t - ox⎤
⎢                       ⎥
⎢a⋅uy + b⋅vy - dy⋅t - oy⎥
⎢                       ⎥
⎣a⋅uz + b⋅vz - dz⋅t - oz⎦
'''


# Equation
from sympy import Eq as Equation
# Solver

zero0 =  Matrix([0,0,0])
#eq = Equation(plane - ray, 0) # Does not work. Obviously.
eq = Equation(plane - ray, zero0)

sympy.pprint(eq)
'''
⎡a⋅ux + b⋅vx - dx⋅t - ox⎤   ⎡0⎤
⎢                       ⎥   ⎢ ⎥
⎢a⋅uy + b⋅vy - dy⋅t - oy⎥ = ⎢0⎥
⎢                       ⎥   ⎢ ⎥
⎣a⋅uz + b⋅vz - dz⋅t - oz⎦   ⎣0⎦
'''
sympy.pprint(eq, use_unicode=True)


#qq = solve(eq, (a,b,t))
#qq = solve(eq, (t,a,b), set=True)

#solve_linear_system(eq, t)
#print(qq)

#print(real_root(eq))
#r = real_root(UV*ab - (O + t * D))
#print('rrr', r)
#sympy.pprint(r, use_unicode=True)

solutions = solve(eq, (a,b,t),force=True, manual=True, set=True)
#print(solutions)
#([a, b, t],
# {((dx*oy*vz - dx*oz*vy - dy*ox*vz + dy*oz*vx + dz*ox*vy - dz*oy*vx)/(dx*uy*vz - dx*uz*vy - dy*ux*vz + dy*uz*vx + dz*ux*vy - dz*uy*vx), (-dx*oy*uz + dx*oz*uy + dy*ox*uz - dy*oz*ux - dz*ox*uy + dz*oy*ux)/(dx*uy*vz - dx*uz*vy - dy*ux*vz + dy*uz*vx + dz*ux*vy - dz*uy*vx), (-ox*uy*vz + ox*uz*vy + oy*ux*vz - oy*uz*vx - oz*ux*vy + oz*uy*vx)/(dx*uy*vz - dx*uz*vy - dy*ux*vz + dy*uz*vx + dz*ux*vy - dz*uy*vx))}
# )

#print(len(solutions))  # 2 ?!
#print(solutions[0])  # [a, b, t]
#print(solutions[1]) # {(a,b,t)}
#(tupl, sol) = solutions

solution = solve(eq, (a,b,t),force=True, manual=True)

#print(solution) #[(,,)]
#print(solution[0])
# ((dx*oy*vz - dx*oz*vy - dy*ox*vz + dy*oz*vx + dz*ox*vy - dz*oy*vx)/(dx*uy*vz - dx*uz*vy - dy*ux*vz + dy*uz*vx + dz*ux*vy - dz*uy*vx), (-dx*oy*uz + dx*oz*uy + dy*ox*uz - dy*oz*ux - dz*ox*uy + dz*oy*ux)/(dx*uy*vz - dx*uz*vy - dy*ux*vz + dy*uz*vx + dz*ux*vy - dz*uy*vx), (-ox*uy*vz + ox*uz*vy + oy*ux*vz - oy*uz*vx - oz*ux*vy + oz*uy*vx)/(dx*uy*vz - dx*uz*vy - dy*ux*vz + dy*uz*vx + dz*ux*vy - dz*uy*vx))


sol_a = solution[0][0]
sol_b = solution[0][1]
sol_t = solution[0][2]

# print(sol_a)
# print(sol_b)
# print(sol_t)
# (dx*oy*vz - dx*oz*vy + dx*vy*z0 - dx*vz*y0 - dy*ox*vz + dy*oz*vx - dy*vx*z0 + dy*vz*x0 + dz*ox*vy - dz*oy*vx + dz*vx*y0 - dz*vy*x0)/(dx*uy*vz - dx*uz*vy - dy*ux*vz + dy*uz*vx + dz*ux*vy - dz*uy*vx)
# (-dx*oy*uz + dx*oz*uy - dx*uy*z0 + dx*uz*y0 + dy*ox*uz - dy*oz*ux + dy*ux*z0 - dy*uz*x0 - dz*ox*uy + dz*oy*ux - dz*ux*y0 + dz*uy*x0)/(dx*uy*vz - dx*uz*vy - dy*ux*vz + dy*uz*vx + dz*ux*vy - dz*uy*vx)
# (-ox*uy*vz + ox*uz*vy + oy*ux*vz - oy*uz*vx - oz*ux*vy + oz*uy*vx + ux*vy*z0 - ux*vz*y0 - uy*vx*z0 + uy*vz*x0 + uz*vx*y0 - uz*vy*x0)/(dx*uy*vz - dx*uz*vy - dy*ux*vz + dy*uz*vx + dz*ux*vy - dz*uy*vx)

d = gcd(sol_a, sol_b)  # Does not work with `sol_t`
print('gcd = ', 1/d)

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
