
"""
source ../../../../timescales-state/p3-for-me/bin/activate
"""

import sympy

# This program generates automatic c-code for gradient of
#implicit objects with complicated formulas. It generates the gradient automatically.

# forked from beeseyes/pycode/derive_formulas.py

#from __future__ import division
import sympy
from sympy import *
from sympy import Eq as Equation

sympy.init_printing()

# unknown
x, y = symbols('x y', real=true)
X = Matrix([x,y])


# Two lines
x1, y1 = symbols('x1 y1', real=true)
x2, y2 = symbols('x2 y2', real=true)

x3, y3 = symbols('x3 y3', real=true)
x4, y4 = symbols('x4 y4', real=true)


X1 = Matrix([x1,y1])
X2 = Matrix([x2,y2])
X3 = Matrix([x3,y3])
X4 = Matrix([x4,y4])

# line parameters
# Ray drive
t12, t34 = symbols('u v', real=true)

# a, b = symbols('a b', real=true)

ray12 = t12 * X1 + (1-t12) * X2
ray34 = t34 * X3 + (1-t34) * X4

# t in terms of x
def solve_param(ray12_X,t12):
    print('========')
    eq = Equation(ray12_X, ray12_X*0)
    sympy.pprint(eq)

    solutions = solve(eq, (t12), force=True, manual=True,
        set=False,
        check=False,
        numerical=False,
        warn=True,
        )

    print(solutions)
    sympy.pprint(solutions)
    assert len(solutions) == 1
    assert len(solutions[0]) == 1
    sol1 = solutions[0][0]
    sympy.pprint(sol1)
    print('========')
    return sol1

sol_u = solve_param(ray12-X,t12)
print(sol_u)
sympy.pprint(sol_u)

# t in terms of x (i.e. eliminating t)
ray12_x = ray12.subs(t12, sol_u)
sympy.pprint(ray12_x)


sol_v = solve_param(ray34 - X,t34)
print(sol_v)
sympy.pprint(sol_v)
# t' in terms of x (i.e. eliminating t)
ray34_x = ray34.subs(t34, sol_v)
sympy.pprint(ray34_x)

print('\n\n')
sympy.pprint(ray12_x-X)
sympy.pprint(ray34_x-X)

# no u,v,t: all in terms of x,y (why no y??)
eq_all = Equation(Matrix([ray12_x-X, ray34_x-X]), Matrix([0,0]))
sympy.pprint(eq_all)
exit()

eq12 = ray12 - X
eq34 = ray34 - X

sympy.pprint(eq12)
sympy.pprint(eq34)




O0 =  Matrix([0,0])
#eq = Equation(Matrix([eq12, eq34]), Matrix([O0,O0]))

eq = Equation(Matrix([ray12, ray34]), Matrix([X,X]))

#eq = Equation(Matrix([t12-3]), Matrix([t12*0]))
print('eq:')
sympy.pprint(eq)

eq= simplify(eq)
print('eq simplified:')
sympy.pprint(eq)

print('expand:')
sympy.pprint(expand(Matrix([ray12, ray34])))
#print('factor:')
#sympy.pprint(factor(Matrix([ray12, ray34])))


#solution = solve(eq, (t12,t34,X), force=True, manual=True)
#solution = solve(eq, (X,t12,t34), force=True, manual=True)
#solution = solve(eq, (X,t12,t34), force=True, manual=True,
solution = solve(eq, (X,t12,t34), force=True, manual=True,
    set=False,
    check=False,
    numerical=False,
    warn=True,
    )

# cannot be correct:
# {u: (x - x2)/(x1 - x2), v: (x - x4)/(x3 - x4)}

print('solution:', type(solution),'\n',solution)
sol_u = solution[t12]
sol_v = solution[t34]
print(sol_u)
print(sol_v)


eq2 = eq.subs(t12, sol_u).subs(t34, sol_v)
sympy.pprint(eq2)

solution2 = solve(eq2, (x,), force=True, manual=True,
    set=False,
    check=False,
    numerical=False,
    warn=True,
    )
print('solution2:', type(solution2),'\n',solution2)

exit()

# Plane: side vectors
U = Matrix([ux,uy,uz])
V = Matrix([vx,vy,vz])
#W = Matrix([wx,wy,wz])
UV = Matrix([U.T, V.T]).T

# Local cordinates:
ab = Matrix([a,b])

plane = UV*ab + C0

sympy.pprint(plane)

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


eq0 = plane - ray
sympy.pprint(eq0)
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

x, y, z = symbols('x y z', real=true)
ax, ay, az = symbols('ax ay az', real=true)

delta = Symbol('delta', real=true)
twist_rate = Symbol('twist_rate', real=true)
phi0 = Symbol('phi0', real=true)
r0 = Symbol('r0', real=true)
pi2 = sympy.pi * 2
inside_ness = 1


X = Matrix([x,y,z])
A = Matrix([ax,ay,az])
u = Matrix([ux,uy,uz])
w = Matrix([wx,wy,wz])

v = u.cross(w)
#sympy.pprint(v)
t = (X - A).dot(w)
#sympy.pprint(t)
P = t * w + A
#planar = X - P
UVW = Matrix([u.T,v.T,w.T])
#sympy.pprint(UVW)
#UVW_inv = UVW.inv()  # Slow, but works
#sympy.pprint(UVW_inv)  #huge
q = []
for i in range(3):
    q.append([])
    for j in range(3):
        q[i].append(sympy.Symbol( 'uvwi'+str(i)+str(j) ))
        q[i][j]
#UVW_inv = Matrix([[q00,q01,q02],[q00,q01,q02],[q20,q21,q22]])

UVW_inv = Matrix(3,3, lambda i,j: q[i][j])
#sympy.pprint(UVW_inv)
ab = UVW_inv.dot(X - P)
#sympy.pprint(ab)

theta = atan2(ab[1], ab[0])
sympy.pprint(theta)

r = (X-P).norm()
#sympy.pprint(r)


angle = t / twist_rate - theta/pi2 + phi0
#sssss = phi(qqqqqqq)  # did not work
pattern = sympy.sin(angle)
screw_ness = (-r + r0 + delta * pattern) * inside_ness
#The screw's implicit funciton

#print "J"
gx = diff(screw_ness, x)
gy = diff(screw_ness, y)
gz = diff(screw_ness, z)
sympy.pprint(gx)

from sympy.utilities.codegen import codegen
from sympy.abc import f, g, h
from sympy import Eq
eqlist = [
    ("implicit", screw_ness),
    ("gradient",
        [
            Eq(f, gx),
            Eq(g, gy),
            Eq(h, gz)
        ]
    )
]
[(c_name, c_code), (h_name, c_header)] = codegen( \
      eqlist, \
      "C", header=False, empty=False)
print ("Filename: ", c_name)
c_code = c_code.replace("pow", "std::pow")
c_code = c_code.replace("sqrt", "std::sqrt")
c_code = c_code.replace("double", "REAL")
print(c_code)


# dump_c = ?

# If you want to use this in a Cythong / numpy, see autowrap module
# http://docs.sympy.org/latest/modules/utilities/autowrap.html

# See Theano for generating a more efficient code.

"""
Result:


implicit.c
#include "implicit.h"
#include <math.h>
double implicit(double ax, double ay, double az, double delta, double phi0, double r0, double twist_rate, double uvwi00, double uvwi01, double uvwi02, double uvwi10, double uvwi11, double uvwi12, double wx, double wy, double wz, double x, double y, double z) {
   double implicit_result;
   implicit_result = delta*sin(phi0 - 1.0L/2.0L*atan2(uvwi10*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x) + uvwi11*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y) + uvwi12*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z), uvwi00*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x) + uvwi01*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y) + uvwi02*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z))/M_PI + (wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z))/twist_rate) + r0 - sqrt(pow(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x, 2) + pow(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y, 2) + pow(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z, 2));
   return implicit_result;
}
void gradient(double ax, double ay, double az, double delta, double phi0, double twist_rate, double uvwi00, double uvwi01, double uvwi02, double uvwi10, double uvwi11, double uvwi12, double wx, double wy, double wz, double x, double y, double z, double *f, double *g, double *h) {
   (*f) = delta*(-1.0L/2.0L*((uvwi00*(-pow(wx, 2) + 1) - uvwi01*wx*wy - uvwi02*wx*wz)*(-uvwi10*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x) - uvwi11*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y) - uvwi12*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z))/(pow(uvwi00*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x) + uvwi01*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y) + uvwi02*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z), 2) + pow(uvwi10*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x) + uvwi11*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y) + uvwi12*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z), 2)) + (uvwi00*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x) + uvwi01*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y) + uvwi02*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z))*(uvwi10*(-pow(wx, 2) + 1) - uvwi11*wx*wy - uvwi12*wx*wz)/(pow(uvwi00*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x) + uvwi01*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y) + uvwi02*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z), 2) + pow(uvwi10*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x) + uvwi11*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y) + uvwi12*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z), 2)))/M_PI + wx/twist_rate)*cos(phi0 - 1.0L/2.0L*atan2(uvwi10*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x) + uvwi11*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y) + uvwi12*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z), uvwi00*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x) + uvwi01*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y) + uvwi02*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z))/M_PI + (wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z))/twist_rate) - (-wx*wy*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y) - wx*wz*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z) + (1.0L/2.0L)*(-2*pow(wx, 2) + 2)*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x))/sqrt(pow(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x, 2) + pow(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y, 2) + pow(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z, 2));
   (*g) = delta*(-1.0L/2.0L*((uvwi00*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x) + uvwi01*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y) + uvwi02*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z))*(-uvwi10*wx*wy + uvwi11*(-pow(wy, 2) + 1) - uvwi12*wy*wz)/(pow(uvwi00*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x) + uvwi01*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y) + uvwi02*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z), 2) + pow(uvwi10*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x) + uvwi11*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y) + uvwi12*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z), 2)) + (-uvwi10*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x) - uvwi11*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y) - uvwi12*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z))*(-uvwi00*wx*wy + uvwi01*(-pow(wy, 2) + 1) - uvwi02*wy*wz)/(pow(uvwi00*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x) + uvwi01*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y) + uvwi02*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z), 2) + pow(uvwi10*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x) + uvwi11*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y) + uvwi12*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z), 2)))/M_PI + wy/twist_rate)*cos(phi0 - 1.0L/2.0L*atan2(uvwi10*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x) + uvwi11*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y) + uvwi12*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z), uvwi00*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x) + uvwi01*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y) + uvwi02*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z))/M_PI + (wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z))/twist_rate) - (-wx*wy*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x) - wy*wz*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z) + (1.0L/2.0L)*(-2*pow(wy, 2) + 2)*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y))/sqrt(pow(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x, 2) + pow(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y, 2) + pow(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z, 2));
   (*h) = delta*(-1.0L/2.0L*((uvwi00*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x) + uvwi01*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y) + uvwi02*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z))*(-uvwi10*wx*wz - uvwi11*wy*wz + uvwi12*(-pow(wz, 2) + 1))/(pow(uvwi00*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x) + uvwi01*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y) + uvwi02*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z), 2) + pow(uvwi10*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x) + uvwi11*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y) + uvwi12*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z), 2)) + (-uvwi10*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x) - uvwi11*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y) - uvwi12*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z))*(-uvwi00*wx*wz - uvwi01*wy*wz + uvwi02*(-pow(wz, 2) + 1))/(pow(uvwi00*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x) + uvwi01*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y) + uvwi02*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z), 2) + pow(uvwi10*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x) + uvwi11*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y) + uvwi12*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z), 2)))/M_PI + wz/twist_rate)*cos(phi0 - 1.0L/2.0L*atan2(uvwi10*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x) + uvwi11*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y) + uvwi12*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z), uvwi00*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x) + uvwi01*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y) + uvwi02*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z))/M_PI + (wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z))/twist_rate) - (-wx*wz*(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x) - wy*wz*(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y) + (1.0L/2.0L)*(-2*pow(wz, 2) + 2)*(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z))/sqrt(pow(-ax - wx*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + x, 2) + pow(-ay - wy*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + y, 2) + pow(-az - wz*(wx*(-ax + x) + wy*(-ay + y) + wz*(-az + z)) + z, 2));
}

"""
