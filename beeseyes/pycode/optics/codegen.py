
# sympy_codegen.py
# a more specific name is needed


# 1. Generate C code
# 2. Generate python code suitable for numpy
# 3. Why not simply Lambify?
#
# About 2:
#      but 2 is preferred (actual code too: hence, not opaque)
#      It's a kind of explainability (but also short)
#      This is important: multi-level, but non-black-box
#
# 4. Use numpy-able C code.
#       Using ufunc
#       See `nl_cast_numpy` folder
# In general, also an issue is to compute-factorise (using substitue, similar to autodiff)

def generate_efficient_code_pure_c(denoms, gcd_divisor):
    """
    But this is C code.
    todo: Generate C code suitable as numpy-compilable!
    """
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

    [(c_name, c_code), (h_name, c_header)] = codegen(
        eqlist,
        "C", header=False, empty=False)

    # how about python code?
    print("Filename: ", c_name)
    c_code = c_code.replace("pow", "std::pow")
    c_code = c_code.replace("sqrt", "std::sqrt")
    c_code = c_code.replace("double", "REAL")
    print(c_code)




def generate_efficient_code_laumbdify_numpy(denoms, gcd_divisor):
    # not tested
    import sympy as sp
    import numpy as np

    x = sp.symbols('x')
    expr = sp.sin(x)**2 + sp.cos(x)**2
    f = sp.lambdify(x, expr, "numpy")  # Will use np.sin and np.cos
    # todo: see scientific_code/fitzhue-nagamo for some nuances of `.lambdify`

    # Do a little test
    print(f(np.array([0, 1, 2])))

    return f

def generate_efficient_code_c_for_ufunc(denoms, gcd_divisor):
   """ `_c` to be used by generate_efficient_code_ufunc. """
   print(denoms)

def generate_efficient_code_ufunc(denoms, gcd_divisor):
  # generate C code
  # a folder with ready-to compile code (using C-compiler)
  # generated/flat_cam, generated/spherical_cam, generated/cylindrical_cam, cylinder_line (versus cylinder_point), cone_cam, implicit_cam (!), sdf_cam, GPU verions, etc.
  generate_efficient_code_c_for_ufunc(denoms, gcd_divisor)

  raise Exception()
  return None


generate_efficient_code = generate_efficient_code_ufunc
