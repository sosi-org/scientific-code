
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

import sympy
# from sympy.utilities.codegen import codegen
import sympy.utilities.codegen


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

# def generate_raycaster(numerators, gcd_divisor, output_names, input_symbols):
#     creates analytical_....
#     which returns that expression
#     and wraps it? but it is perhaps not necessary (manual? but all code generaiton needs ot be changeable (kind of revwresible hence, no manual modificaiton))


def generate_efficient_code_laumbdify_numpy(numerators, gcd_divisor, output_names, input_symbols):
    # not tested
    import sympy as sp
    import numpy as np

    # todo: swap order of (input_symbolsm output_names)

    print('input_symbols', input_symbols)
    print('output_names', output_names)
    print('numerators', numerators)
    print('gcd_divisor', gcd_divisor)

    # x = sp.symbols('x')
    # expr = sp.sin(x)**2 + sp.cos(x)**2
    # Will use np.sin and np.cos
    exprs = numerators
    exprs = [n/gcd_divisor for n in numerators]
    f = sp.lambdify(input_symbols, exprs, "numpy")
    # todo: see scientific_code/fitzhue-nagamo for some nuances of `.lambdify`

    # Do a little test
    # print(f(np.array([0, 1, 2])))
    # raise Exception('tbc')
    return f

# numerator/ (denomerator=divisor)


def generate_efficient_code_c_for_ufunc(numerators, gcd_divisor, output_names):
    """ `_c` to be used by generate_efficient_code_ufunc. """
    print(numerators)
    print(output_names)
    # print(output_symbols)

    routines = []
    all_func_names = []
    routines.append(('common_divisor', gcd_divisor))
    for i in range(len(numerators)):
        print(str(output_names[i]))
        expr = numerators[i] / gcd_divisor
        funcname = 'my_cast_func_' + str(output_names[i])
        routines.append((funcname, expr))
        all_func_names.append(funcname)
        # my_cast_func
    print(all_func_names)

    [(c_name, c_code), (h_name, c_header)] = sympy.utilities.codegen.codegen(
        routines,
        # ("my_cast_func", numerators),
        language="C", header=False, empty=False)
    print(h_name)
    print(c_header)
    print(c_name)
    print(c_code)
    struct_type = f"""
    struct output_type {{ { ";".join(f'float {name}' for name in all_func_names)} }};
    """
    print(struct_type)
    struct_version = f"""
      output_type joint() {{
          `for i {{`
                ..
          `}}`
      }}
    """
    print(struct_version)
    exit()


def generate_efficient_code_ufunc(numerators, gcd_divisor, output_names):
    # generate C code
    # a folder with ready-to compile code (using C-compiler)
    # generated/flat_cam, generated/spherical_cam, generated/cylindrical_cam, cylinder_line (versus cylinder_point), cone_cam, implicit_cam (!), sdf_cam, GPU verions, etc.
    generate_efficient_code_c_for_ufunc(numerators, gcd_divisor, output_names)

    raise Exception()
    return None


generate_efficient_code = generate_efficient_code_ufunc
generate_efficient_code = generate_efficient_code_laumbdify_numpy
