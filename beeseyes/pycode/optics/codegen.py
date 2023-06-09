
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

def generate_efficient_code(denoms, gcd_divisor):
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

