# Please ignore this file.
# It is a bunch of frgments of code for some reason.

model_lamb = sympy.lambdify((*dyn_vars, *model_params, *model_inputs), dyn_derivs, dummify=False)
    #model_lamb = sympy.lambdify((*dyn_vars, *model_params, *model_inputs), dyn_derivs, dummify=False)
    model_lamb = sympy.lambdify((*dyn_vars,t), dyn_vars2, dummify=False)


a2=np.zeros((2,100))
y = model_lamb(a2[0,:],a2[1,:],1,1,15,0)
len(y) # 2. tuple
y[0].shape
y[1].shape


#nice
'''
import code
code.interact(local=locals())
exit()
'''

#import code
#code.interact(local=locals())
#exit()

#fitzhugh_nagumo(np.zeros((2,SIMU_STEPS1)), time_span, 1,1,15,1)
#fitzhugh_nagumo(np.zeros((2,SIMU_STEPS1)), time_span, **({"a":-.3, "b":1.4, "tau":20, "I":0}))
#

# prtial + lambdify
# https://stackoverflow.com/questions/66924592/lambdify-partially-linear-function-in-sympy

# fixed:
model_lamb = sympy.lambdify((*dyn_vars, *model_params, *model_inputs), dyn_derivs, dummify=False)
# --->
model_lamb = sympy.lambdify((*dyn_vars,t), dyn_vars2, dummify=False)

# linspace(start, stop, num=50)

# plt.plot(d_v, d_w, 'k.-') #.-

          #np.format_float_positional(ranges[0][:], precision=3),
          #np.format_float_positional(ranges[1][:], precision=3),

#sympy.core.numbers.Zero

#sympy `solve` notes:
# todo: why is v defined here?
def symbolic_nullclines(_model, param):
    # symbolic (algebraic) null-clines:
    print()
    # see https://github.com/sosi-org/scientific-code/blob/633224dc72692c13acc8263e21c933119eeaaf69/beeseyes/pycode/derive_formulas.py
    dv = _model[2][0].subs(param)
    dw = _model[2][1].subs(param)
    w = _model[0][1]
    v = _model[0][0]
    #v = sympy.Symbol(v, real=True)
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
        # _.doit()      # doit()
        # simplify(_)   # simplify()
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
        c = sympy.symbols("c", real=True)
        nlc1 = dv  # dv=0
        #nlc1 = dw # dw=0
        #lin1 = v-c # single root solution
        lin1 = w-c  # multiple roots
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
        #vars_list, _solution_set = solutions = solve(eq3, (w,),force=True, set=True)
        # ([w], {(-v**3 + v,)})
        vars_list, _solution_set = solutions = solve(eq3, (v,w),force=True, set=True)
        # convert set to list
        print((vars_list))  # (v,w)
        print((_solution_set))
        print()
        print(list(_solution_set))  # len = number of roots
        solution_sympy_list = list(_solution_set)
        # each solution is a tuple, for (v,w)
        print(solution_sympy_list[0]) # has two elements
        print(len(solution_sympy_list[0]))
        print()
        #solution_lambda_list = [
        #    sympy.lambdify((c,), solt, dummify=False)
        #        for solt in solution_sympy_list ]
        # c is the free parameter for the "curve". Since it's aa null-cline, we need (exactly) one free ariable for it
        # todo: rename c to s
        solution_lambda_list = [
            sympy.lambdify((c,), solt_vw, dummify=False)
                for solt_vw in solution_sympy_list ]
        # each is a function that returns ndim

        return solution_sympy_list, solution_lambda_list


# without  `set=True`: returns list of solutions?
#[(-v**3 + v,)]

# solve: set=True:
# simplify=True
# "check=True (default) will run through the symbol tags to elimate unwanted solutions. If no assumptions are included, all possible solutions will be returned"
# why didi I choose this? "manual=True (default is False). Do not use the polys/matrix method to solve a system of equations, solve them one at a time as you might “manually.”"


# index-larifying for ranges, etc.
# todo: from, to 👉, 👈
# ✝︎ ⌶ ▶︎ ▶️⤵️ ℹ️ ❕ 📏 📎 💈 🎗 ☝️

📎M_VARS
M📎_VARS


#{
#  (-(27*c/2 + sqrt(729*c**2 - 108)/2)**(1/3)/3 - 1/(27*c/2 + sqrt(729*c**2 - 108)/2)**(1/3), c),
#  (-(-1/2 + sqrt(3)*I/2)*(27*c/2 + sqrt(729*c**2 - 108)/2)**(1/3)/3 - 1/((-1/2 + sqrt(3)*I/2)*(27*c/2 + sqrt(729*c**2 - 108)/2)**(1/3)), c),
#  (-(-1/2 - sqrt(3)*I/2)*(27*c/2 + sqrt(729*c**2 - 108)/2)**(1/3)/3 - 1/((-1/2 - sqrt(3)*I/2)*(27*c/2 + sqrt(729*c**2 - 108)/2)**(1/3)), c)
#}

print('======')
for solt_vw in solution_sympy_list:
print()
sympy.pprint(solt_vw)
print(vars_list)
print('=====')
#exit()

# solve by this
sympy.lambdify((c,), solt_vw, dummify=False, modules=["scipy", "numpy"])  # modules=["math", "mpmath", "sympy"])
