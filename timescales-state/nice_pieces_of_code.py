

model_lamb = sympy.lambdify((*dyn_vars, *model_params, *model_inputs), dyn_derivs, dummify=False)
    #model_lamb = sympy.lambdify((*dyn_vars, *model_params, *model_inputs), dyn_derivs, dummify=False)
    model_lamb = sympy.lambdify((*dyn_vars,t), dyn_vars2, dummify=False)


a2=np.zeros((2,100))
y = model_lamb(a2[0,:],a2[1,:],1,1,15,0)
len(y) # 2. tuple
y[0].shape
y[1].shape


#nice

import code
code.interact(local=locals())
exit()

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
