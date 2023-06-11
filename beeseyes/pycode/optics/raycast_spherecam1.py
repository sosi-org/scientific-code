# a pluggable version

# 1. See beeseyes/pycode/optics/ray_cast_flatcamera.py
# Also see the oflow... version (better refactored)
# possible name: raycast_sphericalcamera.py

#  related to `spherecam1`, hence
#   * beeseyes/pycode/optics/analytical_cast_spherecam1.py

# The idea was to simply interpret

# incomplete, in progress
# but given lambdify, it is perhaps not needed

numerators = (
    -dx*dy*oy - dx*dz*oz - dx*sqrt(-dx**2*oy**2 - dx**2*oz**2 + dx**2 + 2*dx*dy*ox*oy + 2*dx*dz*ox*oz - dy**2*ox**2 - dy**2*oz**2 + dy**2 + 2*dy*dz*oy*oz - dz**2*ox**2 - dz**2*oy**2 + dz**2) + dy**2*ox + dz**2*ox,
    dx**2*oy - dx*dy*ox - dy*dz*oz - dy*sqrt(-dx**2*oy**2 - dx**2*oz**2 + dx**2 + 2*dx*dy*ox*oy + 2*dx*dz*ox*oz - dy**2*ox**2 - dy**2*oz**2 + dy**2 + 2*dy*dz*oy*oz - dz**2*ox**2 - dz**2*oy**2 + dz**2) + dz**2*oy,
    dx**2*oz - dx*dz*ox + dy**2*oz - dy*dz*oy - dz*sqrt(-dx**2*oy**2 - dx**2*oz**2 + dx**2 + 2*dx*dy*ox*oy + 2*dx*dz*ox*oz - dy**2*ox**2 - dy**2*oz**2 + dy**2 + 2*dy*dz*oy*oz - dz**2*ox**2 - dz**2*oy**2 + dz**2),
    -dx*ox - dy*oy - dz*oz - sqrt(-dx**2*oy**2 - dx**2*oz**2 + dx**2 + 2*dx*dy*ox*oy + 2*dx*dz*ox*oz - dy**2*ox**2 - dy**2*oz**2 + dy**2 + 2*dy*dz*oy*oz - dz**2*ox**2 - dz**2*oy**2 + dz**2)
    )

gcd_divisor = 1/(dx**2 + dy**2 + dz**2)

# (sol_xx, sol_yy, sol_zz, sol_t) = numerators
# multiply numerically:
(sol_xx, sol_yy, sol_zz, sol_t) = (nume * gcd_divisor for nume in numerators)
