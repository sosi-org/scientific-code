
/*
  solveAxb(): solves:
    [a,b]   [x1]   [x]
    [c,d] * [y1] = [y]

*/
template <typename real>
inline side_side_intersection_solution_t<real> solveAxb(real a, real b, real c, real d, real x, real y)
{
    real det = (a * d - b * c);
    // real [d,-b, -c, a] / det;
    //[d,-b, -c, a] * [x,y].T / det
    //[d,-b,  -c, a] * [x,y].T / det
    //[d *x  + -b*y,  -c*x+ a*y] / det
    real x1 = (+d * x - b * y) / det;
    real y1 = (-c * x + a * y) / det;

    return side_side_intersection_solution_t<real>{true, x1, y1};
}

/*
template <typename real>
inline inv2x2( real a,  real b,  real c, real d) {
    real det = (a*d-b*c);
    real (d,-b, -c, a) / det;
    (d,-b, -c, a) / det * B
}
*/

// todo: move to side_meta_data_t.hpp
template <typename real>
// bool
void intersect_lines_deprecated(const side_meta_data_t &side1, const side_meta_data_t &side2)
{
    /*
    double x0, y0;
    double dx, dy;
    double x = ...;

    // a1 * x + b1 * y = c1
    // a2 * x + b2 * y = c2
    [a1, b1]
    [a2, b2]
    [c1,c2].T
    det = (a1);
    */

    real dx1 = side1.dx;
    real dy1 = side1.dy;
    real dx2 = side2.dx;
    real dy2 = side2.dy;
    real x10 = side1.x0;
    real y10 = side1.y0;
    real x20 = side2.x0;
    real y20 = side2.y0;
    /*
    line equation: (1/dy|-1/dx) . ( (x|y) - (x0|y0) ) == 0
    a x + b y == c
         a = 1/dy
         b = -1/dx
         c = a*x0 + b*y0
    */
    // double a1 = 1 / dy1, b1 = -1 / dx1;
    // double a2 = 1 / dy2, b2 = -1 / dx2;
    double a1 = dx1, b1 = dy1;
    double a2 = dx2, b2 = dy2;
    real c1 = a1 * x10 + b1 * y10;
    real c2 = a2 * x20 + b2 * y20;
    // returntype<real> funcname<real>  --> why none is needed?
    side_side_intersection_solution_t ip =
        solveAxb(
            a1, b1,
            a2, b2,

            -c1, -c2);

    std::cout << "intersection: " << ip.x << "," << ip.y << std::endl;
}
