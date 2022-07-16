// no "pragma once"

#include <cassert>

#include "./cpolyg_intersection.hpp"


// rename: make_simple_hacky_polygp
//  template <typename real>
// simple_polygi_t
simple_hacky_polygp_t testutil_simply_polygon(simple_hacky_polygp_t coords)
{
    simple_hacky_polygp_t pp;
    for (const auto &c : coords)
    {
        pt2_t pt{c.first, c.second};
        pp.push_back(pt);
    }
    return pp;
}

full_tesselation testutil_tessellation_from_single_polygon(simple_hacky_polygp_t coords)
{
    full_tesselation pnp;
    simple_polygi_t pgi;
    for (side_point_t i = 0; const auto &c : coords)
    {
        // pair<> is used only for simpler constructor-literal
        point_t pnt{c.first, c.second};
        pnp.points.push_back(pnt);
        // i = pnp.points.size()-1;
        pgi.push_back(i);
        ++i;
    }
    pnp.trigulation.push_back(pgi);
    return pnp;
}

/*
test{} (
    (0,0), (1,0), (0,1), (1,1)
    ->
    1.0
);
*/

// for a single poylgon, not the full tessellation:
fixedsize_side_metadata_t t2patch(const simple_polygi_t &polyg, const points_t &points);

void test1_convex_polygon_area()
{
    full_tesselation square = testutil_tessellation_from_single_polygon(simple_hacky_polygp_t{{0, 0}, {1, 0}, {1, 1}, {0, 1}});
    // convert to side_meta_data_t
    // using constructor side_meta_data_t{p1, p2}
    fixedsize_side_metadata_t poly1 = t2patch(square.trigulation[0], square.points);
    double a = convex_polygon_area<double>(poly1);
    std::cout << a << " expected.to.be 1" << std::endl;
}
/*
fixedsize_side_metadata_t  t2patch(const simple_polygi_t& polyg)
*/

void test2_convex_polygon_area()
{

    simple_hacky_polygp_t square = testutil_simply_polygon(simple_hacky_polygp_t{{0, 0}, {1, 0}, {1, 1}, {0, 1}});
    double a = convex_polygon_area2<double, true>(square);

    std::cout << a << " expected.to.be 1" << std::endl;
}

void dummy1()
{
    /*  works but not useful: */
    simple_hacky_polygp_t square = testutil_simply_polygon(simple_hacky_polygp_t{{0, 0}, {1, 0}, {1, 1}, {0, 1}});
}

fixedsize_side_metadata_t testhelper_polyg(const simple_hacky_polygp_t &shp)
{
    // based on: test1_convex_polygon_area
    full_tesselation square1 = testutil_tessellation_from_single_polygon(shp);
    fixedsize_side_metadata_t poly = t2patch(square1.trigulation[0], square1.points);
    return poly;
}

void test1_cpoly_intersection__two_points_parametrised(double x0, double y0, double x1, double y1, double xm, double ym, double x2, double y2)
{

    fixedsize_side_metadata_t poly1 = testhelper_polyg(simple_hacky_polygp_t{
        {x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}});

    fixedsize_side_metadata_t poly2 = testhelper_polyg(simple_hacky_polygp_t{
        {xm, ym}, {x2, ym}, {x2, y2}, {xm, y2}});

    collision_of_polyg cr = cpoly_intersection__two_points<double>(poly1, poly2);
    cr.debug_print();
    std::cout << std::endl;

    double a1 = convex_polygon_area<double>(poly1);
    double a2 = convex_polygon_area<double>(poly2);
    // double a1= convex_polygon_area(poly1);
    std::cout << "areas:" << a1 << "," << a2 << std::endl;
}

void test1_cpoly_intersection__two_points()
{
    // good overlap
    double x0 = 0, y0 = 0;
    double x1 = 1, y1 = 1;
    double xm = 0.5, ym = 0.5;
    double x2 = 2, y2 = 2;

    test1_cpoly_intersection__two_points_parametrised(x0, y0, x1, y1, xm, ym, x2, y2);
}

void test2_cpoly_intersection__two_points()
{
    // no overlap
    double x0 = 0, y0 = 0;
    double x1 = 0.5, y1 = 0.5;
    double xm = 1, ym = 1;
    double x2 = 2, y2 = 2;

    test1_cpoly_intersection__two_points_parametrised(x0, y0, x1, y1, xm, ym, x2, y2);
}

/*
map:

simple polygon -> fixedsize_side_metadata_t
// simple polygon = simple_polygi_t

map:


cpoly_intersection(fixedsize_side_metadata_t, fixedsize_side_metadata_t);
cpoly_intersection(simple_poly, simple_poly);

// how to relate ot tesselation?
// iterator for tesselation?? with ++ operator

*/
