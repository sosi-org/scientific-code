// no "pragma once"

#include <cassert>

#include "./cpolyg_intersection.hpp"
#include "./test_helpers.hpp"

void test1_cpoly_intersection__two_points_parametrised(double x0, double y0, double x1, double y1, double xm, double ym, double x2, double y2)
{

    fixedsize_polygon_with_side_metadata_t poly1 = testhelper_polyg(simple_hacky_polygp_t{
        {x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}});

    fixedsize_polygon_with_side_metadata_t poly2 = testhelper_polyg(simple_hacky_polygp_t{
        {xm, ym}, {x2, ym}, {x2, y2}, {xm, y2}});

    collision_of_polyg cr = cpoly_intersection__two_points<double>(poly1, poly2);
    std::cout << "Collision result's two points: ";
    cr.debug_print();
    std::cout << std::endl;

    double a1 = convex_polygon_area<double>(poly1);
    double a2 = convex_polygon_area<double>(poly2);
    // double a1= convex_polygon_area(poly1);
    std::cout << "areas:" << a1 << "," << a2 << std::endl;
}

// checks if those two points are reported correctly. (not sure)
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

// todo: move to that module
// only corners
void debug_print_xy0(const fixedsize_polygon_with_side_metadata_t &poly)
{
    std::cout << poly.size() << ":";
    for (auto v : poly)
    {
        std::cout << "(" << v.x0 << "," << v.y0 << ") ";
    }
    // todo: how to avoid such repeat?
}

// todo: nested inside eachother

void test1_cpoly_intersection__produced()
{

    // good overlap
    double x0 = 0, y0 = 0;
    double x1 = 1, y1 = 1;
    double xm = 0.5, ym = 0.5;
    double x2 = 2, y2 = 2;

    fixedsize_polygon_with_side_metadata_t poly1 = testhelper_polyg(simple_hacky_polygp_t{
        {x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}});

    fixedsize_polygon_with_side_metadata_t poly2 = testhelper_polyg(simple_hacky_polygp_t{
        {xm, ym}, {x2, ym}, {x2, y2}, {xm, y2}});

    std::cout << "Pre-collision poly:";
    // debug_print(std::cout, result_poly);
    debug_print_xy0(poly1);
    std::cout << std::endl;

    simple_hacky_polygp_t result_poly =
        cpoly_intersection__complete_poly<double>(poly1, poly2, erasing_mod_t::no_erase);

    std::cout << "Full collision poly (no erase):";
    // debug_print(std::cout, result_poly);
    debug_print(result_poly);
    std::cout << std::endl;

    /* failed: not complete:
            Full collision poly:5:(0,0) (1,0.5) (1,0) (1,1) (0,1)
    */

    simple_hacky_polygp_t result_poly2 =
        cpoly_intersection__complete_poly<double>(poly1, poly2, erasing_mod_t::erase_between);

    std::cout << "Full collision poly (erase A):";
    // debug_print(std::cout, result_poly);
    debug_print(result_poly2);
    std::cout << std::endl;

    simple_hacky_polygp_t result_poly3 =
        cpoly_intersection__complete_poly<double>(poly1, poly2, erasing_mod_t::erase_outside);

    std::cout << "Full collision poly  (erase B):";
    // debug_print(std::cout, result_poly);
    debug_print(result_poly3);
    std::cout << std::endl;

    simple_hacky_polygp_t result_poly4 =
        cpoly_intersection__complete_poly<double>(poly1, poly2, erasing_mod_t::erase_auto);

    std::cout << "Full collision poly  (erase auto):";
    debug_print(result_poly4);
    std::cout << std::endl;

    // fixedsize_polygon_with_side_metadata_t p1 = testhelper_polyg(result_poly4);
    full_tesselation ft = testutil_tessellation_from_single_polygon(result_poly4);
    std::vector<point_t> helper_points{};
    save_svg_file(
        "./intersection1.svg",
        ft.trigulation,
        ft.points,
        helper_points,
        std::vector<side_meta_data_t>{});

    svg_saver{}
        .set_tessellation(ft)
        .set_helper_points(helper_points)
        .add_helper_point(point_t{0, 0})
        .add_helper_line(side_meta_data_t{point_t{0, 0}, point_t{2, 0}})
        .add_helper_line(side_meta_data_t{point_t{0, 0}, point_t{0, 2}})
        .write("./intersection1_.svg");
}
/*
map:

simple polygon -> fixedsize_polygon_with_side_metadata_t
// simple polygon = simple_polygi_t

map:


cpoly_intersection(fixedsize_polygon_with_side_metadata_t, fixedsize_polygon_with_side_metadata_t);
cpoly_intersection(simple_poly, simple_poly);

// how to relate ot tesselation?
// iterator for tesselation?? with ++ operator

*/

void test1_insideness()
{
    // good overlap
    double x0 = 0, y0 = 0;
    double x1 = 1, y1 = 1;

    fixedsize_polygon_with_side_metadata_t poly = testhelper_polyg(simple_hacky_polygp_t{
        {x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}});

    // point_t point{.x= 0.5, .y=0.5};
    // pt2_t point{.first= 0.5, .second=0.5};
    pt2_t point{0.5, 0.5};
    bool ins1 = is_inside_poly(poly, point);
    assert_equality_i(ins1, true);

    bool ins2 = is_inside_poly(poly, pt2_t{1.5, 0.5});
    assert_equality_i(ins2, false);
}
