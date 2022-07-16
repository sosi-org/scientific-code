#pragma once

#include <cassert>

// for: intersect_lines_segment
#include "./line_intersection.hpp"

// one of the many representaitons of point !
/*
template<typename real>
typedef std::pair<real,real> pt2_t_;
// a typedef cannot be a template
*/
typedef std::pair<double, double> pt2_t;

// use only for temporary (test, debug, etc), before conversions
// template<typename real>
typedef std::vector<pt2_t> simple_hacky_polygp_t;

/*
template <typename real>
inline simple_hacky_polygp_t<real>
*/
/*
intersection between two convex polygons.

todo: add new augmented points, and add indices (?)
no: you just need the area
*/
template <typename real>
inline simple_hacky_polygp_t cpoly_intersection(const fixedsize_side_metadata_t &poly1, const fixedsize_side_metadata_t &poly2)
{
    return simple_hacky_polygp_t();
}

// result_thisfunc
struct collision_of_polyg
{
    // point1, side_id1 point2l side_id2
    // point_t point1, point2;
    // side_index_int_t side_1a, side_1b, side2a, side2b;
    point_t point[2];
    side_index_int_t side_1[2], side_2[2];
    // bool intersect;
    // no intersec
    // bool corner_on_corner;
    // bool corner_on_edge;
    /*
    enum class {no_intersect, two_intersect, corner_on_cronder, sorner_on_edge} result_situation;
    */
    // enum class {no_intersect, yes_intersection} situation;
    size_t count;

    void debug_print()
    {
        std::cout << count << ": ";
        for (int i = 0; i < count; ++i)
        {
            std::cout << side_1[i] << ":(" << point[i].x << "," << point[i].y << ")";
        }
        std::cout << count << ". ";
    }
};
// collision
template <typename real>
inline collision_of_polyg cpoly_intersection__two_points(const fixedsize_side_metadata_t &poly1, const fixedsize_side_metadata_t &poly2)
{

    // C++20
    collision_of_polyg result{.count = 0};

    size_t ctr = 0;

    for (side_index_int_t s1 = 0; s1 < poly1.size(); ++s1)
    {
        const side_meta_data_t &side1 = poly1[s1];

        for (side_index_int_t s2 = 0; s2 < poly2.size(); ++s2)
        {

            const side_meta_data_t &side2 = poly2[s2];

            side_side_intersection_solution_t<real> is =
                intersect_lines_segment<real>(side1, side2);

            if (is.intersect)
            {
                result.point[ctr] = point_t{is.x, is.y};
                result.side_1[ctr] = s1;
                result.side_2[ctr] = s2;
                result.count = ctr + 1;
                ctr++;
                if (ctr >= 2)
                {
                    return result;
                }
            }
        }
    }
    return result;
    // return simple_hacky_polygp_t();
}

// std::vector<point_t>
template <typename real>
inline collision_of_polyg cpoly_intersection__two_points(const fixedsize_side_metadata_t &poly1, const fixedsize_side_metadata_t &poly2);

template <typename real>
inline real convex_polygon_area(const fixedsize_side_metadata_t &poly1)
{
    real area = 0;
    for (const side_meta_data_t &side : poly1)
    {
        real xyp = side.x0 * side.y1;
        real xyn = side.y0 * side.x1;
        area += xyp - xyn;
    }
    return area / 2.0;
}

// area from simple_hacky_polygp_t

// verbose, debug
template <typename real, bool verbose>
inline real convex_polygon_area2(const simple_hacky_polygp_t &poly1)
{
    real area = 0;
    {
        // C++23:
        // std::cout << "area(2):  " if consteval (verbose);
        // if constexpr (verbose) {
    if (verbose)
    std::cout << "area(2):  ";
    real last_x = poly1[poly1.size() - 1].first;
    real last_y = poly1[poly1.size() - 1].second;

    for (const pt2_t &pt : poly1)
    {
        real x = pt.first;
        real y = pt.second;
        if (verbose)
        std::cout << x << "," << y << ". " << std::endl;

        real xyp = last_x * y;
        real xyn = last_y * x;

        if (verbose)
        std::cout << "xy+= " << xyp << "-" << xyn << ". " << std::endl;

        area += xyp - xyn;
        if (verbose)
        std::cout << "area:" << area << std::endl;

        last_x = x;
        last_y = y;
    }
    if (verbose)
    std::cout << std::endl;
    }
    return area / 2.0;
}

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

void test1_cpoly_intersection__two_points()
{
    double x0 = 0, y0 = 0;
    double x1 = 1, y1 = 1;
    double xm = 0.5, ym = 0.5;
    double x2 = 2, y2 = 2;
    fixedsize_side_metadata_t poly1 = testhelper_polyg(simple_hacky_polygp_t{
        {x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}});

    fixedsize_side_metadata_t poly2 = testhelper_polyg(simple_hacky_polygp_t{
        {xm, ym}, {x2, ym}, {x2, y2}, {xm, y2}});


    collision_of_polyg cr = cpoly_intersection__two_points<double>(poly1, poly2);
    cr.debug_print();
    std::cout << std::endl;
}

void test2_cpoly_intersection__two_points()
{
    double x0 = 0, y0 = 0;
    double x1 = 1, y1 = 1;
    double xm = 0.5, ym = 0.5;
    double x2 = 2, y2 = 2;
    fixedsize_side_metadata_t poly1 = testhelper_polyg(simple_hacky_polygp_t{
        {x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}});

    fixedsize_side_metadata_t poly2 = testhelper_polyg(simple_hacky_polygp_t{
        {xm, ym}, {x2, ym}, {x2, y2}, {xm, y2}});

    collision_of_polyg cr = cpoly_intersection__two_points<double>(poly1, poly2);
    cr.debug_print();
    std::cout << std::endl;
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
