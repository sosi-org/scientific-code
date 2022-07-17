#pragma once

// todo: separate helpers for varisous aspects
// keep global and multi-purpose ones here in this file

#include <cassert>
#include <iostream>

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
        assert(i == pnp.points.size()-1);
        pgi.push_back(i);
        ++i;
    }
    pnp.trigulation.push_back(pgi);
    return pnp;
}

void dummy1()
{
    /*  works but not useful: */
    simple_hacky_polygp_t square = testutil_simply_polygon(simple_hacky_polygp_t{{0, 0}, {1, 0}, {1, 1}, {0, 1}});
}

fixedsize_polygon_with_side_metadata_t testhelper_polyg(const simple_hacky_polygp_t &shp)
{
    // based on: test1_convex_polygon_area
    full_tesselation square1 = testutil_tessellation_from_single_polygon(shp);
    fixedsize_polygon_with_side_metadata_t poly = t2patch(square1.trigulation[0], square1.points);
    return poly;
}

#include <cmath>

// C++20:

template <typename real>
requires(!std::integral<real>)
void assert_equality_real(real actual_value, real expected_value)
{
    constexpr real ε = 0.00000000001;
    if (std::abs(actual_value - expected_value) < ε)
    {
        std::cout << "passed. (" << actual_value << " expected.to.be ~ " << expected_value << ")" << std::endl;
    }
    else
    {
        std::cout << "failure: expected (actual value) " << actual_value << " to almost-equal " << expected_value << "(expected value)" << std::endl;
    }
}

template <typename T>
requires std::integral<T>
void assert_equality_i(T actual_value, T expected_value)
{
    if (actual_value == expected_value)
    {
        std::cout << "passed. (" << actual_value << " expected.to.be " << expected_value << ")" << std::endl;
    }
    else
    {
        std::cout << "failure: expected (actual value) " << actual_value << " to almost-equal " << expected_value << "(expected value)" << std::endl;
    }
}

template <class T>
concept Integral = std::is_integral<T>::value;
