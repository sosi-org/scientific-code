#pragma once

// one of the many representaitons of point !
/*
template<typename real>
typedef std::pair<real,real> pt2_t_;
// a typedef cannot be a template
*/
typedef std::pair<double, double> pt2_t;

// use only for temporary (test, debug, etc), before conversions
//template<typename real>
typedef std::vector<pt2_t>  simple_hacky_polygp_t;


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
inline simple_hacky_polygp_t  cpoly_intersection(const fixedsize_side_metadata_t &poly1, const fixedsize_side_metadata_t &poly2)
{
    return simple_hacky_polygp_t();
}

template <typename real>
inline real convex_polygon_area (const fixedsize_side_metadata_t &poly1) {
    real area = 0;
    for(const side_meta_data_t& side:  poly1) {
        real xyp = side.x0 * side.y1;
        real xyn = side.y0 * side.x1;
        area += xyp - xyn;
    }
    return area;
}

// area from simple_hacky_polygp_t

template<typename real>
inline real convex_polygon_area2 (const simple_hacky_polygp_t &poly1) {
    real area = 0;

    real last_x = poly1[poly1.size()-1].first;
    real last_y = poly1[poly1.size()-1].second;

    for(const pt2_t& pt:  poly1) {
        real x = pt.first;
        real y = pt.second;

        real xyp = last_x * y;
        real xyn = last_y * x;

        area += xyp - xyn;

        last_x = x;
        last_y = y;
    }
    return area;
}


//rename: make_simple_hacky_polygp
// template <typename real>
//simple_polygi_t
simple_hacky_polygp_t testutil_simply_polygon(simple_hacky_polygp_t coords) {
    simple_hacky_polygp_t pp;
    for( const auto& c : coords) {
        pt2_t pt {c.first, c.second};
        pp.push_back(pt);
    }
    return pp;
}

full_tesselation testutil_tessellation_from_single_polygon(simple_hacky_polygp_t coords) {
    full_tesselation pnp;
    simple_polygi_t pgi;
    side_point_t i = 0;
    for( const auto& c : coords) {
        // pair<> is used only for simpler constructor-literal
        point_t pnt {c.first, c.second};
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
fixedsize_side_metadata_t  t2patch(const simple_polygi_t& polyg, const points_t& points);

void test1_convex_polygon_area() {
    full_tesselation  square
        = testutil_tessellation_from_single_polygon(simple_hacky_polygp_t{{0,0}, {1,0}, {0,1}, {1,1}});
    // convert to side_meta_data_t
    // using constructor side_meta_data_t{p1, p2}
    fixedsize_side_metadata_t poly1 = t2patch(square.trigulation[0], square.points);
    double a = convex_polygon_area<double>(poly1);
    std::cout << a << " expeted.to.be 1" << std::endl;
}
/*
fixedsize_side_metadata_t  t2patch(const simple_polygi_t& polyg)
*/

void test2_convex_polygon_area() {

    simple_hacky_polygp_t square
        = testutil_simply_polygon(simple_hacky_polygp_t{{0,0}, {1,0}, {0,1}, {1,1}});
    double a = convex_polygon_area2<double>(square);

    std::cout << a << " expeted.to.be 1" << std::endl;
}


void dummy1() {
    /*  works but not useful: */
    simple_hacky_polygp_t square
        = testutil_simply_polygon(simple_hacky_polygp_t{{0,0}, {1,0}, {0,1}, {1,1}});
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
