#pragma once

/*
intersection between two convex polygons.
*/
template <typename real>
inline cpoly_intersection<real> cpoly_intersection(const fixedsize_side_metadata_t &poly1, const fixedsize_side_metadata_t &poly2)
{
}

template <typename real>
inline real convex_polygon_area (const fixedsize_side_metadata_t &poly1) {
    real area = 0;
    for(const side_meta_data_t& side:  poly1) {
        real xyp = poly1.x0 * poly1.y1;
        real xyn = poly1.y0 * poly1.x1;
        area += xyp - xyn;
    }
    return area;
}



/*
test{} (
    (0,0), (1,0), (0,1), (1,1)
    ->
    1.0
);
*/
function test1_convex_polygon_area() {

    // convert to side_meta_data_t
    // using constructor side_meta_data_t{p1, p2}
    fixedsize_side_metadata_t poly1 = ;
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