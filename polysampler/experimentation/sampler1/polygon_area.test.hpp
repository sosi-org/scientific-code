#include "./test_helpers.hpp"

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
    assert_equality_real<double>(a, 1.0);
}
/*
fixedsize_side_metadata_t  t2patch(const simple_polygi_t& polyg)
*/

void test2_convex_polygon_area()
{

    simple_hacky_polygp_t square = testutil_simply_polygon(simple_hacky_polygp_t{{0, 0}, {1, 0}, {1, 1}, {0, 1}});
    double a = convex_polygon_area2<double, true>(square);

    std::cout << a << " expected.to.be 1" << std::endl;

    assert_equality_real<double>(a, 1.0);
    assert_equality_i<int>((int)2, 1 + 1);
}
