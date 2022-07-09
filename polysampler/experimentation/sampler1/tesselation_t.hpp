export module tesselation_t;
// #pragma once

#include <vector>

// #include "point_t.hpp"
import point_t;

export {
// typedef struct {struct{int point_idx} main; struct {double a;} cache;} side_point;
typedef int side_point_t; // FROM, and TO is the next one.

typedef std::vector<std::vector<side_point_t>> tesselation_t; // old name: triaglation_t
// Also add meta-data to each item.

typedef std::vector<point_t> points_t;
// Delaunay triangulation

}
/*
todo: rename `points_t`. Too similar to `point_t`

*/
