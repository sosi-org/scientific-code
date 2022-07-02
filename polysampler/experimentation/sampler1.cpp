

/*
#include <algorithm> // for reverse, unique

#include <boost/geometry/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

typedef model::d2::point_xy point;
*/

#include <vector>

// todo: int type
typedef struct
{
    double x;
    double y;
} point_t;
// typedef struct {struct{int point_idx} main; struct {double a;} cache;} side_point;
typedef int side_point_t; // FROM, and TO is the next one.
typedef std::vector<std::vector<side_point_t>> triaglation_t;
// Also add meta-data to each item.
typedef std::vector<point_t> points_t;
// Delaunay triangulation

points_t points = // {{1,2}, {3,4}};
    {point_t{0, 1}, {0.4, 0.8}, {0.8, 0.5}, {1, 0.3}, {0, -1}, {-1, 0}, {-0.3, 0.3}};

triaglation_t trigulation = {{1, 2, 3, 4, 5}, {1, 2, 6}};

// https://github.com/sosi-org/scientific-code/blob/main/beeseyes/pycode/polygon_sampler.py

#include <iostream>
#include <string>

void do_side(const int &from_idx, const int &to_idx)
{
    std::cout << from_idx << "-" << to_idx << ". ";
}

void traverse(const triaglation_t trigulation, const points_t points)
{
    for (auto plg = trigulation.begin(); plg < trigulation.end(); ++plg)
    {
        const auto &polyg_i = *plg;
        std::cout << polyg_i.size() << ": ";

        for (auto vert = polyg_i.begin(); vert < polyg_i.end() - 1; ++vert)
        {
            const auto &from_i = *vert;
            const auto &to_i = *(vert + 1);
            do_side(from_i, to_i);
        }
        do_side(*(polyg_i.end() - 1), *(polyg_i.begin()));

        std::cout << std::endl;
    }
}
/*
    clang++ sampler1.cpp -std=c++2b
*/

int main()
{
    std::cout << "hi" << std::endl;

    traverse(trigulation, points);
    return 0;
}
// ExecutionPolicy
