#pragma once

struct side_meta_data_t
{
    double x0, y0;
    double dx, dy; // x1-x0, y1-y0

    side_meta_data_t(const point_t &p0, const point_t &p1)
    {
        this->x0 = p0.x;
        this->y0 = p0.y;
        this->dx = p1.x - p0.x;
        this->dy = p1.y - p0.y;
    }
};
