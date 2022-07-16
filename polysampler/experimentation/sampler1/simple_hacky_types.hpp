#pragma once

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


//void debug_print(const auto& cout, const simple_hacky_polygp_t& poly ) {
// template <typename COUT>
// void debug_print(const COUT& cout, const simple_hacky_polygp_t& poly ) {
void debug_print(const simple_hacky_polygp_t& poly ) {
    std::cout << poly.size() << ":";
    for(auto v : poly) {
        std::cout << "(" << v.first << "," << v.second << ") ";
    }
}
