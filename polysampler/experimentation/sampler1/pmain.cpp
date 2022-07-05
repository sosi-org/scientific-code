// import "point_t.hpp";
// import point_t.hpp.gch;
import point_t;

#include <iostream>
int main() {
   point_t p;
   p.x=5;
   std::cout << p.x << std::endl;
   std::cout << p.tostr() << std::endl;
   return 0;
}

