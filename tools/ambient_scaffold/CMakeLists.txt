cmake_minimum_required(VERSION 3.20)
project(frayid_ambient_scaffold LANGUAGES CXX)

find_package(CGAL 6.2 EXACT REQUIRED)

# The public E16 constructor extends the retained E6 exact constrained-
# tetrahedralization construction with a base-complex interchange whose actual
# binary64 tetrahedra are nondegenerate. Historical E6 targets remain untouched.
add_executable(
  frayid_e16_ambient_scaffold_builder
  ambient_scaffold_builder.cpp
)
target_compile_features(frayid_e16_ambient_scaffold_builder PRIVATE cxx_std_20)
target_link_libraries(frayid_e16_ambient_scaffold_builder PRIVATE CGAL::CGAL)
