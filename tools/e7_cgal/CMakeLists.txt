cmake_minimum_required(VERSION 3.22)
project(frayid_e7_cgal LANGUAGES CXX)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

find_package(CGAL 6.2 EXACT REQUIRED COMPONENTS Core)

add_executable(frayid_e7_collision_audit collision_audit.cpp)
target_compile_features(frayid_e7_collision_audit PRIVATE cxx_std_20)
target_link_libraries(frayid_e7_collision_audit PRIVATE CGAL::CGAL CGAL::CGAL_Core)
target_compile_options(frayid_e7_collision_audit PRIVATE -Wall -Wextra -Wpedantic)
