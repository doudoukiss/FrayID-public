// SPDX-License-Identifier: GPL-3.0-or-later
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/intersection.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/version.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace PMP = CGAL::Polygon_mesh_processing;
static_assert(CGAL_VERSION_NR >= 1060200000 && CGAL_VERSION_NR < 1060300000,
              "E10 is pinned to CGAL 6.2.x");
using Kernel = CGAL::Exact_predicates_exact_constructions_kernel;
using Point = Kernel::Point_3;
using SurfaceMesh = CGAL::Surface_mesh<Point>;
using Primitive = CGAL::AABB_face_graph_triangle_primitive<SurfaceMesh>;
using AABBTraits = CGAL::AABB_traits_3<Kernel, Primitive>;
using AABBTree = CGAL::AABB_tree<AABBTraits>;

struct RawMesh {
  std::vector<Point> vertices;
  std::vector<std::array<std::size_t, 3>> faces;
};

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

RawMesh read_mesh(const std::string& path, const std::string& expected_magic, bool bounds) {
  std::ifstream stream(path);
  std::string magic;
  int version = 0;
  std::size_t vertex_count = 0, face_count = 0;
  stream >> magic >> version >> vertex_count >> face_count;
  if (!stream || magic != expected_magic || version != 1) fail("unsupported input mesh");
  if (bounds) {
    double ignored = 0;
    for (int index = 0; index < 6; ++index) stream >> ignored;
  }
  RawMesh result;
  result.vertices.reserve(vertex_count);
  for (std::size_t index = 0; index < vertex_count; ++index) {
    double x = 0, y = 0, z = 0;
    stream >> x >> y >> z;
    if (!stream || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      fail("invalid vertex");
    }
    result.vertices.emplace_back(x, y, z);
  }
  result.faces.reserve(face_count);
  for (std::size_t index = 0; index < face_count; ++index) {
    std::array<std::size_t, 3> face{};
    stream >> face[0] >> face[1] >> face[2];
    if (!stream || face[0] >= vertex_count || face[1] >= vertex_count ||
        face[2] >= vertex_count) fail("invalid face");
    result.faces.push_back(face);
  }
  return result;
}

SurfaceMesh make_surface(const RawMesh& raw) {
  SurfaceMesh mesh;
  std::vector<SurfaceMesh::Vertex_index> vertices;
  vertices.reserve(raw.vertices.size());
  for (const auto& point : raw.vertices) vertices.push_back(mesh.add_vertex(point));
  for (const auto& face : raw.faces) {
    if (mesh.add_face(vertices[face[0]], vertices[face[1]], vertices[face[2]]) ==
        SurfaceMesh::null_face()) fail("output is non-manifold or inconsistently connected");
  }
  return mesh;
}

int main(int argc, char** argv) {
  try {
    if (argc != 4) {
      std::cerr << "usage: frayid_e10_exact_audit SOURCE.e6mesh WRAP.e10mesh REPORT.json\n";
      return 2;
    }
    const RawMesh source = read_mesh(argv[1], "FRAYID_E6_MESH", true);
    const RawMesh raw_wrap = read_mesh(argv[2], "FRAYID_E10_MESH", false);
    SurfaceMesh wrap = make_surface(raw_wrap);
    const bool valid = CGAL::is_valid_polygon_mesh(wrap) && CGAL::is_triangle_mesh(wrap);
    const bool closed = valid && CGAL::is_closed(wrap);
    const bool outward = closed && PMP::is_outward_oriented(wrap);
    std::vector<std::pair<SurfaceMesh::Face_index, SurfaceMesh::Face_index>> intersections;
    if (valid) PMP::self_intersections(wrap, std::back_inserter(intersections));
    auto component_map = wrap.add_property_map<SurfaceMesh::Face_index, std::size_t>(
        "f:e10_component", 0).first;
    const std::size_t components = valid ? PMP::connected_components(wrap, component_map) : 0;
    const long euler = static_cast<long>(wrap.number_of_vertices()) -
                       static_cast<long>(wrap.number_of_edges()) +
                       static_cast<long>(wrap.number_of_faces());

    std::size_t boundary_samples = 0;
    std::size_t outside_samples = 0;
    std::size_t input_intersections = 0;
    if (closed && intersections.empty()) {
      CGAL::Side_of_triangle_mesh<SurfaceMesh, Kernel> side(wrap);
      AABBTree tree(faces(wrap).first, faces(wrap).second, wrap);
      tree.accelerate_distance_queries();
      auto check = [&](const Point& point) {
        const auto result = side(point);
        boundary_samples += static_cast<std::size_t>(result == CGAL::ON_BOUNDARY);
        outside_samples += static_cast<std::size_t>(result == CGAL::ON_UNBOUNDED_SIDE);
      };
      for (const auto& point : source.vertices) check(point);
      for (const auto& face : source.faces) {
        const Point& a = source.vertices[face[0]];
        const Point& b = source.vertices[face[1]];
        const Point& c = source.vertices[face[2]];
        check(CGAL::midpoint(a, b));
        check(CGAL::midpoint(b, c));
        check(CGAL::midpoint(c, a));
        check(CGAL::centroid(a, b, c));
        input_intersections += static_cast<std::size_t>(tree.do_intersect(Kernel::Triangle_3(a, b, c)));
      }
    }
    const bool pass = valid && closed && outward && intersections.empty() && components == 1 &&
                      euler == 2 && outside_samples == 0 && boundary_samples == 0 &&
                      input_intersections == 0;
    std::ofstream output(argv[3]);
    if (!output) fail("cannot open report");
    output << "{\n"
           << "  \"schema_version\": \"e10_exact_audit.v1\",\n"
           << "  \"status\": \"" << (pass ? "pass" : "fail") << "\",\n"
           << "  \"valid_triangle_mesh\": " << (valid ? "true" : "false") << ",\n"
           << "  \"watertight\": " << (closed ? "true" : "false") << ",\n"
           << "  \"outward_oriented\": " << (outward ? "true" : "false") << ",\n"
           << "  \"component_count\": " << components << ",\n"
           << "  \"euler_number\": " << euler << ",\n"
           << "  \"self_intersection_pair_count\": " << intersections.size() << ",\n"
           << "  \"source_boundary_sample_count\": " << boundary_samples << ",\n"
           << "  \"source_outside_sample_count\": " << outside_samples << ",\n"
           << "  \"source_triangle_wrap_intersection_count\": " << input_intersections << "\n"
           << "}\n";
    return pass ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "E10_EXACT_AUDIT_ERROR: " << error.what() << "\n";
    return 1;
  }
}
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_tree.h>
