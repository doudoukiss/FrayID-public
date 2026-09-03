#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/version.h>

#include <algorithm>
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
              "E7 is pinned to CGAL 6.2.x");
using Kernel = CGAL::Exact_predicates_exact_constructions_kernel;
using Point = Kernel::Point_3;
using SurfaceMesh = CGAL::Surface_mesh<Point>;

struct InputMesh {
  std::vector<Point> vertices;
  std::vector<std::array<std::size_t, 3>> faces;
};

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

InputMesh read_input(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) {
    fail("cannot open input mesh");
  }
  std::string magic;
  int version = 0;
  std::size_t vertex_count = 0;
  std::size_t face_count = 0;
  std::array<double, 6> ignored_bounds{};
  stream >> magic >> version >> vertex_count >> face_count;
  if (magic != "FRAYID_E6_MESH" || version != 1) {
    fail("unsupported input format");
  }
  for (double& value : ignored_bounds) {
    stream >> value;
  }
  InputMesh result;
  result.vertices.reserve(vertex_count);
  for (std::size_t index = 0; index < vertex_count; ++index) {
    double x = 0;
    double y = 0;
    double z = 0;
    stream >> x >> y >> z;
    if (!stream || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      fail("nonfinite or truncated vertex");
    }
    result.vertices.emplace_back(x, y, z);
  }
  result.faces.reserve(face_count);
  for (std::size_t index = 0; index < face_count; ++index) {
    std::array<std::size_t, 3> face{};
    stream >> face[0] >> face[1] >> face[2];
    if (!stream || face[0] >= vertex_count || face[1] >= vertex_count ||
        face[2] >= vertex_count || face[0] == face[1] || face[1] == face[2] ||
        face[2] == face[0]) {
      fail("invalid or truncated face");
    }
    result.faces.push_back(face);
  }
  return result;
}

std::size_t shared_vertex_count(const std::array<std::size_t, 3>& first,
                                const std::array<std::size_t, 3>& second) {
  std::size_t count = 0;
  for (const auto left : first) {
    count += static_cast<std::size_t>(std::find(second.begin(), second.end(), left) != second.end());
  }
  return count;
}

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      std::cerr << "usage: frayid_e7_collision_audit INPUT.e6mesh OUTPUT.json\n";
      return 2;
    }
    const InputMesh input = read_input(argv[1]);
    SurfaceMesh mesh;
    std::vector<SurfaceMesh::Vertex_index> vertices;
    vertices.reserve(input.vertices.size());
    for (const auto& point : input.vertices) {
      vertices.push_back(mesh.add_vertex(point));
    }
    for (const auto& face : input.faces) {
      if (mesh.add_face(vertices[face[0]], vertices[face[1]], vertices[face[2]]) ==
          SurfaceMesh::null_face()) {
        fail("source is non-manifold or inconsistently connected");
      }
    }
    if (!CGAL::is_valid_polygon_mesh(mesh) || !CGAL::is_triangle_mesh(mesh)) {
      fail("source is not a valid triangle mesh");
    }
    using FacePair = std::pair<SurfaceMesh::Face_index, SurfaceMesh::Face_index>;
    std::vector<FacePair> pairs;
    PMP::self_intersections(mesh, std::back_inserter(pairs));
    std::sort(pairs.begin(), pairs.end(), [](const FacePair& left, const FacePair& right) {
      const std::pair<std::size_t, std::size_t> left_key{
          std::min(left.first.idx(), left.second.idx()),
          std::max(left.first.idx(), left.second.idx())};
      const std::pair<std::size_t, std::size_t> right_key{
          std::min(right.first.idx(), right.second.idx()),
          std::max(right.first.idx(), right.second.idx())};
      return left_key < right_key;
    });
    std::size_t shared_edge = 0;
    std::size_t shared_vertex = 0;
    std::size_t disjoint = 0;
    for (const auto& pair : pairs) {
      const auto shared = shared_vertex_count(input.faces[pair.first.idx()],
                                              input.faces[pair.second.idx()]);
      shared_edge += static_cast<std::size_t>(shared >= 2);
      shared_vertex += static_cast<std::size_t>(shared == 1);
      disjoint += static_cast<std::size_t>(shared == 0);
    }
    std::ofstream output(argv[2]);
    if (!output) {
      fail("cannot open output report");
    }
    output << "{\n  \"schema_version\": \"e7_cgal_collision_audit.v1\",\n"
           << "  \"cgal_version\": \"6.2\",\n"
           << "  \"kernel\": \"Exact_predicates_exact_constructions_kernel\",\n"
           << "  \"vertex_count\": " << input.vertices.size() << ",\n"
           << "  \"face_count\": " << input.faces.size() << ",\n"
           << "  \"closed\": " << (CGAL::is_closed(mesh) ? "true" : "false") << ",\n"
           << "  \"intersection_pair_count\": " << pairs.size() << ",\n"
           << "  \"classification\": {\"shared_edge\": " << shared_edge
           << ", \"shared_vertex\": " << shared_vertex << ", \"disjoint\": " << disjoint
           << "},\n  \"face_pairs\": [";
    for (std::size_t index = 0; index < pairs.size(); ++index) {
      const auto first = std::min(pairs[index].first.idx(), pairs[index].second.idx());
      const auto second = std::max(pairs[index].first.idx(), pairs[index].second.idx());
      const auto shared = shared_vertex_count(input.faces[first], input.faces[second]);
      const char* classification = shared >= 2 ? "shared_edge" :
                                   shared == 1 ? "shared_vertex" : "disjoint";
      output << (index == 0 ? "\n" : ",\n") << "    {\"first\": " << first
             << ", \"second\": " << second << ", \"classification\": \""
             << classification << "\"}";
    }
    output << (pairs.empty() ? "" : "\n  ") << "]\n}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "E7_COLLISION_AUDIT_ERROR: " << error.what() << "\n";
    return 1;
  }
}
