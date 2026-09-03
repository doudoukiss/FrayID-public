// SPDX-License-Identifier: GPL-3.0-or-later
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/alpha_wrap_3.h>
#include <CGAL/version.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

static_assert(CGAL_VERSION_NR >= 1060200000 && CGAL_VERSION_NR < 1060300000,
              "E10 is pinned to CGAL 6.2.x");
using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = Kernel::Point_3;
using SurfaceMesh = CGAL::Surface_mesh<Point>;

struct InputMesh {
  std::vector<Point> vertices;
  std::vector<std::array<std::size_t, 3>> faces;
};

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

InputMesh read_input(const std::string& path) {
  std::ifstream stream(path);
  std::string magic;
  int version = 0;
  std::size_t vertex_count = 0;
  std::size_t face_count = 0;
  std::array<double, 6> ignored_bounds{};
  stream >> magic >> version >> vertex_count >> face_count;
  if (!stream || magic != "FRAYID_E6_MESH" || version != 1) {
    fail("unsupported or truncated input mesh");
  }
  for (double& value : ignored_bounds) stream >> value;
  InputMesh input;
  input.vertices.reserve(vertex_count);
  for (std::size_t index = 0; index < vertex_count; ++index) {
    double x = 0, y = 0, z = 0;
    stream >> x >> y >> z;
    if (!stream || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      fail("nonfinite or truncated input vertex");
    }
    input.vertices.emplace_back(x, y, z);
  }
  input.faces.reserve(face_count);
  for (std::size_t index = 0; index < face_count; ++index) {
    std::array<std::size_t, 3> face{};
    stream >> face[0] >> face[1] >> face[2];
    if (!stream || face[0] >= vertex_count || face[1] >= vertex_count ||
        face[2] >= vertex_count || face[0] == face[1] || face[1] == face[2] ||
        face[2] == face[0]) {
      fail("invalid or truncated input face");
    }
    input.faces.push_back(face);
  }
  return input;
}

std::array<std::size_t, 3> canonical_rotation(std::array<std::size_t, 3> face) {
  if (face[1] < face[0] && face[1] < face[2]) {
    return {face[1], face[2], face[0]};
  }
  if (face[2] < face[0] && face[2] < face[1]) {
    return {face[2], face[0], face[1]};
  }
  return face;
}

int main(int argc, char** argv) {
  try {
    if (argc != 5) {
      std::cerr << "usage: frayid_e10_alpha_wrap INPUT.e6mesh ALPHA OFFSET OUTPUT.e10mesh\n";
      return 2;
    }
    const InputMesh input = read_input(argv[1]);
    const double alpha = std::stod(argv[2]);
    const double offset = std::stod(argv[3]);
    if (!std::isfinite(alpha) || !std::isfinite(offset) || alpha <= 0 || offset <= 0) {
      fail("alpha and offset must be finite and strictly positive");
    }
    SurfaceMesh wrap;
    CGAL::alpha_wrap_3(input.vertices, input.faces, alpha, offset, wrap);
    if (wrap.is_empty()) fail("alpha wrapping returned an empty mesh");

    std::vector<SurfaceMesh::Vertex_index> descriptors;
    descriptors.reserve(wrap.number_of_vertices());
    for (const auto vertex : wrap.vertices()) descriptors.push_back(vertex);
    std::sort(descriptors.begin(), descriptors.end(), [&](const auto left, const auto right) {
      const Point& a = wrap.point(left);
      const Point& b = wrap.point(right);
      return std::tuple{a.x(), a.y(), a.z(), left.idx()} <
             std::tuple{b.x(), b.y(), b.z(), right.idx()};
    });
    std::vector<std::size_t> remap(wrap.num_vertices());
    for (std::size_t index = 0; index < descriptors.size(); ++index) {
      remap[descriptors[index].idx()] = index;
    }
    std::vector<std::array<std::size_t, 3>> faces;
    faces.reserve(wrap.number_of_faces());
    for (const auto face : wrap.faces()) {
      std::array<std::size_t, 3> values{};
      std::size_t corner = 0;
      for (const auto vertex : CGAL::vertices_around_face(wrap.halfedge(face), wrap)) {
        if (corner >= 3) fail("alpha wrap contains a non-triangle face");
        values[corner++] = remap[vertex.idx()];
      }
      if (corner != 3) fail("alpha wrap contains a non-triangle face");
      faces.push_back(canonical_rotation(values));
    }
    std::sort(faces.begin(), faces.end());
    if (descriptors.size() > 3000000 || faces.size() > 6000000) {
      fail("alpha wrap exceeds registered complexity cap");
    }
    std::ofstream output(argv[4]);
    if (!output) fail("cannot open output mesh");
    output << "FRAYID_E10_MESH 1\n" << descriptors.size() << " " << faces.size() << "\n";
    output << std::setprecision(17);
    for (const auto descriptor : descriptors) {
      const Point& point = wrap.point(descriptor);
      output << point.x() << " " << point.y() << " " << point.z() << "\n";
    }
    for (const auto& face : faces) {
      output << face[0] << " " << face[1] << " " << face[2] << "\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "E10_ALPHA_WRAP_ERROR: " << error.what() << "\n";
    return 1;
  }
}
