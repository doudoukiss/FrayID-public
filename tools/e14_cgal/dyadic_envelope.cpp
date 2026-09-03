// SPDX-License-Identifier: GPL-3.0-or-later
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Random.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/convex_hull_3.h>
#include <CGAL/number_utils.h>
#include <CGAL/version.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace PMP = CGAL::Polygon_mesh_processing;
static_assert(CGAL_VERSION_NR >= 1060200000 && CGAL_VERSION_NR < 1060300000,
              "E14 is pinned to CGAL 6.2.x");
using Kernel = CGAL::Exact_predicates_exact_constructions_kernel;
using Point = Kernel::Point_3;
using SurfaceMesh = CGAL::Surface_mesh<Point>;

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

std::vector<Point> read_input_vertices(const std::string& path) {
  std::ifstream stream(path);
  std::string magic;
  int version = 0;
  std::size_t vertex_count = 0;
  std::size_t face_count = 0;
  stream >> magic >> version >> vertex_count >> face_count;
  if (!stream || magic != "FRAYID_E6_MESH" || version != 1 || vertex_count < 4 ||
      face_count < 4) {
    fail("unsupported or truncated input mesh");
  }
  double ignored = 0;
  for (int index = 0; index < 6; ++index) stream >> ignored;
  std::vector<Point> vertices;
  vertices.reserve(vertex_count);
  for (std::size_t index = 0; index < vertex_count; ++index) {
    double x = 0, y = 0, z = 0;
    stream >> x >> y >> z;
    if (!stream || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      fail("nonfinite or truncated input vertex");
    }
    vertices.emplace_back(x, y, z);
  }
  for (std::size_t index = 0; index < face_count; ++index) {
    std::size_t a = 0, b = 0, c = 0;
    stream >> a >> b >> c;
    if (!stream || a >= vertex_count || b >= vertex_count || c >= vertex_count) {
      fail("invalid or truncated input face");
    }
  }
  return vertices;
}

std::array<double, 6> bounds(const std::vector<Point>& input) {
  std::array<double, 6> result{
      CGAL::to_double(input.front().x()), CGAL::to_double(input.front().y()),
      CGAL::to_double(input.front().z()), CGAL::to_double(input.front().x()),
      CGAL::to_double(input.front().y()), CGAL::to_double(input.front().z())};
  for (const Point& point : input) {
    const double values[3] = {CGAL::to_double(point.x()), CGAL::to_double(point.y()),
                              CGAL::to_double(point.z())};
    for (int axis = 0; axis < 3; ++axis) {
      result[axis] = std::min(result[axis], values[axis]);
      result[axis + 3] = std::max(result[axis + 3], values[axis]);
    }
  }
  return result;
}

std::vector<Point> deterministic_general_position(const std::vector<Point>& input,
                                                  const std::array<double, 6>& box) {
  const double scale = std::max({box[3] - box[0], box[4] - box[1], box[5] - box[2]});
  if (!std::isfinite(scale) || scale <= 0) fail("input has no three-dimensional extent");
  const double epsilon = scale * 1e-10;
  std::vector<Point> perturbed;
  perturbed.reserve(input.size());
  for (std::size_t index = 0; index < input.size(); ++index) {
    auto offset = [&](std::size_t salt) {
      const std::size_t mixed = (index * 1103515245ULL + salt * 2654435761ULL) % 1000003ULL;
      return epsilon * (static_cast<double>(mixed) / 1000003.0 - 0.5);
    };
    const Point& point = input[index];
    perturbed.emplace_back(CGAL::to_double(point.x()) + offset(1),
                           CGAL::to_double(point.y()) + offset(2),
                           CGAL::to_double(point.z()) + offset(3));
  }
  return perturbed;
}

std::array<std::size_t, 3> canonical_rotation(std::array<std::size_t, 3> face) {
  if (face[1] < face[0] && face[1] < face[2]) return {face[1], face[2], face[0]};
  if (face[2] < face[0] && face[2] < face[1]) return {face[2], face[0], face[1]};
  return face;
}

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      std::cerr << "usage: frayid_e14_dyadic_envelope INPUT.e6mesh OUTPUT.e10mesh\n";
      return 2;
    }
    const std::vector<Point> input = read_input_vertices(argv[1]);
    const std::array<double, 6> box = bounds(input);
    const std::vector<Point> hull_input = deterministic_general_position(input, box);
    CGAL::get_default_random() = CGAL::Random(20260901);
    SurfaceMesh envelope;
    CGAL::convex_hull_3(hull_input.begin(), hull_input.end(), envelope);
    if (!CGAL::is_triangle_mesh(envelope) || !CGAL::is_closed(envelope)) {
      fail("convex hull is not a closed triangle mesh");
    }
    if (!PMP::is_outward_oriented(envelope)) PMP::reverse_face_orientations(envelope);

    const Point center((box[0] + box[3]) * 0.5, (box[1] + box[4]) * 0.5,
                       (box[2] + box[5]) * 0.5);
    const Kernel::FT expansion = Kernel::FT(129) / Kernel::FT(128);
    const double dx = box[3] - box[0];
    const double dy = box[4] - box[1];
    const double dz = box[5] - box[2];
    const double diagonal = std::sqrt(dx * dx + dy * dy + dz * dz);
    double magnitude = diagonal;
    for (double value : box) magnitude = std::max(magnitude, std::abs(value));
    if (!std::isfinite(magnitude) || magnitude <= 0) fail("invalid input magnitude");
    const int grid_exponent = static_cast<int>(std::floor(std::log2(magnitude))) - 40;
    const double grid = std::ldexp(1.0, grid_exponent);
    if (!std::isfinite(grid) || grid <= 0) fail("invalid dyadic grid");
    constexpr double maximum_grid_integer = 35184372088832.0;  // 2^45
    for (const auto vertex : envelope.vertices()) {
      const Point expanded = center + expansion * (envelope.point(vertex) - center);
      const double raw[3] = {CGAL::to_double(expanded.x()), CGAL::to_double(expanded.y()),
                             CGAL::to_double(expanded.z())};
      double snapped[3]{};
      for (int axis = 0; axis < 3; ++axis) {
        const double integer = std::round(raw[axis] / grid);
        if (!std::isfinite(integer) || std::abs(integer) > maximum_grid_integer) {
          fail("dyadic grid integer exceeds the registered guard");
        }
        snapped[axis] = integer * grid;
      }
      envelope.point(vertex) = Point(snapped[0], snapped[1], snapped[2]);
    }
    if (!PMP::is_outward_oriented(envelope)) PMP::reverse_face_orientations(envelope);

    std::vector<SurfaceMesh::Vertex_index> descriptors;
    descriptors.reserve(envelope.number_of_vertices());
    for (const auto vertex : envelope.vertices()) descriptors.push_back(vertex);
    std::sort(descriptors.begin(), descriptors.end(), [&](const auto left, const auto right) {
      const Point& a = envelope.point(left);
      const Point& b = envelope.point(right);
      return std::tuple{CGAL::to_double(a.x()), CGAL::to_double(a.y()),
                        CGAL::to_double(a.z()), left.idx()} <
             std::tuple{CGAL::to_double(b.x()), CGAL::to_double(b.y()),
                        CGAL::to_double(b.z()), right.idx()};
    });
    std::vector<std::size_t> remap(envelope.num_vertices());
    for (std::size_t index = 0; index < descriptors.size(); ++index) {
      remap[descriptors[index].idx()] = index;
    }
    std::vector<std::array<std::size_t, 3>> faces;
    faces.reserve(envelope.number_of_faces());
    for (const auto face : envelope.faces()) {
      std::array<std::size_t, 3> values{};
      std::size_t corner = 0;
      for (const auto vertex : CGAL::vertices_around_face(envelope.halfedge(face), envelope)) {
        if (corner >= 3) fail("envelope contains a non-triangle face");
        values[corner++] = remap[vertex.idx()];
      }
      if (corner != 3) fail("envelope contains a non-triangle face");
      faces.push_back(canonical_rotation(values));
    }
    std::sort(faces.begin(), faces.end());

    std::ofstream output(argv[2]);
    if (!output) fail("cannot open output mesh");
    output << "FRAYID_E10_MESH 1\n" << descriptors.size() << " " << faces.size() << "\n";
    output << std::setprecision(17);
    for (const auto descriptor : descriptors) {
      const Point& point = envelope.point(descriptor);
      output << CGAL::to_double(point.x()) << " " << CGAL::to_double(point.y()) << " "
             << CGAL::to_double(point.z()) << "\n";
    }
    for (const auto& face : faces) {
      output << face[0] << " " << face[1] << " " << face[2] << "\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "E14_DYADIC_ENVELOPE_ERROR: " << error.what() << "\n";
    return 1;
  }
}
