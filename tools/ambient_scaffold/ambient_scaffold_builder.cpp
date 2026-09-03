// SPDX-License-Identifier: GPL-3.0-or-later
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/intersection.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/make_conforming_constrained_Delaunay_triangulation_3.h>
#include <CGAL/number_utils.h>
#include <CGAL/version.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace PMP = CGAL::Polygon_mesh_processing;
static_assert(CGAL_VERSION_NR >= 1060200000 && CGAL_VERSION_NR < 1060300000,
              "E16 is pinned to CGAL 6.2.x");
using Kernel = CGAL::Exact_predicates_exact_constructions_kernel;
using Point = Kernel::Point_3;
using SurfaceMesh = CGAL::Surface_mesh<Point>;

struct InputMesh {
  std::vector<Point> vertices;
  std::vector<std::array<std::size_t, 3>> faces;
  std::array<double, 3> lower{};
  std::array<double, 3> upper{};
};

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

InputMesh read_input(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) fail("cannot open input mesh");
  std::string magic;
  int version = 0;
  std::size_t vertex_count = 0, face_count = 0;
  stream >> magic >> version >> vertex_count >> face_count;
  if (!stream || magic != "FRAYID_E6_MESH" || version != 1) {
    fail("unsupported input mesh format");
  }
  InputMesh result;
  stream >> result.lower[0] >> result.lower[1] >> result.lower[2] >> result.upper[0] >>
      result.upper[1] >> result.upper[2];
  for (int axis = 0; axis < 3; ++axis) {
    if (!std::isfinite(result.lower[axis]) || !std::isfinite(result.upper[axis]) ||
        result.lower[axis] >= result.upper[axis]) {
      fail("outer bounds must be finite and strictly ordered");
    }
  }
  result.vertices.reserve(vertex_count);
  for (std::size_t index = 0; index < vertex_count; ++index) {
    double x = 0, y = 0, z = 0;
    stream >> x >> y >> z;
    if (!stream || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      fail("source vertex is invalid");
    }
    const std::array<double, 3> values{x, y, z};
    for (int axis = 0; axis < 3; ++axis) {
      if (values[axis] <= result.lower[axis] || values[axis] >= result.upper[axis]) {
        fail("source must lie strictly inside the outer box");
      }
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
      fail("source face is invalid");
    }
    if (CGAL::collinear(result.vertices[face[0]], result.vertices[face[1]],
                        result.vertices[face[2]])) {
      fail("source face is degenerate");
    }
    result.faces.push_back(face);
  }
  if (!stream) fail("truncated input mesh");
  return result;
}

void validate_source(const InputMesh& input) {
  SurfaceMesh mesh;
  std::vector<SurfaceMesh::Vertex_index> vertices;
  vertices.reserve(input.vertices.size());
  for (const auto& point : input.vertices) vertices.push_back(mesh.add_vertex(point));
  for (const auto& face : input.faces) {
    if (mesh.add_face(vertices[face[0]], vertices[face[1]], vertices[face[2]]) ==
        SurfaceMesh::null_face()) {
      fail("source components are non-manifold or inconsistently oriented");
    }
  }
  if (!CGAL::is_valid_polygon_mesh(mesh) || !CGAL::is_triangle_mesh(mesh) ||
      !CGAL::is_closed(mesh) || !PMP::is_outward_oriented(mesh) ||
      PMP::does_self_intersect(mesh)) {
    fail("source components failed exact manifold/orientation/intersection validation");
  }
}

std::array<Point, 8> box_vertices(const InputMesh& input) {
  const auto& lo = input.lower;
  const auto& hi = input.upper;
  return {Point(lo[0], lo[1], lo[2]), Point(hi[0], lo[1], lo[2]),
          Point(hi[0], hi[1], lo[2]), Point(lo[0], hi[1], lo[2]),
          Point(lo[0], lo[1], hi[2]), Point(hi[0], lo[1], hi[2]),
          Point(hi[0], hi[1], hi[2]), Point(lo[0], hi[1], hi[2])};
}

std::array<std::array<std::size_t, 3>, 12> box_faces(std::size_t offset) {
  const auto f = [offset](std::size_t a, std::size_t b, std::size_t c) {
    return std::array<std::size_t, 3>{offset + a, offset + b, offset + c};
  };
  return {f(0, 2, 1), f(0, 3, 2), f(4, 5, 6), f(4, 6, 7),
          f(0, 1, 5), f(0, 5, 4), f(3, 7, 6), f(3, 6, 2),
          f(0, 4, 7), f(0, 7, 3), f(1, 2, 6), f(1, 6, 5)};
}

double serialized_determinant(const std::array<Point, 4>& points) {
  std::array<std::array<double, 3>, 3> edge{};
  for (int column = 0; column < 3; ++column) {
    edge[column][0] = CGAL::to_double(points[column + 1].x()) - CGAL::to_double(points[0].x());
    edge[column][1] = CGAL::to_double(points[column + 1].y()) - CGAL::to_double(points[0].y());
    edge[column][2] = CGAL::to_double(points[column + 1].z()) - CGAL::to_double(points[0].z());
  }
  return edge[0][0] * (edge[1][1] * edge[2][2] - edge[1][2] * edge[2][1]) -
         edge[1][0] * (edge[0][1] * edge[2][2] - edge[0][2] * edge[2][1]) +
         edge[2][0] * (edge[0][1] * edge[1][2] - edge[0][2] * edge[1][1]);
}

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      std::cerr << "usage: frayid_e16_ambient_scaffold_builder INPUT.e6mesh "
                   "OUTPUT.e16scaffold\n";
      return 2;
    }
    const InputMesh input = read_input(argv[1]);
    validate_source(input);
    std::vector<Point> plc_points = input.vertices;
    const auto outer_vertices = box_vertices(input);
    plc_points.insert(plc_points.end(), outer_vertices.begin(), outer_vertices.end());
    std::vector<std::array<std::size_t, 3>> plc_faces = input.faces;
    const auto outer_faces = box_faces(input.vertices.size());
    plc_faces.insert(plc_faces.end(), outer_faces.begin(), outer_faces.end());

    auto ccdt = CGAL::make_conforming_constrained_Delaunay_triangulation_3(
        plc_points, plc_faces, CGAL::parameters::geom_traits(Kernel{}));
    const auto& triangulation = ccdt.triangulation();
    if (!triangulation.is_valid()) fail("CGAL returned an invalid triangulation");
    using Triangulation = std::remove_cvref_t<decltype(triangulation)>;
    using VertexHandle = typename Triangulation::Vertex_handle;
    using CellHandle = typename Triangulation::Cell_handle;

    std::map<VertexHandle, std::size_t> vertex_ids;
    std::vector<VertexHandle> vertices;
    vertices.reserve(triangulation.number_of_vertices());
    for (const auto handle : triangulation.finite_vertex_handles()) {
      vertex_ids.emplace(handle, vertices.size());
      vertices.push_back(handle);
    }
    std::map<CellHandle, std::size_t> cell_ids;
    std::vector<CellHandle> cells;
    cells.reserve(triangulation.number_of_finite_cells());
    for (const auto handle : triangulation.finite_cell_handles()) {
      cell_ids.emplace(handle, cells.size());
      cells.push_back(handle);
    }

    std::vector<std::int16_t> regions(cells.size(), 0);
    std::queue<std::size_t> queue;
    for (std::size_t id = 0; id < cells.size(); ++id) {
      for (int opposite = 0; opposite < 4; ++opposite) {
        if (triangulation.is_infinite(cells[id]->neighbor(opposite))) {
          regions[id] = 1;
          queue.push(id);
          break;
        }
      }
    }
    while (!queue.empty()) {
      const auto id = queue.front();
      queue.pop();
      for (int opposite = 0; opposite < 4; ++opposite) {
        const auto neighbor = cells[id]->neighbor(opposite);
        if (triangulation.is_infinite(neighbor)) continue;
        bool source_barrier = false;
        if (ccdt.is_facet_constrained(cells[id], opposite)) {
          const auto constraint = ccdt.face_constraint_index(cells[id], opposite);
          source_barrier = constraint >= 0 &&
                           static_cast<std::size_t>(constraint) < input.faces.size();
        }
        if (source_barrier) continue;
        const auto neighbor_id = cell_ids.at(neighbor);
        if (regions[neighbor_id] == 0) {
          regions[neighbor_id] = 1;
          queue.push(neighbor_id);
        }
      }
    }
    for (auto& region : regions) {
      if (region == 0) region = -1;
    }

    struct InterfaceFace {
      std::array<std::size_t, 3> vertices{};
      std::size_t source_face = 0;
    };
    std::vector<InterfaceFace> interface_faces;
    std::set<std::size_t> covered_source_faces;
    for (const auto facet : ccdt.constrained_facets()) {
      const auto constraint = ccdt.face_constraint_index(facet);
      if (constraint < 0 || static_cast<std::size_t>(constraint) >= input.faces.size()) continue;
      InterfaceFace output;
      output.source_face = static_cast<std::size_t>(constraint);
      int slot = 0;
      for (int local = 0; local < 4; ++local) {
        if (local != facet.second) output.vertices[slot++] = vertex_ids.at(facet.first->vertex(local));
      }
      const auto& source = input.faces[output.source_face];
      const auto source_normal = CGAL::cross_product(
          input.vertices[source[1]] - input.vertices[source[0]],
          input.vertices[source[2]] - input.vertices[source[0]]);
      const auto candidate_normal = CGAL::cross_product(
          vertices[output.vertices[1]]->point() - vertices[output.vertices[0]]->point(),
          vertices[output.vertices[2]]->point() - vertices[output.vertices[0]]->point());
      if (CGAL::scalar_product(source_normal, candidate_normal) < 0) {
        std::swap(output.vertices[1], output.vertices[2]);
      }
      interface_faces.push_back(output);
      covered_source_faces.insert(output.source_face);
    }
    if (covered_source_faces.size() != input.faces.size()) {
      fail("not every source face was recovered as a conforming subcomplex");
    }

    struct OutputCell {
      std::array<std::size_t, 4> vertices{};
      std::int16_t region = 0;
    };
    std::vector<OutputCell> output_cells;
    output_cells.reserve(cells.size());
    for (std::size_t id = 0; id < cells.size(); ++id) {
      OutputCell output;
      output.region = regions[id];
      std::array<Point, 4> points{};
      for (int local = 0; local < 4; ++local) {
        output.vertices[local] = vertex_ids.at(cells[id]->vertex(local));
        points[local] = cells[id]->vertex(local)->point();
      }
      const auto exact_orientation = CGAL::orientation(points[0], points[1], points[2], points[3]);
      if (exact_orientation == CGAL::COPLANAR) fail("CGAL produced a degenerate cell");
      if (exact_orientation == CGAL::NEGATIVE) {
        std::swap(output.vertices[0], output.vertices[1]);
        std::swap(points[0], points[1]);
      }
      const double determinant = serialized_determinant(points);
      if (!std::isfinite(determinant) || determinant == 0.0) {
        fail("binary64 serialization degenerates an exact tetrahedron");
      }
      if (determinant < 0.0) std::swap(output.vertices[0], output.vertices[1]);
      output_cells.push_back(output);
    }

    std::ofstream output(argv[2]);
    if (!output) fail("cannot open output scaffold");
    output << std::setprecision(17);
    output << "FRAYID_E16_SCAFFOLD 1\n";
    output << vertices.size() << ' ' << output_cells.size() << ' ' << interface_faces.size()
           << ' ' << input.faces.size() << "\n";
    output << input.lower[0] << ' ' << input.lower[1] << ' ' << input.lower[2] << ' '
           << input.upper[0] << ' ' << input.upper[1] << ' ' << input.upper[2] << "\n";
    for (const auto vertex : vertices) {
      const auto& point = vertex->point();
      output << CGAL::to_double(point.x()) << ' ' << CGAL::to_double(point.y()) << ' '
             << CGAL::to_double(point.z()) << "\n";
    }
    for (const auto& cell : output_cells) {
      output << cell.vertices[0] << ' ' << cell.vertices[1] << ' ' << cell.vertices[2] << ' '
             << cell.vertices[3] << ' ' << cell.region << "\n";
    }
    for (const auto& face : interface_faces) {
      output << face.vertices[0] << ' ' << face.vertices[1] << ' ' << face.vertices[2] << ' '
             << face.source_face << "\n";
    }
    if (!output) fail("failed while writing output scaffold");
    const auto outside = std::count(regions.begin(), regions.end(), 1);
    const auto inside = std::count(regions.begin(), regions.end(), -1);
    std::cerr << "vertices=" << vertices.size() << " tetrahedra=" << output_cells.size()
              << " interface_facets=" << interface_faces.size() << " outside=" << outside
              << " inside=" << inside << " serialized_nonpositive=0\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "E16_SCAFFOLD_ERROR: " << error.what() << '\n';
    return 1;
  }
}
