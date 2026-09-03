#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_triangle_primitive_3.h>
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
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace PMP = CGAL::Polygon_mesh_processing;
static_assert(CGAL_VERSION_NR >= 1060200000 && CGAL_VERSION_NR < 1060300000,
              "E6 is pinned to CGAL 6.2.x");
using Kernel = CGAL::Exact_predicates_exact_constructions_kernel;
using Point = Kernel::Point_3;
using Triangle = Kernel::Triangle_3;
using SurfaceMesh = CGAL::Surface_mesh<Point>;

struct InputMesh {
  std::vector<Point> vertices;
  std::vector<std::array<std::size_t, 3>> faces;
  std::array<double, 3> lower{};
  std::array<double, 3> upper{};
};

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

InputMesh read_input(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) {
    fail("cannot open input mesh: " + path);
  }
  std::string magic;
  int version = 0;
  stream >> magic >> version;
  if (magic != "FRAYID_E6_MESH" || version != 1) {
    fail("unsupported input mesh format");
  }
  std::size_t vertex_count = 0;
  std::size_t face_count = 0;
  stream >> vertex_count >> face_count;
  InputMesh result;
  stream >> result.lower[0] >> result.lower[1] >> result.lower[2] >> result.upper[0] >>
      result.upper[1] >> result.upper[2];
  for (int axis = 0; axis < 3; ++axis) {
    if (!std::isfinite(result.lower[axis]) || !std::isfinite(result.upper[axis]) ||
        result.lower[axis] >= result.upper[axis]) {
      fail("outer domain bounds must be finite and strictly ordered");
    }
  }
  result.vertices.reserve(vertex_count);
  for (std::size_t index = 0; index < vertex_count; ++index) {
    double x = 0;
    double y = 0;
    double z = 0;
    stream >> x >> y >> z;
    if (!stream || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      fail("source vertices must be finite");
    }
    for (const auto [value, axis] :
         {std::pair{x, 0}, std::pair{y, 1}, std::pair{z, 2}}) {
      if (value <= result.lower[axis] || value >= result.upper[axis]) {
        fail("source must lie strictly inside the outer domain");
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
      fail("source face contains invalid vertex indices");
    }
    if (CGAL::collinear(result.vertices[face[0]], result.vertices[face[1]],
                        result.vertices[face[2]])) {
      fail("source contains a degenerate face");
    }
    result.faces.push_back(face);
  }
  if (!stream) {
    fail("truncated input mesh");
  }
  return result;
}

SurfaceMesh validate_source(const InputMesh& input) {
  SurfaceMesh mesh;
  std::vector<SurfaceMesh::Vertex_index> vertices;
  vertices.reserve(input.vertices.size());
  for (const auto& point : input.vertices) {
    vertices.push_back(mesh.add_vertex(point));
  }
  for (const auto& face : input.faces) {
    const auto added = mesh.add_face(vertices[face[0]], vertices[face[1]], vertices[face[2]]);
    if (added == SurfaceMesh::null_face()) {
      fail("source is non-manifold or inconsistently connected");
    }
  }
  if (!CGAL::is_valid_polygon_mesh(mesh) || !CGAL::is_triangle_mesh(mesh)) {
    fail("source is not a valid triangle manifold");
  }
  if (!CGAL::is_closed(mesh)) {
    fail("source is not closed");
  }
  if (!PMP::is_outward_oriented(mesh)) {
    fail("source is not consistently outward oriented");
  }
  if (PMP::does_self_intersect(mesh)) {
    fail("source has an exact global self-intersection");
  }
  return mesh;
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

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      std::cerr << "usage: frayid_e6_field_builder INPUT.e6mesh OUTPUT.e6field\n"
                   "   or: frayid_e6_field_builder --audit INPUT.e6mesh\n";
      return 2;
    }
    const bool audit_only = std::string(argv[1]) == "--audit";
    const std::string input_path = audit_only ? argv[2] : argv[1];
    const std::string output_path = audit_only ? std::string() : argv[2];
    const InputMesh input = read_input(input_path);
    const SurfaceMesh validated = validate_source(input);
    if (audit_only) {
      std::cerr << "source_vertices=" << validated.number_of_vertices()
                << " source_faces=" << validated.number_of_faces()
                << " exact_self_intersections=0 outward=1 closed=1\n";
      return 0;
    }

    std::vector<Point> plc_points = input.vertices;
    const auto outer_vertices = box_vertices(input);
    plc_points.insert(plc_points.end(), outer_vertices.begin(), outer_vertices.end());
    std::vector<std::array<std::size_t, 3>> plc_faces = input.faces;
    const auto outer_faces = box_faces(input.vertices.size());
    plc_faces.insert(plc_faces.end(), outer_faces.begin(), outer_faces.end());

    auto ccdt = CGAL::make_conforming_constrained_Delaunay_triangulation_3(
        plc_points, plc_faces, CGAL::parameters::geom_traits(Kernel{}));
    const auto& triangulation = ccdt.triangulation();
    if (!triangulation.is_valid()) {
      fail("CGAL returned an invalid triangulation");
    }

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
    std::vector<std::int8_t> regions(cells.size(), 0);
    std::queue<std::size_t> queue;
    for (std::size_t cell_id = 0; cell_id < cells.size(); ++cell_id) {
      for (int opposite = 0; opposite < 4; ++opposite) {
        if (triangulation.is_infinite(cells[cell_id]->neighbor(opposite))) {
          regions[cell_id] = 1;
          queue.push(cell_id);
          break;
        }
      }
    }
    while (!queue.empty()) {
      const auto cell_id = queue.front();
      queue.pop();
      const auto cell = cells[cell_id];
      for (int opposite = 0; opposite < 4; ++opposite) {
        const auto neighbor = cell->neighbor(opposite);
        if (triangulation.is_infinite(neighbor)) {
          continue;
        }
        bool source_barrier = false;
        if (ccdt.is_facet_constrained(cell, opposite)) {
          const auto constraint = ccdt.face_constraint_index(cell, opposite);
          source_barrier = constraint >= 0 &&
                           static_cast<std::size_t>(constraint) < input.faces.size();
        }
        if (source_barrier) {
          continue;
        }
        const auto neighbor_id = cell_ids.at(neighbor);
        if (regions[neighbor_id] == 0) {
          regions[neighbor_id] = 1;
          queue.push(neighbor_id);
        }
      }
    }
    for (auto& region : regions) {
      if (region == 0) {
        region = -1;
      }
    }
    const auto outside_count = std::count(regions.begin(), regions.end(), 1);
    const auto inside_count = std::count(regions.begin(), regions.end(), -1);
    if (outside_count == 0 || inside_count == 0) {
      fail("source constraints did not separate finite cells into two regions");
    }

    struct InterfaceFace {
      std::array<std::size_t, 3> vertices{};
      std::size_t source_face = 0;
    };
    std::vector<InterfaceFace> interface_faces;
    std::set<std::size_t> interface_vertices;
    for (const auto facet : ccdt.constrained_facets()) {
      const auto constraint = ccdt.face_constraint_index(facet);
      if (constraint < 0 || static_cast<std::size_t>(constraint) >= input.faces.size()) {
        continue;
      }
      InterfaceFace output;
      output.source_face = static_cast<std::size_t>(constraint);
      int slot = 0;
      for (int local = 0; local < 4; ++local) {
        if (local == facet.second) {
          continue;
        }
        output.vertices[slot++] = vertex_ids.at(facet.first->vertex(local));
      }
      const auto& source_face = input.faces[output.source_face];
      const auto source_normal = CGAL::cross_product(
          input.vertices[source_face[1]] - input.vertices[source_face[0]],
          input.vertices[source_face[2]] - input.vertices[source_face[0]]);
      const auto candidate_normal = CGAL::cross_product(
          vertices[output.vertices[1]]->point() - vertices[output.vertices[0]]->point(),
          vertices[output.vertices[2]]->point() - vertices[output.vertices[0]]->point());
      if (CGAL::scalar_product(source_normal, candidate_normal) < 0) {
        std::swap(output.vertices[1], output.vertices[2]);
      }
      for (const auto vertex : output.vertices) {
        interface_vertices.insert(vertex);
      }
      interface_faces.push_back(output);
    }
    if (interface_faces.empty()) {
      fail("no source-constrained facets were recovered");
    }

    using TriangleIterator = std::vector<Triangle>::const_iterator;
    using Primitive = CGAL::AABB_triangle_primitive_3<Kernel, TriangleIterator>;
    using Traits = CGAL::AABB_traits_3<Kernel, Primitive>;
    using Tree = CGAL::AABB_tree<Traits>;
    std::vector<Triangle> source_triangles;
    source_triangles.reserve(input.faces.size());
    for (const auto& face : input.faces) {
      source_triangles.emplace_back(input.vertices[face[0]], input.vertices[face[1]],
                                    input.vertices[face[2]]);
    }
    const Tree tree(source_triangles.begin(), source_triangles.end());

    std::vector<std::int8_t> vertex_regions(vertices.size(), 0);
    for (std::size_t cell_id = 0; cell_id < cells.size(); ++cell_id) {
      for (int local = 0; local < 4; ++local) {
        const auto vertex_id = vertex_ids.at(cells[cell_id]->vertex(local));
        if (interface_vertices.contains(vertex_id)) {
          continue;
        }
        if (vertex_regions[vertex_id] != 0 && vertex_regions[vertex_id] != regions[cell_id]) {
          fail("a non-interface vertex belongs to both signed regions");
        }
        vertex_regions[vertex_id] = regions[cell_id];
      }
    }
    std::vector<double> values(vertices.size(), 0.0);
    for (std::size_t vertex_id = 0; vertex_id < vertices.size(); ++vertex_id) {
      if (interface_vertices.contains(vertex_id)) {
        continue;
      }
      if (vertex_regions[vertex_id] == 0) {
        fail("a finite non-interface vertex has no region label");
      }
      const double distance = std::sqrt(CGAL::to_double(tree.squared_distance(
          vertices[vertex_id]->point())));
      if (!(distance > 0.0) || !std::isfinite(distance)) {
        fail("a non-interface field node has non-positive distance");
      }
      values[vertex_id] = static_cast<double>(vertex_regions[vertex_id]) * distance;
    }

    // Barycentrically subdivide the complete tetrahedral complex.  A node is
    // zero exactly when its underlying simplex belongs to the constrained
    // source subcomplex.  Every non-interface edge, face, and cell therefore
    // gains a strictly signed node, eliminating zero chords and zero cells.
    std::set<std::array<std::size_t, 2>> source_edges;
    std::map<std::array<std::size_t, 3>, std::size_t> source_facets;
    for (const auto& face : interface_faces) {
      auto face_key = face.vertices;
      std::sort(face_key.begin(), face_key.end());
      source_facets.emplace(face_key, face.source_face);
      for (const auto edge : {std::array<std::size_t, 2>{face.vertices[0], face.vertices[1]},
                              std::array<std::size_t, 2>{face.vertices[0], face.vertices[2]},
                              std::array<std::size_t, 2>{face.vertices[1], face.vertices[2]}}) {
        auto edge_key = edge;
        std::sort(edge_key.begin(), edge_key.end());
        source_edges.insert(edge_key);
      }
    }

    std::vector<Point> refined_points;
    refined_points.reserve(vertices.size() + cells.size() * 8);
    std::vector<double> refined_values;
    refined_values.reserve(vertices.size() + cells.size() * 8);
    std::vector<bool> refined_interface;
    refined_interface.reserve(vertices.size() + cells.size() * 8);
    std::vector<std::int8_t> refined_node_regions;
    refined_node_regions.reserve(vertices.size() + cells.size() * 8);
    for (std::size_t vertex_id = 0; vertex_id < vertices.size(); ++vertex_id) {
      refined_points.push_back(vertices[vertex_id]->point());
      refined_values.push_back(values[vertex_id]);
      refined_interface.push_back(interface_vertices.contains(vertex_id));
      refined_node_regions.push_back(vertex_regions[vertex_id]);
    }

    const auto append_refined_node = [&](const Point& point, bool is_interface,
                                         std::int8_t region) -> std::size_t {
      if (!is_interface && region == 0) {
        fail("a non-interface barycentric node has no signed region");
      }
      double value = 0.0;
      if (!is_interface) {
        const double distance =
            std::sqrt(CGAL::to_double(tree.squared_distance(point)));
        if (!(distance > 0.0) || !std::isfinite(distance)) {
          fail("a non-interface barycentric node has non-positive distance");
        }
        value = static_cast<double>(region) * distance;
      }
      const auto id = refined_points.size();
      refined_points.push_back(point);
      refined_values.push_back(value);
      refined_interface.push_back(is_interface);
      refined_node_regions.push_back(is_interface ? 0 : region);
      return id;
    };

    std::map<std::array<std::size_t, 2>, std::size_t> edge_nodes;
    std::map<std::array<std::size_t, 3>, std::size_t> face_nodes;
    struct RefinedCell {
      std::array<std::size_t, 4> vertices{};
      std::int8_t region = 0;
    };
    std::vector<RefinedCell> refined_cells;
    refined_cells.reserve(cells.size() * 24);

    const auto edge_node = [&](std::array<std::size_t, 2> key,
                               std::int8_t region) -> std::size_t {
      std::sort(key.begin(), key.end());
      const auto existing = edge_nodes.find(key);
      if (existing != edge_nodes.end()) {
        const auto node_region = refined_node_regions[existing->second];
        if (node_region != 0 && node_region != region) {
          fail("a non-interface edge is incident to both signed regions");
        }
        return existing->second;
      }
      const bool is_interface = source_edges.contains(key);
      const auto& first = refined_points[key[0]];
      const auto& second = refined_points[key[1]];
      const Point midpoint((first.x() + second.x()) / 2, (first.y() + second.y()) / 2,
                           (first.z() + second.z()) / 2);
      const auto id = append_refined_node(midpoint, is_interface, region);
      edge_nodes.emplace(key, id);
      return id;
    };

    const auto face_node = [&](std::array<std::size_t, 3> key,
                               std::int8_t region) -> std::size_t {
      std::sort(key.begin(), key.end());
      const auto existing = face_nodes.find(key);
      if (existing != face_nodes.end()) {
        const auto node_region = refined_node_regions[existing->second];
        if (node_region != 0 && node_region != region) {
          fail("a non-interface face is incident to both signed regions");
        }
        return existing->second;
      }
      const bool is_interface = source_facets.contains(key);
      const auto& first = refined_points[key[0]];
      const auto& second = refined_points[key[1]];
      const auto& third = refined_points[key[2]];
      const Point centroid((first.x() + second.x() + third.x()) / 3,
                           (first.y() + second.y() + third.y()) / 3,
                           (first.z() + second.z() + third.z()) / 3);
      const auto id = append_refined_node(centroid, is_interface, region);
      face_nodes.emplace(key, id);
      return id;
    };

    for (std::size_t cell_id = 0; cell_id < cells.size(); ++cell_id) {
      std::array<std::size_t, 4> base{};
      for (int local = 0; local < 4; ++local) {
        base[local] = vertex_ids.at(cells[cell_id]->vertex(local));
      }
      const auto& a = refined_points[base[0]];
      const auto& b = refined_points[base[1]];
      const auto& c = refined_points[base[2]];
      const auto& d = refined_points[base[3]];
      const Point centroid((a.x() + b.x() + c.x() + d.x()) / 4,
                           (a.y() + b.y() + c.y() + d.y()) / 4,
                           (a.z() + b.z() + c.z() + d.z()) / 4);
      const auto center_node = append_refined_node(centroid, false, regions[cell_id]);
      std::array<int, 4> permutation{0, 1, 2, 3};
      do {
        const auto vertex_node = base[permutation[0]];
        const auto midpoint_node = edge_node(
            {base[permutation[0]], base[permutation[1]]}, regions[cell_id]);
        const auto facet_node = face_node(
            {base[permutation[0]], base[permutation[1]], base[permutation[2]]},
            regions[cell_id]);
        refined_cells.push_back(
            {{vertex_node, midpoint_node, facet_node, center_node}, regions[cell_id]});
      } while (std::next_permutation(permutation.begin(), permutation.end()));
    }

    std::vector<InterfaceFace> refined_interface_faces;
    refined_interface_faces.reserve(interface_faces.size() * 6);
    for (const auto& face : interface_faces) {
      auto face_key = face.vertices;
      std::sort(face_key.begin(), face_key.end());
      const auto centroid_node = face_nodes.at(face_key);
      std::array<int, 3> permutation{0, 1, 2};
      do {
        InterfaceFace output_face;
        output_face.source_face = face.source_face;
        output_face.vertices = {
            face.vertices[permutation[0]],
            edge_node(
                {face.vertices[permutation[0]], face.vertices[permutation[1]]}, 1),
            centroid_node,
        };
        const auto& source_face = input.faces[output_face.source_face];
        const auto source_normal = CGAL::cross_product(
            input.vertices[source_face[1]] - input.vertices[source_face[0]],
            input.vertices[source_face[2]] - input.vertices[source_face[0]]);
        const auto candidate_normal = CGAL::cross_product(
            refined_points[output_face.vertices[1]] - refined_points[output_face.vertices[0]],
            refined_points[output_face.vertices[2]] - refined_points[output_face.vertices[0]]);
        if (CGAL::scalar_product(source_normal, candidate_normal) < 0) {
          std::swap(output_face.vertices[1], output_face.vertices[2]);
        }
        refined_interface_faces.push_back(output_face);
      } while (std::next_permutation(permutation.begin(), permutation.end()));
    }

    std::ofstream output(output_path);
    if (!output) {
      fail("cannot open output field");
    }
    output << std::setprecision(17);
    output << "FRAYID_E6_FIELD 1\n";
    output << refined_points.size() << ' ' << refined_cells.size() << ' '
           << refined_interface_faces.size() << ' ' << input.faces.size() << "\n";
    output << outside_count * 24 << ' ' << inside_count * 24 << "\n";
    for (std::size_t vertex_id = 0; vertex_id < refined_points.size(); ++vertex_id) {
      const auto& point = refined_points[vertex_id];
      output << CGAL::to_double(point.x()) << ' ' << CGAL::to_double(point.y()) << ' '
             << CGAL::to_double(point.z()) << ' ' << refined_values[vertex_id] << ' '
             << (refined_interface[vertex_id] ? 1 : 0) << "\n";
    }
    for (const auto& cell : refined_cells) {
      auto ids = cell.vertices;
      const auto orientation = CGAL::orientation(
          refined_points[ids[0]], refined_points[ids[1]], refined_points[ids[2]],
          refined_points[ids[3]]);
      if (orientation == CGAL::COPLANAR) {
        fail("barycentric subdivision contains a degenerate tetrahedron");
      }
      if (orientation == CGAL::NEGATIVE) {
        std::swap(ids[0], ids[1]);
      }
      output << ids[0] << ' ' << ids[1] << ' ' << ids[2] << ' ' << ids[3] << ' '
             << static_cast<int>(cell.region) << "\n";
    }
    for (const auto& face : refined_interface_faces) {
      output << face.vertices[0] << ' ' << face.vertices[1] << ' ' << face.vertices[2] << ' '
             << face.source_face << "\n";
    }
    if (!output) {
      fail("failed while writing output field");
    }
    std::cerr << "base_vertices=" << vertices.size() << " base_tetrahedra=" << cells.size()
              << " refined_vertices=" << refined_points.size()
              << " refined_tetrahedra=" << refined_cells.size()
              << " interface_facets=" << refined_interface_faces.size() << " outside="
              << outside_count * 24 << " inside=" << inside_count * 24 << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "E6_FIELD_ERROR: " << error.what() << '\n';
    return 1;
  }
}
