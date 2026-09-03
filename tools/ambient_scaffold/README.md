# E16 ambient-scaffold constructor

This target extends the retained CGAL 6.2 exact constrained-tetrahedralization
construction under an E16-specific executable. The input may contain the
disjoint, nested moving carrier and fixed public source as separate closed
components. It emits the unsplit complete base complex, every constrained
surface facet with its original face ID, region labels, and fixed bounds. It
fails if an exact tetrahedron degenerates when serialized to the binary64
coordinates used by the path certificate.

The historical E6 source, executable, reports, and experiment status are not
modified or reinterpreted.
