# Structural predictor specification for the next modelling stage

Status: Stage 3 implementation complete on 2026-11-06. Structural predictors
have been generated on the new equal-area hexagonal support, but no model has
been fitted. Implementation results are in
`docs/geology_foundation/structural_predictors_v1.md`.

This specification uses the verified named-major-structure layer described in
`docs/geology_foundation/major_structures_v2.md`. Its purpose is to define a
transparent structural feature set before any model architecture is selected.
All distances are planar distances in EPSG:32735 and are expressed in
kilometres. All densities are areal line densities in kilometres of structure
per 1,000 km2.

## 1. Inputs

The canonical input is:

- `data/processed/geology_foundation/daly_major_structures_named_v2.gpkg`
  (SHA-256 `10ba59c7bbe771f3e74dfe0e6db3263718f292f95efe394d39a5fcd73413e330`)

The relevant layers are:

| Layer | Geometry | Rows | Use in predictor construction |
|---|---|---:|---|
| `v1_segments_attributed` | LineString | 248 | Observed printed heavy linework with attribution status |
| `named_structure_centrelines` | LineString | 15 | Approximate parent axes, including explicit bridges |
| `bridge_connectors` | LineString | 21 | Explicitly inferred gap links; sensitivity only |
| `named_structure_catalogue` | Point | 15 | Name status, confidence and label anchors |

The tectonic-domain context may be read from:

- `data/processed/geology_foundation/daly_tectonic_domains_v1.gpkg`,
  layer `daly_domains`, fields `domain_id`, `abbreviation`, `domain_name`;
- `data/processed/geology_foundation/daly_domain_boundaries_v1.gpkg`,
  layer `daly_domain_boundaries`, fields `boundary_id`, `boundary_type`,
  `domain_a`, `domain_b`.

The domain boundaries are cartographic class boundaries. They are not mapped
fault surfaces and are not used as a substitute for named structure geometry.

## 2. What is being represented

The structural features distinguish three different objects:

1. **Observed printed linework**: the 248 v1 segments in
   `v1_segments_attributed`. This is the primary source for distance and density.
   Attribution confidence is an attribute, not a geometric error estimate.
2. **Approximate parent axes**: the 15 ordered parts in
   `named_structure_centrelines`. These are useful for parent-system summaries
   but are not observed continuous traces.
3. **Cartographic bridges**: the 21 straight links in `bridge_connectors`.
   These are not observed geology and must not be promoted to primary
   predictor geometry.

The cartographic stroke width of a printed fault-zone symbol is not physical
fault-zone width. No fault-zone polygon, damage-zone width or displacement
surface is available from this layer. A predictor called a "fault-zone
boundary" therefore means the mapped line trace, not a physical polygon edge.

## 3. Attribution sets

The following named subsets are fixed for the next stage:

| Set | Selection rule | Segments | Purpose |
|---|---|---:|---|
| `HM` | `attributed_status` in {`named_high`, `named_medium`} | 47 | Primary named-structure set |
| `HIGH` | `attributed_status` = `named_high` | 35 | Confidence sensitivity set |
| `ALL_NAMED` | `attributed_structure_id` is not null | 54 | Broad named set, including low-confidence support and the disconnected Ky SW component |
| `UNNAMED` | `attributed_structure_id` is null | 194 | Baseline and mapping-texture sensitivity only |

The primary set excludes `named_low` (4) and `named_medium_low` (3). Those seven
segments are retained in `ALL_NAMED` but are not allowed to drive the primary
geological predictor. This prevents weakly attributed junction fragments or the
disconnected Kanyama south-west component from being treated as equivalent to a
directly associated main trace.

Unresolved names (Kasempa, Matebo and Sialinga) have no geometry. They are not
converted to nearest-line proxies and do not appear as named predictors.

## 4. Candidate feature set

The following names are proposed. The final model may use only a subset; that
selection is not made here.

| Feature ID | Definition | Source and rule | Role |
|---|---|---|---|
| `STR_DIST_NAMED_HM_KM` | Distance to the nearest observed printed segment in the `HM` set | `v1_segments_attributed`, status `named_high` or `named_medium` | Primary named-structure proximity |
| `STR_DIST_NAMED_HIGH_KM` | Distance to the nearest `HIGH` segment | `v1_segments_attributed`, status `named_high` | Confidence sensitivity |
| `STR_DIST_NAMED_ALL_KM` | Distance to the nearest `ALL_NAMED` segment | All attributed v1 segments | Broad sensitivity |
| `STR_DIST_<ABBR>_KM` | Distance to each named parent system's observed segments | Group by `attributed_structure_id`; 12 systems with geometry | Exploratory parent-system diagnostics |
| `STR_DIST_NAMED_BRIDGED_KM` | Distance to the union of `named_structure_centrelines` | Including explicit bridge links | Geometry-construction sensitivity only |
| `STR_DIST_MAJOR_INTERSECTION_KM` | Distance to the nearest intersection of two distinct named parent systems | Pairwise intersections of dissolved parent axes; no near-miss inference | Secondary/sensitivity |
| `STR_MAJOR_INTERSECTION_COUNT_25KM` | Count of distinct named-system intersections within 25 km | Same intersection points as above | Secondary/sensitivity |
| `STR_NAMED_LINE_DENSITY_10KM` | Length of `HM` linework within a 10 km radius, per 1,000 km2 | Exact line-buffer intersection length | Structural density, short scale |
| `STR_NAMED_LINE_DENSITY_25KM` | Same quantity within 25 km | Same method | Structural density, medium scale |
| `STR_NAMED_LINE_DENSITY_50KM` | Same quantity within 50 km | Same method | Structural density, regional scale |
| `STR_NAMED_SYSTEM_COUNT_25KM` | Number of distinct named parent systems represented within 25 km | Distinct `attributed_structure_id`; `HM` set | Structural complexity/context |
| `STR_DIST_DALY_DOMAIN_BOUNDARY_KM` | Distance to the nearest Daly domain boundary | `daly_domain_boundaries` | Geological-context sensitivity |
| `STR_UNNAMED_LINE_DENSITY_25KM` | Length of `UNNAMED` heavy linework within 25 km, per 1,000 km2 | `v1_segments_attributed`, null attribution | Mapping-texture and coverage sensitivity |

`STR_DIST_NAMED_HM_KM` is the operational form of "distance to a major
structural boundary" in this project. It is a distance to the mapped line
trace, not to a physical fault-zone polygon. If an independent source later
supplies mapped fault-zone polygons, a separate
`STR_DIST_MAPPED_FAULT_ZONE_BOUNDARY_KM` can be added without changing this
definition.

## 5. Geometry definitions

### 5.1 Distance

For a support point `s` and a line union `L`, compute the shortest 2D Euclidean
distance from `s` to `L` with Shapely in EPSG:32735, then divide by 1,000 to
obtain kilometres. For linework crossing the support point, the distance is
zero. For a polygon support, compute the distance at the chosen quadrature
point; do not substitute a different spatial support silently.

For each parent system, dissolve its accepted segments for the distance
operation. Do not use the label point as the structure. Do not use a buffer
radius to convert a line into a polygon for the primary distance.

### 5.2 Parent-system intersections

Intersections are computed only between dissolved geometries of distinct named
parent systems. A point is retained only if the two parent geometries actually
intersect. Near misses, shared label areas and arbitrary extrapolations are not
intersections. Deduplicate points within 1 m.

An exploratory check of the current v2 layer finds exactly two pairwise
intersections between distinct named parent systems:

- Kafue (`Kf`) and Mubulashi (`Mb`);
- Kanyama (`Ky`) and Lwawa (`LB`).

These are junction contacts in the dissolved parent axes. They are not
automatically interpreted as structural intersections in the geological sense.
Because the count is so small, intersection predictors are secondary and may be
omitted if they provide no within-fold variation.

### 5.3 Structural density

For a radius `R` in kilometres, compute the length of `HM` linework inside the
circular neighbourhood of the support point. The raw feature is:

`density_R = line_length_km / (pi * R^2) * 1000`

This yields kilometres of line per 1,000 km2. Use exact line-buffer intersection
length rather than counting segment midpoints. The three fixed radii (10, 25 and
50 km) are specified before model evaluation. They are not tuned on held-out
occurrence data.

The 10 km scale describes local structural texture, 25 km describes a
district-scale context, and 50 km describes regional structural intensity. The
three scales are correlated by construction and should not be treated as
independent geological variables.

### 5.4 Parent-system count

For each support point, count the distinct `attributed_structure_id` values in
the `HM` set whose linework lies within 25 km. This is a proximity-based
complexity descriptor. It is not a count of all faults, because unnamed heavy
linework and unresolved names are excluded.

## 6. Transformations

Raw distance features are in kilometres. The following transformations may be
compared in later work:

- identity: `d`;
- `log1p(d)`;
- a monotone decreasing spline or exponential decay with a prespecified scale
  family (for example 5, 10, 25 and 50 km).

Raw distance should not be binned at arbitrary thresholds before exploratory
inspection. Density features should be inspected on their natural scale and may
be `log1p`-transformed if their distributions require it.

Any scaling, centring, spline basis or decay-scale choice must be estimated
using training-fold information only. When spatial folds are used, do not
standardise using the full study area.

## 7. Spatial support and coverage

The structural features are support-agnostic, but their meaning changes with
the spatial unit. They must be computed at the exact support used by the model:

- **Point or quadrature support:** compute distance to the line union directly.
- **5 km cell-centroid support:** compute at the cell centroid and record that
  the value is a centroid sample, not an areal average.
- **Equal-area hexagon support:** compute at the hexagon centroid or at the
  specified integration points.
- **Irregular clipped polygons:** a centroid-only distance can be misleading for
  narrow edge slivers. Use polygon-to-line distance as a sensitivity and, where
  required, integrate over the polygon support.

Do not add a separate `cell_area` correction inside the structural distance
itself. If clipped cells remain in the model, their unequal support must be
handled in the observation or exposure component, not hidden in the fault
distance.

Distances outside the mapped frame still produce a value to the nearest mapped
feature, but they do not establish geological absence. Carry an explicit
mapping-coverage limitation into interpretation. If a reliable source-coverage
polygon becomes available, add a separate `STR_MAPPING_COVERAGE` indicator or
mask; do not invent one from modern mining or occurrence data.

## 8. Fold-aware protocol

The structural map is not derived from the occurrence labels, so the geometry
may be assembled across the whole study frame. That does not permit
occurrence-informed feature selection. The following rules apply:

1. Fix the attribution rules and feature definitions before examining
   validation-fold performance.
2. Fix the density radii (10, 25 and 50 km) in advance. Do not choose a radius
   because it improves a held-out fold.
3. Estimate transformations, spline bases, decay scales and missing-data rules
   from training folds only.
4. Use the same geometry-processing algorithm for all folds. Do not remove
   mapped structures from a validation sector merely because its occurrences
   are held out.
5. Retain the historical four-fold V14 partitions for benchmark comparison, but
   evaluate any new modelling claim with buffered spatial separation as well.
   V14 folds have no buffer between sectors, so they are not sufficient evidence
   of geographic transferability.
6. Report fold-specific distributions of each structural feature. A predictor
   with little within-fold variation cannot be interpreted from a pooled
   coefficient or improvement.

## 9. Baselines and ablations

The first modelling experiment should isolate structural information rather
than assume that a richer feature set is better. At minimum:

1. A geological-null baseline using only the occurrence prevalence or a
   spatially explicit intercept.
2. The existing V14 predictor set, reproduced as a historical comparator only.
3. `STR_DIST_NAMED_HM_KM` alone.
4. `STR_DIST_NAMED_HM_KM` plus the domain context.
5. `STR_NAMED_LINE_DENSITY_25KM` as an alternative description of the same
   linework.
6. The `HIGH` and `ALL_NAMED` confidence variants as sensitivity analyses.
7. Distance to bridged parent axes as a geometry-construction sensitivity, not
   as a replacement for the observed-linework primary predictor.

These are design options, not results. No model has been fitted, and no
predictor has been selected on the basis of occurrence performance.

## 10. Exclusions and misinterpretations

The following must not be done:

- Do not attach Kasempa, Matebo or Sialinga to the nearest heavy line.
- Do not use deposit locations to choose which component belongs to a name.
- Do not treat `bridge_connectors` as observed fault traces.
- Do not treat `named_low` junction fragments as independent major structures.
- Do not interpret a long distance to mapped linework as evidence of no fault.
- Do not convert printed stroke width into fault-zone width or displacement.
- Do not assign a kinematic class, age, dip or slip sense from this layer.
- Do not use an unnamed-line density as a proxy for geological prospectivity
  without acknowledging that it also reflects map compilation and linework
  visibility.
- Do not use modern mining footprints or occurrence density to construct an
  "exploration effort" correction. No such correction is available from this
  layer.

## 11. Validation tests before modelling

The feature builder must pass these tests before its outputs are used:

1. **Source identity:** the v2 GeoPackage hash matches the manifest and the
   verification report.
2. **Coverage:** every support point receives a defined `HM` distance; missing
   values are explained, not silently filled.
3. **Zero-distance test:** support points sampled on an `HM` line return zero
   distance within numerical tolerance.
4. **Monotonic test:** distance increases away from an isolated straight test
   segment and is symmetric on both sides.
5. **Nearest-feature audit:** randomly selected support points are checked
   against an independent Shapely nearest-distance calculation.
6. **Density units:** density from a known test line of length `L` inside a
   known circular window equals the expected value to numerical tolerance.
7. **Confidence sensitivity:** `HM`, `HIGH` and `ALL_NAMED` distances are
   retained separately, and their differences are reported rather than hidden.
8. **Bridge sensitivity:** the difference between observed and bridged
   distances is reported by structure and by fold.
9. **Intersection audit:** the feature builder reports the exact intersection
   points and does not create a point from a near miss.
10. **Fold audit:** feature distributions are summarised by fold and by
    geological domain before any model comparison.

## 12. Implementation order

The next implementation steps are:

1. Choose the spatial support for the first redesign experiment. Do not assume
   the clipped V14 5 km grid is the final support.
2. Implement one feature builder that reads the v2 GeoPackage and writes a
   support-keyed table with the fixed feature IDs above.
3. Run the geometric validation tests on a small, documented sample and then on
   the full support.
4. Produce descriptive maps and fold/domain summaries of the structural
   features. Do not fit a model in this step.
5. Freeze a feature configuration and record its hash.
6. Only then specify the model matrix and fitting procedure.

This staged order prevents a sophisticated model from obscuring whether the
structural representation is meaningful in the first place.

## 13. Acceptance criteria

The structural predictor set is acceptable for the next modelling stage only
when:

- the v2 layer and its source hashes are unchanged and verified;
- every feature has an unambiguous definition, source column and spatial
  operation;
- observed linework, approximate parent axes and inferred bridges remain
  distinguishable in the feature table;
- unresolved names are absent rather than silently proxied;
- the three density scales and their expected correlation are documented;
- the intersection limitation is explicit and the two current named-system
  intersections are independently reproducible;
- feature construction is support-aware and does not conceal unequal cell area;
- all transformations are training-fold-aware; and
- no occurrence, deposit-proximity or exploration data were used in the
  attribution or feature definition.
