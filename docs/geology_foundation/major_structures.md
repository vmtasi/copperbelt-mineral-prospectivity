# Daly et al. (2025) major structural linework

Status: vectorised and verified on 2026-11-06. This layer is a candidate
geometry product derived from the georeferenced Figure 14 raster. It does not
replace the published figure, the anonymous Selley compilation, or an
author-supplied GIS layer.

The separate evidence-based naming and parent-axis reconstruction is documented
in `docs/geology_foundation/major_structures_v2.md`. This file remains the frozen
v1 extraction record and does not describe the v2 attribution.

## What was extracted

Figure 12 of Daly et al. (2025) states: "Emboldened are the major fault zones of
the basin." Figure 14 is based on Figures 12-13 and retains the emboldened
linework. The extraction therefore targets the thick black ribbons in Figure 14
and excludes the thin bedding, form and cartographic lines.

The resulting layer contains 248 line segments in 173 connected networks,
totalling approximately 7,059 km of extracted centreline. The segment lengths
range from 2.3 km to 216.2 km. Of these, 67 are long ribbons, 60 are medium
ribbons and 121 are short segments, mostly individual dashes from dashed fault
traces. Dashes are retained as separate segments; they are not bridged across
the printed gaps.

No fault name, kinematic class, displacement, age or mineralisation association
has been assigned. The 15 names published in the Figure 12 caption remain in
`named_structure_reference.csv` without a geometry link. This avoids naming a
segment by visual proximity alone.

## Method

1. Read the georeferenced RGB raster and retain dark pixels with
   `max(R,G,B) < 100`.
2. Compute the Euclidean distance transform inside the dark mask. Retain pixels
   with a distance of at least 2.2 pixels as the core of a thick ribbon. This
   corresponds to an approximately 4.4-pixel stroke and rejects the 1-2 pixel
   form lines.
3. Mask both map legends and the seven internal domain labels using fixed
   pixel exclusion rectangles. The rectangles are recorded in the build
   configuration.
4. Remove components smaller than 8 pixels, thin the remaining core with
   Zhang-Suen skeletonisation, and trace the one-pixel skeleton into graph
   paths. Diagonal shortcuts through filled raster corners are suppressed so
   that a raster staircase does not create thousands of false junctions.
5. Drop paths shorter than 4 pixels. Simplify retained paths in source-pixel
   coordinates at 0.8-pixel tolerance and transform every vertex from the
   Figure 14 EPSG:4326 geotransform to EPSG:32735.
6. Attach segment identity, network identity, source-pixel length, estimated
   stroke width and extraction confidence. The confidence field describes
   cartographic detection; it is not a geological probability.

The centred paths are approximately one source pixel (about 0.6 km) thin. They
should not be interpreted as the full width of a mapped fault zone.

## Outputs

Canonical data:

- `data/processed/geology_foundation/daly_major_structures_v1.gpkg`
  (layer `daly_major_structures`)
- `data/processed/geology_foundation/daly_major_structures_attributes_v1.csv`
- `data/processed/geology_foundation/daly_major_structures_summary_v1.json`
- `data/processed/geology_foundation/daly_major_structures_manifest_v1.json`

Visual products:

- `outputs/geology_foundation/daly_major_structures_overlay.png`
- `outputs/geology_foundation/daly_major_structures_overlay_detail.png`
- `outputs/geology_foundation/daly_major_structures_segments.png`
- `outputs/geology_foundation/daly_major_structures_map.png`

The first overlay places every extracted centreline in red on the published
figure. The segment overlay alternates colours to expose separate paths. The
clean map shows the tectonic-domain reconstruction and the extracted linework
without the source image.

## Attribute dictionary

| Field | Meaning |
|---|---|
| `structure_id` | Stable segment identifier assigned by this build |
| `network_id` | Connected component of the extracted skeleton |
| `feature_class` | `major_fault_zone_candidate` for every segment |
| `line_class` | Long ribbon, medium ribbon or short fragment/dash |
| `structure_name` | Deliberately null until an evidence-based name link is made |
| `structure_type` | Deliberately null; Figure 14 does not encode kinematics per segment |
| `figure_class` | `major_fault_zone`, following the Figure 12 caption |
| `tier` | Deliberately null; no structural hierarchy has been assigned |
| `hierarchy_status` | `not assigned` |
| `source_figure` | `Daly et al. (2025) Figure 14` |
| `source_doi` | `10.1029/2025TC008955` |
| `extraction_method` | Algorithm summary |
| `source_pixel_length` | Length in source-image pixels |
| `median_stroke_width_px` | Median estimated full stroke width along the path |
| `mean_stroke_width_px` | Mean estimated full stroke width along the path |
| `length_km` | Projected centreline length in EPSG:32735 |
| `extraction_confidence` | High/medium/low cartographic detection confidence |
| `name_status` | `unassigned` |
| `attribution_status` | Caption-supported figure linework; name and role not assigned |
| `review_status` | `automated_candidate` |
| `notes` | Source statement and method note |

## Reproduction and verification

Use the QGIS Python environment. The extraction is deterministic and does not
fit a model.

```powershell
& 'C:\Program Files\QGIS 3.40.15\bin\python-qgis-ltr.bat' -m py_compile src\geology_foundation\major_structures.py src\geology_foundation\verify_major_structures.py

& 'C:\Program Files\QGIS 3.40.15\bin\python-qgis-ltr.bat' -m src.geology_foundation.major_structures --config configs/major_structures.json --overwrite

& 'C:\Program Files\QGIS 3.40.15\bin\python-qgis-ltr.bat' -m src.geology_foundation.verify_major_structures --manifest data\processed\geology_foundation\daly_major_structures_manifest_v1.json
```

The verifier checks source and output hashes, the layer schema, CRS, geometry
validity, source extent, null name/type/hierarchy fields, the frozen V14 input
hash and the presence of all four visual products.

## Interpretation boundary

This is a faithful extraction of the heavy linework printed in Figure 14. It is
not an independent field compilation, and it is not author-supplied vector data.
The linework is suitable for a major-structure proximity predictor and for a
major-structure density or intersection feature after the segment geometry has
been reviewed. It is not yet a named-fault catalogue and does not encode fault
kinematics or displacement.
