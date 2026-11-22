# V15 lithology provenance audit

## Question

The V14 table contains two different lithology-related quantities: distance to the
nearest internal lithological contact, and a categorical `litho_contact_litho_class`.
The contact-distance feature is retained. This audit asks whether the categorical
field can be independently reconstructed as the host lithology at each cell centroid.

## Inputs

- `data/copperbelt_training_v14_centroid_contact.csv`
- `C:\Project\data for projects\grid_internal_contact_distance_1899.gpkg`
- `C:\Project\data for projects\internal_contacts_only.gpkg`
- `C:\Project\data for projects\lithography_square_93m.tif`

The contacts layer is indexed by cell and contact class. The raster is a surficial
lithology product, not a host-lithology polygon layer. The stored categorical field
was inherited from an unscripted V5 feature build.

## Reproduction

Four candidate rules were tested against the stored field in all 1,899 cells:

| Rule | Agreement |
|---|---:|
| Raster class at the cell centroid | 798 / 1,899 (42.02%) |
| Majority raster class within the clipped cell | 858 / 1,889 (45.42%) |
| Class of the nearest contact to the centroid | 942 / 1,899 (49.61%) |
| Stored class among any tied nearest contacts to the cell polygon | 1,812 / 1,899 (95.42%) |

The stored field is therefore a contact-line attribute selected at the minimum
polygon-to-contact distance. It is neither the centroid raster class nor the
cell-majority class. In 1,890 cells the nearest point is shared by contacts of
different classes, so the source does not uniquely determine which class was stored.
Eighty-seven cells are not matched by the tied-nearest rule; five of them are positive.

## Decision

There is no defensible centroid host-lithology assignment in the acquired source
products. The 42% agreement with the surficial raster is not sufficient to relabel
cells, and no host-lithology polygon layer with a documented centroid rule was found.

Accordingly:

1. The primary M0–M3 ladder retains the V14 categorical field, consistently across all
   four models.
2. `M2_cloglog_no_litho` is a sensitivity model that removes the categorical terms.
3. The raster-centroid variant remains exploratory and must not be promoted to the
   primary specification.

This limitation affects interpretation of the categorical coefficients, not the
validated contact-distance predictor, which is retained throughout.

## Outputs

- `paper_1/v15/diagnostics/lithology_assignment_audit.json`
