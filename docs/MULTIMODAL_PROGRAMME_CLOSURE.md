# Closure memo: the heterogeneous multimodal-transfer programme

**Date:** 20 November 2026
**Decision:** **Shelved.** No further work on this programme until the reopening condition in Section 4 is met.
**Scope:** This memo covers the programme recorded in
`docs/PROJECT_PROGRESS_RESEARCH_PHASE_V1.md` (18 Nov) and
`docs/PROJECT_PROGRESS_STAGE_3B.md` (19 Nov). Those records and all of
their outputs under `data/raw/rdc_rocks/derived/research_phase_v1/` are
kept unchanged as the historical record.

## 1. Why the programme is shelved

The programme asked whether chemistry, geophysics, structure and
occurrence evidence could be learned jointly and transferred between
regions A and B. Stage 3B answered that question with the evidence that
exists. The answer is negative:

1. **There is only one chemical dataset.** The 568 "geochemical" values and
   the 568 "drilling assay" values are the same 71 intervals × 8 analytes
   from six holes in a single source (`RDC_4374`).
2. **No geophysics can be located.** Neither `RDC_0205` (magnetics) nor
   `RDC_0208` (Bouguer) has a graticule, ticks, corner coordinates,
   projection or datum. No control points can be derived without inventing
   them.
3. **Sample chemistry has no coordinates.** 90 samples and 2,199
   determinations now have locality records, but none has coordinates.
4. **No structure is located at regional scale.** All recovered
   structural observations are mine-scale. The 62 station coordinates have
   no established grid origin.
5. **The occurrence evidence is weak.** 147 RDC observations include zero
   at recovery class L1 (directly locatable), and zero identities are
   confirmed.
6. **Regions A and B share only mapped geology and occurrence labels.**
   Chemistry and geophysics transfer cannot be identified (Stage 3B gates G4
   not met; G5, G6 and G9 only partly met).

Every remaining P0 item depends on sources the project does not hold:
datum statements, control points, sample-to-station keys and original map
sheets. That work has no foreseeable end, and it cannot change the answer
until at least one new located modality exists.

## 2. What is kept

- All research-phase and Stage 3B reports, registers, tools and images are
  kept as they are. They document a valid negative result.
- The evidence registers stay available as inputs to other workstreams:
  - the occurrence-recovery classes (L1–L5);
  - the gazetteer review;
  - the `RDC_4374` collar CRS register (zone 35S, datum provisional);
  - the drilling-register repair specification (not applied).
- The E1/E2 mapping-plus-occurrence protocol is kept as a design document.
  If a later paper uses mapping and occurrence evidence alone, it can draw
  on that protocol. It must not be described as multimodal transfer.

## 3. What stops

- Stops: coordinate, datum, control-point and gazetteer recovery done for
  this programme.
- Stops: the E3/E4 transfer experiments and every chemistry, geophysics or
  full multimodal transfer experiment (E5).
- Stops: new progress records for this programme.

None of this removes or alters existing outputs. The frozen V14 benchmark,
Stages 3–5, Phase 6/7 and the atlas are unaffected.

## 4. Condition for reopening

Reopen only if a **second located non-occurrence modality** becomes
available in **at least two regions**. Examples are a georeferenced
geophysical grid with documented projection, or sample chemistry with
coordinates. The new source must come from authorised acquisition. When
that condition is met, start from the Stage 3B gate table rather than
repeating the audit.

## 5. Project focus after this decision

1. **Paper 1 (V14):** finish and submit. See `paper_1/benchmark/V14_BENCHMARK.md`.
2. **Paper 2:** to be chosen separately from evidence that is already
   verified. The candidates are the 2,193-hex structural predictors with
   Stage 5D observation-process scenarios, or the E1/E2
   mapping-plus-occurrence pilot.
