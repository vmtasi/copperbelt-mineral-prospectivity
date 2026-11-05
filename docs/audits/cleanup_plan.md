# Controlled Repository Cleanup Plan

**Audit date:** 2026-11-05<br>
**Status:** proposed only. No file has been moved, deleted, renamed, regenerated or committed.<br>
**Scope:** repository hygiene, duplicate reduction, historical archiving and environment cleanup. Scientific reporting corrections and the model redesign are separate review units.

---

## 1. Non-negotiable preservation rules

1. V14 is a frozen historical benchmark. Its training table, traces, OOF predictions, fold metrics, validation summaries and manifests are never edited, overwritten or deleted during cleanup.
2. The six protected uncommitted files are not touched, staged, formatted, rebuilt or reverted:

```text
paper_1/copperbelt_exploration_figure1_recovered.qgz
paper_1/manuscript/images/fig1_study_area_context_annotated.pdf
paper_1/manuscript/images/fig1_study_area_context_annotated.png
paper_1/manuscript/main.pdf
paper_1/manuscript/main.tex
tests/test_manuscript_numerical_consistency.py
```

3. `archive/litho_contact_repair_2026-10-18/` and `archive/centroid_contact_repair_2026-10-20/` are retained in place. They contain the surviving V5 table and V11 evidence.
4. A file that is old is not automatically obsolete. A file that has no current reference is not automatically safe to delete. Each removal requires an inventory entry and a recovery route.
5. Recursive deletion is never performed on a computed or wildcard path without resolving and checking the absolute target first.
6. No Git history rewrite, `git reset`, `git clean`, `git prune` or branch deletion is part of this plan.
7. No cleanup commit mixes manuscript content, model results and file hygiene. Reporting corrections receive their own review.
8. `docs/audits/cleanup_verification.md` must not be created until a cleanup phase is actually executed and verified.

---

## 2. Current repository state relevant to cleanup

| Item | Current value | Implication |
|---|---:|---|
| Branch / HEAD | `main` / `2be0dbb` | one commit ahead of origin |
| Tracked files | 3,426 | 3,315 are under `node_modules/` |
| Tracked Node files | 180.91 MB | accidental dependency tree; no `package.json` |
| `.venv` | 825.87 MB / 29,212 files | broken interpreter path |
| `.venv-1` | 11.54 MB | ignored duplicate environment |
| `.venv312` | 5.10 MB | purpose unresolved |
| `.runtime_vendor` | 26.82 MB | vendored runtime; purpose unresolved |
| `.runtime_cache` | 16.77 MB | PyTensor cache |
| `pytensor_cache` | 8.62 MB | PyTensor cache |
| `.trash` | 30.13 MB / 974 files | local trash and compiled-cache leaks |
| `.mcp` | 57.08 MB total; 6 tracked source/metadata files | operational MCP/tunnel assets, not research evidence |
| `literature` | 148.08 MB | ignored reference library |
| `figures` | 118.37 MB | contains historical-essential outputs |
| `archive` | 108.26 MB | contains historical-essential snapshots |
| `.git` | 484.92 MB | contains history; not reduced by simple untracking |

Cleanup is not equivalent to shrinking these totals. Scientific evidence comes before size reduction.

---

## 3. Classification rules

| Class | Rule | Recovery |
|---|---|---|
| **KEEP** | Active source, canonical evidence, protected manuscript files or required history | no action |
| **ARCHIVE** | No longer active but potentially needed to explain provenance, previous results or manuscript evolution | move to a dated archive location; retain checksum |
| **REMOVE** | Reproducible cache, build artefact, duplicate confirmed by SHA-256, or demonstrably accidental dependency | quarantine first, then delete only after approval |
| **REVIEW** | Purpose or dependency is unresolved | no move until owner confirms |

The default for unresolved items is **REVIEW**, never deletion.

---

## 4. KEEP: canonical and active assets

| Path or group | Reason | Constraint |
|---|---|---|
| `data/copperbelt_training_v14_centroid_contact.csv` | V14 modelling table | ignored; must be in benchmark bundle |
| `data/derived/canonical_training_manifest_v14_centroid_contact.json` | canonical hash and configuration record | tracked; do not edit |
| `data/derived/litho_contact_centroid_distance_v14.csv/.gpkg` | V14 distance reconstruction | do not regenerate |
| `figures/v14_centroid_contact/` | traces, metrics, OOF predictions | historical-essential |
| `figures/audit/*_v14_centroid*` | phase outputs and validation summaries | keep V14-suffixed files; do not substitute V13 |
| `figures/audit/phase4_domain_heterogeneity_trace_v14_centroid.nc` | Phase-4B descriptive model | label as Phase 4B, not V14 DVHM |
| `figures/audit/*v14_model_output_validation*` | post-refit validation evidence | keep |
| `archive/litho_contact_repair_2026-10-18/` | V5 root table and contact provenance | keep in place |
| `archive/centroid_contact_repair_2026-10-20/` | V11 pickles, V13 snapshot and rollback data | keep in place |
| `archive/paper1_provenance_2026-10-20/` | manuscript provenance | keep |
| `src` active V14 chain and support modules | reproduces current benchmark | keep; old scripts archived separately |
| `tests/` | numerical and pipeline checks | keep |
| `paper_1/manuscript/main.tex`, `main.pdf`, `references.bib` | current manuscript package | protected; separate review |
| `paper_1/manuscript/images/fig1_study_area_context_annotated.*` | current Figure 1 | protected |
| `paper_1/copperbelt_exploration_figure1_recovered.qgz` | Figure 1 source | protected |
| `docs/audits/` and `docs/limitation_register.md` | current audit record | retain; review before tracking |
| `outputs/` | empty target for future generated runs | keep |

V14 artefacts are not made smaller by deleting summaries: the traces are the expensive primary evidence and the CSVs are small.

---

## 5. ARCHIVE: historical and superseded material

### 5.1 Manuscript and figure history

| Item | Evidence | Proposed archive action |
|---|---|---|
| `paper_1/manuscript/main.tex.20261029-audit.bak` | tracked pre-caption edit backup | move to `archive/manuscript/` after checksum; keep one copy |
| `daly_draft_layout_validation.png` and `paper_1/manuscript/images/fig1_study_area_context.png` | exact duplicate, SHA `59709D3E...`, 455,557 B | keep one representative; archive outside active figure set |
| `final_single_boundary_layout_validation.png`, `tract_focused_closure_QA.png` and `paper_1/manuscript/images/fig1_study_area_context_current.png` | exact duplicate group, SHA `688CC163...`, 735,615 B | keep one representative; archive duplicate copies |
| Unused manuscript images `fig1_spatial_folds.png`, `fig2_prediction_map.png`, `fig3_predictor_contributions.png`, `fig4_response_fault.png`, `fig5_response_lithology.png` | not referenced by the current manuscript | archive as figure-history material; verify references first |
| Remaining root QA PNGs | layout/QA outputs, not active manuscript figures | archive to a single QA-history folder; remove from Git after review |

Archive the smallest defensible set. There is no scientific need to keep three byte-identical copies.

### 5.2 Audit outputs

| Item | Evidence | Proposed action |
|---|---|---|
| Ten non-suffixed CSVs byte-identical to V14-suffixed counterparts | hashes enumerated in `obsolete_assets_report.md` | archive the non-suffixed copies; retain V14-suffixed canonical set |
| Seventeen V13/V14 pairs with different content | not duplicates; V13 values differ | retain as labelled history, preferably under a dedicated `archive/v13_audit/` area rather than mixed beside V14 |
| `v14_centroid_contact_failed_run_manifest_20261020T204228.json` | failed pre-refit run | archive with an explicit "failed run" label |
| `figures/audit - Shortcut.lnk` | Windows shortcut, no scientific content | quarantine and remove after no-dependency check |

Do not collapse the V13 and V14 directories by filename. The file names are parallel; the content is not.

### 5.3 Superseded code

The following should be archived only after the active dependency map is rechecked:

```text
src/ablation_suite.py
src/compare_validation_strategies.py
src/compute_fold_counts.py
src/feature_x_dom_x_target.py
src/hierachical_spatial_cv.py
src/v9_posterior_extraction.py
src/v10_spatail_heterogeneity_upgraded.py
src/extract_v11_oof.py
src/audit_v11_trace_provenance.py
src/rebuild_litho_contact_features.py
tools/build_submission_readiness.py
```

`src/v9_posterior_extraction.py` and `src/v10_spatail_heterogeneity_upgraded.py` are historical model implementations. `src/rebuild_litho_contact_features.py` contains a placeholder external path and is not the V14 rebuild. `tools/build_submission_readiness.py` hard-codes stale V13/V11 metrics and must not be used as current evidence.

---

## 6. REMOVE AFTER QUARANTINE: operational clutter

### 6.1 `node_modules/`

**Evidence:** 3,315 tracked files, 180.91 MB, no `package.json`, no repository import path.<br>
**Plan:** remove from Git tracking first; then quarantine the disk directory and verify no open process or MCP dependency.<br>
**Do not:** delete it while a tool or server is using it.<br>
**Recovery:** Git can restore tracked files from the current commit; the disk copy remains in quarantine until the approval window closes.

This action stops future accidental commits. It does **not** remove `node_modules` objects already stored in Git history; history size remains approximately 485 MB until a separate history-maintenance decision is approved.

### 6.2 Python and runtime environments

| Path | Evidence | Proposed action |
|---|---|---|
| `.venv/` | broken launcher; 825.87 MB; Python 3.10.11 points to a missing Store installation | quarantine; rebuild only from a declared lock file |
| `.venv-1/` | Python 3.14 msys environment; 11.54 MB; unused | quarantine after checking no process holds it |
| `.venv312/` | 5.10 MB; purpose inferred as QGIS-compatible fragment | REVIEW before any move |
| `.runtime_vendor/` | 26.82 MB vendored runtime | REVIEW; may be needed by prior QGIS/PyMC execution |

### 6.3 Caches and build residue

| Path | Evidence | Proposed action |
|---|---|---|
| `.runtime_cache/` | PyTensor compiled cache, 16.77 MB | remove after no active run; fully regenerable |
| `pytensor_cache/` | compiled cache, 8.62 MB | remove after no active run |
| `.trash/` | 30.13 MB, 974 files; includes `latex-stale` and flattened PyTensor path caches | inventory hashes first; quarantine; no known V14 trace or OOF artefact found by name |
| `__pycache__/`, `src/__pycache__/`, `tests/__pycache__/` | Python bytecode | remove |
| `.pytest_cache/` | test cache | remove |
| `main.aux`, `.bbl`, `.blg`, `.fdb_latexmk`, `.fls`, `.log`, `.out`, `.synctex.gz` | reproducible LaTeX intermediates | remove; preserve `main.pdf` |
| `figures/audit - Shortcut.lnk` | Windows shortcut | remove |

`.trash` is not assumed safe solely because its name says trash. Confirm that no file has a unique V14 hash before deletion.

---

## 7. REVIEW: unresolved ownership or dependency

| Path | Question | Why it is not approved for removal |
|---|---|---|
| `.mcp/` | Is the local MCP/tunnel server still used by the research workflow? | 57.08 MB; tracked `server1.py`, licence and manifest plus ignored binaries; removal could break tooling |
| `server2.py`, `oauth_gateway.py` | Are these part of the MCP service or an unrelated experiment? | root-level operational code; no evidence relationship documented |
| `.runtime_vendor/` | Which scripts import it? | may be required by QGIS or PyMC compatibility path |
| `.venv312/` | Is this a required QGIS environment or a partial duplicate? | purpose not verified |
| `literature/` | Which PDFs are cited and under what licence? | ignored large source library; keep until bibliography and licence audit complete |
| `.git/objects/ca/tmp_obj_iLlger` | Is this a recoverable temporary object or reclaimable junk? | do not delete before `git fsck` and repository backup |
| Root `server2.py` and `oauth_gateway.py` | Are they active service entry points? | unknown external consumers |

---

## 8. Stale documentation: separate content review, not cleanup

`README.md`, stages 03-06 and `main.tex.20261029-audit.bak` are not ordinary cleanup targets. Their treatment depends on whether the project intends to keep stage narratives as provenance.

Recommended separate review:

1. Add a superseded banner to stages 03-06 and link to the current V14 benchmark record, **or** archive them as V11/V13 history.
2. Rewrite `README.md` only after the benchmark bundle and environment are settled.
3. Do not silently change their numerical values. The reporting corrections in `main.tex` and the stage updates must be reviewable independently.

---

## 9. Deduplication rules

1. A duplicate is proven by byte-identical SHA-256, not by similar names or sizes alone.
2. For every duplicate group, retain one copy and record which file was retained.
3. Never remove both members of a pair in the same operation.
4. Do not deduplicate the V13/V14 pairs merely because file stems match.
5. Do not deduplicate folders containing ignored V14 outputs before the benchmark bundle is verified.

The exact groups and hashes are recorded in `docs/audits/obsolete_assets_report.md`.

---

## 10. Execution stages and approval gates

### Gate C0: review

**Action:** accept or amend this plan and `docs/limitation_register.md`.<br>
**No file operation.**<br>
**Exit criterion:** written approval of the cleanup scope.

### Gate C1: preserve before changing

**Action:** create a recoverable snapshot outside the working tree:

- Git bundle of the current branch;
- copy of ignored `data/`, `figures/`, `archive/`, `literature/` where storage permits;
- SHA-256 manifest for every preserved V14 file;
- record of `git status --short --branch`.

**Exit criterion:** restore test succeeds for at least the V14 manifest, training table, four traces and OOF predictions.

### Gate C2: untrack accidental files (Git index only)

**Action:** remove `node_modules/` from the Git index and add `node_modules/` to `.gitignore`. Do not delete the disk copy in this gate.<br>
**Exit criterion:** `git ls-files` no longer reports `node_modules`; no source or paper file changed.

### Gate C3: quarantine local caches and old environments

**Action:** move approved items to a dated quarantine directory on the same volume; do not use cross-shell recursive deletion.<br>
**Exit criterion:** path inventory and checksums recorded; no benchmark file is in quarantine.

### Gate C4: archive duplicates and superseded material

**Action:** move approved historical images, backup TeX and stale code into dated archive destinations.<br>
**Exit criterion:** every moved path has a source/destination record; no active manuscript reference points to a moved file.

### Gate C5: remove quarantined material

**Action:** delete only quarantine contents approved after a waiting period.<br>
**Exit criterion:** `git status`, repository inventory and V14 checksums remain unchanged except for intended index changes.

### Gate C6: environment rebuild

**Action:** create a declared, pinned environment and run non-MCMC checks. This is separate from file cleanup.<br>
**Exit criterion:** import checks, data-manifest checks and manuscript numerical-consistency tests pass.

### Gate C7: reporting corrections

**Action:** separately correct domain names, Table 5/Figure 5, Table 2 qualification, stage/README status, supplement, reference and licence.<br>
**Exit criterion:** reviewable diff; no model rerun; manuscript build and tests pass.

### Gate C8: verification

**Action:** verify the repository and preserve the evidence.<br>
**Exit criterion:** all checks in Section 12 pass; only then create `docs/audits/cleanup_verification.md`.

---

## 11. Rollback design

Before Gate C2:

1. `git bundle create <external-path>/copperbelt_pre_cleanup.bundle --all` (approved execution only).
2. Record `git rev-parse HEAD` and `git status --short`.
3. Copy ignored directories to an external or dated archive location.
4. Generate SHA-256 lists for V14 files.

If a cleanup step fails:

1. Stop; do not continue with later gates.
2. Restore the affected path from quarantine first, not from a fresh clone.
3. If Git index state is wrong, restore the index from the bundle/commit without rewriting history.
4. Re-run the V14 hash verification and `git status`.

No cleanup operation should require reconstructing a V14 trace from model fitting.

---

## 12. Verification checklist for C8

The following must all hold after any approved cleanup:

1. All six protected files retain their pre-cleanup hashes and Git status entries.
2. The V14 manifest, training table, four traces, OOF predictions, Phase-7 validation and validation JSON retain their recorded hashes.
3. `git status --short` contains no unexpected deleted or renamed source/paper files.
4. `git ls-files` contains no `node_modules` path.
5. No `.venv`, `.runtime_cache`, `pytensor_cache`, `.trash` or LaTeX intermediate remains in the active tree, unless explicitly retained at REVIEW.
6. Every archive move has a source/destination manifest and a recovery location.
7. The active source chain still matches `docs/audits/script_dependency_map.md`.
8. `main.tex` still references existing figures and supplementary identifiers.
9. V14 numerical checks still reproduce the stored values.
10. The folder itself is not declared clean if a known V14 limitation remains open; cleanup success is not scientific validation.

Deferred or unresolved items remain in the limitation register and are not relabelled as "fixed" because their files were tidied.

---

## 13. Approval bundles

To avoid all-or-nothing approval, cleanup can be authorized in four independent bundles:

| Bundle | Contents | Risk | Scientific effect |
|---|---|---|---|
| **P1: Git hygiene** | untrack `node_modules`; add ignore; no disk deletion | low | none |
| **P2: caches/build residue** | `.venv`, `.venv-1`, caches, LaTeX intermediates, `.trash` after checks | low-medium | none if benchmark preserved |
| **P3: duplicate/archive pass** | root QA images, duplicate figure copies, old backup, superseded code | medium | none if archive manifest retained |
| **P4: environment and reporting** | pinned environment, README/stage corrections, figure/table values, supplement | medium | affects reproducibility and manuscript, not V14 data |

**Recommended first approval:** P1 only. It is reversible, does not touch research data, and immediately stops future accidental Node tracking. **P2** should follow only after the V14 bundle is verified. **P3** should follow a duplicate manifest review. **P4** must not be bundled with P1-P3.

---

## 14. Cleanup verdict

The repository is not ready for wholesale deletion. It is ready for a staged, selective cleanup after a V14 preservation snapshot. The highest-value first action is to untrack `node_modules` without touching research artefacts; the highest scientific priority is not cleanup at all, but preserving the V14 benchmark and carrying the open limitations into the redesign acceptance tests.
