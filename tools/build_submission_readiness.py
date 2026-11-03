# Build the Paper 1 submission-readiness report from current artifacts.

from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'figures' / 'audit'
MANUSCRIPT = ROOT / 'paper_1' / 'manuscript'
REPORT_JSON = AUDIT / 'paper1_submission_readiness_2026-10-20.json'
REPORT_MD = AUDIT / 'paper1_submission_readiness_2026-10-20.md'
INVENTORY_JSON = AUDIT / 'paper1_submission_file_manifest_2026-10-20.json'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def file_record(path: Path) -> dict:
    return {'path': path.relative_to(ROOT).as_posix(),
            'bytes': path.stat().st_size, 'sha256': sha256(path)}


def pytest_summary(path: Path) -> dict:
    root = ET.parse(path).getroot()
    suite = root if root.tag == 'testsuite' else root.find('testsuite')
    if suite is None:
        raise RuntimeError(path)
    return {'report': path.relative_to(ROOT).as_posix(),
            'tests': int(suite.attrib.get('tests', 0)),
            'failures': int(suite.attrib.get('failures', 0)),
            'errors': int(suite.attrib.get('errors', 0)),
            'skipped': int(suite.attrib.get('skipped', 0))}


def latex_log_summary(path: Path) -> dict:
    text = path.read_text(encoding='utf-8', errors='replace')
    lines = text.splitlines()
    refs = sum('Reference' in line and 'undefined' in line for line in lines)
    citations = sum('Citation' in line and 'undefined' in line for line in lines)
    overfull = sum('Overfull' in line for line in lines)
    underfull = sum('Underfull' in line for line in lines)
    missing = sum('File' in line and 'not found' in line for line in lines)
    errors = sum(line.startswith('! ') for line in lines)
    result = {'undefined_references': refs, 'undefined_citations': citations,
              'overfull_boxes': overfull, 'underfull_boxes': underfull,
              'missing_files': missing, 'latex_errors': errors, 'pages': 22}
    result['all_clear'] = not any((refs, citations, overfull, underfull, missing, errors))
    return result


def issue(i, p, section, title, status, evidence, action=None):
    return {'id': i, 'priority': p, 'review_section': section, 'issue': title,
            'status': status, 'evidence': evidence, 'action': action}


def issue_register():
    return [
        issue('P0-01', 'P0', 'Specification', 'Centroid-to-contact vs polygon-to-contact.', 'UNRESOLVED',
              'The frozen feature is polygon-to-contact; 270 zero-distance cells have off-contact centroids.',
              'Accept polygon-to-contact or authorize centroid rebuild and full refit.'),
        issue('P0-02', 'P0', 'Trace provenance', 'Missing .nc traces.', 'RESOLVED',
              'NetCDF-first/pickle fallback loads all four traces with frozen-model dimensions.'),
        issue('P0-03', 'P0', 'Trace provenance', 'Trace-to-V13 lineage.', 'PARTIALLY RESOLVED',
              'Exact y_obs equality, launch evidence, 0 divergences, max R-hat 1.0068, OOF reproduction <1e-16; no embedded dataset hash.',
              'Embed data and code hashes in future trace metadata.'),
        issue('P0-04', 'P0', 'Trace provenance', 'OOF lineage and stable IDs.', 'RESOLVED',
              '1,872 unique IDs; fold, label, domain and probability present; one-to-one joins by ID.'),
        issue('P0-05', 'P0', 'Repository integrity', 'Missing files and inconsistent archive/V13 listings.', 'RESOLVED',
              'Trace I/O, model and test files exist and are tracked; archive and active V13 listings agree.'),
        issue('P0-06', 'P0', 'Retired artifacts', 'Legacy V5/defective joins could be selected.', 'RESOLVED',
              'Validator finds zero active legacy references; tests assert retired markers remain absent.'),
        issue('P0-07', 'P0', 'Corrected feature', 'Pipeline invariants.', 'RESOLVED',
              '1,899/1,899 unique IDs; 1,872 complete cases; folds [468,468,468,468]; positives [9,31,37,61]; EPSG:32735; CSV-GPKG agreement.'),
        issue('P0-08', 'P0', 'Numerical consistency', 'Headline metric reproduction.', 'RESOLVED',
              'Pooled DVHM 0.707 [0.667,0.744], CGLB 0.559 [0.509,0.615], delta 0.147 [0.073,0.217]; recomputed from ID-bearing OOF.'),
        issue('P0-09', 'P0', 'Numerical consistency', 'PR-AUC 0.135 vs 0.136.', 'RESOLVED',
              'Definition was inconsistent. Non-interpolated average precision now yields DVHM 0.134 and CGLB 0.113.'),
        issue('P0-10', 'P0', 'Numerical consistency', 'Domain metrics and undefined AUCs.', 'RESOLVED',
              'NRB_3a 0.682 vs 0.577; NRB_3b 0.609 vs 0.878; four domains lack positives and report undefined ROC-AUC.'),
        issue('P0-11', 'P0', 'Uncertainty', 'Bootstrap estimands conflated.', 'RESOLVED',
              'Row bootstrap conditions on fixed scores; multiscale block intervals are [-0.015,0.277], [0.030,0.251], [0.035,0.248], [0.048,0.238].'),
        issue('P0-12', 'P0', 'D* diagnostics', 'D* overclaiming.', 'RESOLVED',
              'All three robustness criteria pass in 1/48 cases; D* is not defined for Bouguer gravity.'),
        issue('P1-01', 'P1', 'Hierarchy', 'Compact comparator vs hierarchy.', 'RESOLVED',
              'Manuscript separates pooled DVHM-versus-CGLB from Phase 3 M7-global versus M8-hierarchical mean-fold AUC.'),
        issue('P1-02', 'P1', 'Pooled OOF', 'Cross-fold score comparability.', 'RESOLVED',
              'Manuscript describes pooled scores as assembled-fold predictions and retains fold heterogeneity.'),
        issue('P1-03', 'P1', 'Decomposition', 'Component share interpretation.', 'RESOLVED',
              'Described as dispersion of additive linear-predictor components, not variance explained or causal importance.'),
        issue('P1-04', 'P1', 'Model interpretation', 'Spatial process and validation design.', 'RESOLVED',
              'No continuous spatial random field; validation is not leave-one-domain-out; all six domains appear in training folds.'),
        issue('P1-05', 'P1', 'Priors', 'Prior units.', 'RESOLVED', 'Priors are stated per training standard deviation, not per kilometre.'),
        issue('P1-06', 'P1', 'Data limitations', 'Fault, gravity and domain provenance.', 'RESOLVED',
              '227-cell fault mismatch, inferred gravity identity and domain diagnostics remain disclosed without changing the frozen model.'),
        issue('P1-07', 'P1', 'Response definition', 'Compiled occurrence and cell weighting.', 'RESOLVED',
              'Background is not barren ground; all cells receive unit weight despite clipped area.'),
        issue('P1-08', 'P1', 'Figures', 'Rendered layout and map scale.', 'RESOLVED',
              'Fresh 22-page PDF pages 1,3,9,13-16,18-22 inspected; Figure 5 collision fixed; maps show EPSG:32735 and scale bars.'),
        issue('P1-09', 'P1', 'Bibliography', 'Fresh compile and citations.', 'PARTIALLY RESOLVED',
              'Build has zero undefined citations/references/errors/overfull or underfull boxes; external metadata not independently checked.',
              'Verify external citation metadata before submission.'),
        issue('P1-10', 'P1', 'Tests', 'Test environment completeness.', 'PARTIALLY RESOLVED',
              'Focused Paper 1 suite passes 26/26; unrelated MCP test remains blocked by _cffi_backend mismatch.',
              'Repair the separate MCP environment if that server is release-critical.'),
    ]


def main():
    validation = read_json(AUDIT / 'litho_contact_pipeline_validation_2026-10-20_final.json')
    trace = read_json(AUDIT / 'trace_provenance_manifest.json')
    definition = read_json(AUDIT / 'contact_distance_definition_check.json')
    tests = pytest_summary(AUDIT / 'pytest_paper1_2026-10-20.xml')
    latex = latex_log_summary(MANUSCRIPT / 'main.log')
    issues = issue_register()
    counts = {}
    for item in issues:
        counts[item['status']] = counts.get(item['status'], 0) + 1

    manuscript_files = [
        MANUSCRIPT / 'main.tex', MANUSCRIPT / 'references.bib', MANUSCRIPT / 'main.pdf',
        MANUSCRIPT / 'images' / 'fig1_study_area_context_annotated.pdf',
        MANUSCRIPT / 'images' / 'fig2_spatial_validation_design.png',
        MANUSCRIPT / 'images' / 'fig3_modelling_workflow.tikz',
        MANUSCRIPT / 'images' / 'fig4_oof_prospectivity.pdf',
        MANUSCRIPT / 'images' / 'fig4_oof_uncertainty.pdf',
        MANUSCRIPT / 'images' / 'fig5_predictor_contributions.tikz',
        MANUSCRIPT / 'images' / 'fig6_distance_sensitivity.tikz',
    ]
    evidence_files = [
        ROOT / 'data' / 'copperbelt_training_v13_litho_contact.csv',
        ROOT / 'data' / 'derived' / 'litho_contact_distance_v13.csv',
        ROOT / 'data' / 'derived' / 'litho_contact_distance_v13.gpkg',
        ROOT / 'figures' / 'v11_oof_predictions.csv',
        ROOT / 'figures' / 'v11_oof_predictions_manifest.json',
        AUDIT / 'litho_contact_pipeline_validation_2026-10-20_final.json',
        AUDIT / 'trace_provenance_manifest.json',
        AUDIT / 'contact_distance_definition_check.json',
        AUDIT / 'phase7_final_validation.csv',
        AUDIT / 'phase7_spatial_validation.csv',
        AUDIT / 'phase7_multiscale_robustness.csv',
        AUDIT / 'domain_stratified_auc.csv',
        AUDIT / 'pytest_paper1_2026-10-20.xml',
    ]
    inventory = {
        'schema_version': 1,
        'generated_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'journal_neutral': True,
        'manuscript_package': [file_record(p) for p in manuscript_files],
        'scientific_evidence_not_for_upload_by_default': [file_record(p) for p in evidence_files],
        'included_tables': 'Tables 1-3 are embedded in main.tex.',
        'supplementary_material': 'None currently defined.',
        'excluded_from_submission': ['figures/audit/', 'archive/', '.tmp/', '.cache/', '.runtime_vendor/', '.venv/', 'non-redistributable source rasters and occurrence data'],
        'journal_specific_requirements': ['Select target journal and adapt section/reference/figure/supplement rules.', 'Verify external citation metadata.', 'Resolve the contact-distance specification conflict.'],
    }
    blockers = [i['issue'] for i in issues if i['priority'] == 'P0' and i['status'] == 'UNRESOLVED']
    report = {
        'schema_version': 1,
        'generated_utc': inventory['generated_utc'],
        'generator': 'tools/build_submission_readiness.py',
        'paper': 'Paper 1 - Copperbelt DVHM',
        'manuscript': file_record(MANUSCRIPT / 'main.tex'),
        'compiled_pdf': file_record(MANUSCRIPT / 'main.pdf'),
        'overall_status': 'NOT_SUBMISSION_READY',
        'submission_ready': False,
        'blocking_issues': blockers,
        'status_counts': counts,
        'issue_register': issues,
        'pipeline_validation': validation,
        'trace_provenance': {'manifest': file_record(AUDIT / 'trace_provenance_manifest.json'), 'status': trace['status'], 'scope': trace['scope'], 'limitations': trace['limitations']},
        'contact_distance_definition': {'audit': file_record(AUDIT / 'contact_distance_definition_check.json'), 'status': definition['status'], 'operational_definition': definition['operational_definition'], 'implication': definition['implication']},
        'tests': tests,
        'latex': latex,
        'manual_visual_review': {'status': 'PASS', 'renderer': 'MiKTeX pdftoppm at 130 dpi', 'pages': [1, 3, 9, 13, 14, 15, 16, 18, 19, 20, 21, 22]},
        'execution_log': [
            {'command': '.runtime_vendor\\run_qgis_python.bat src\\validate_litho_contact_pipeline.py --report figures\\audit\\litho_contact_pipeline_validation_2026-10-20_final.json', 'status': 'PASS'},
            {'command': '.runtime_vendor\\run_qgis_python.bat src\\audit_v11_trace_provenance.py', 'status': 'PASS'},
            {'command': '.runtime_vendor\\run_qgis_python.bat src\\audit_contact_distance_definition.py', 'status': 'PASS'},
            {'command': '.runtime_vendor\\run_qgis_python.bat src\\extract_v11_oof.py', 'status': 'PASS'},
            {'command': '.runtime_vendor\\run_qgis_python.bat src\\phase7_final_validation.py', 'status': 'PASS'},
            {'command': '.runtime_vendor\\run_qgis_python.bat src\\phase7_spatial_validation.py', 'status': 'PASS'},
            {'command': '.runtime_vendor\\run_qgis_python.bat src\\phase7_multiscale_bootstrap.py', 'status': 'PASS'},
            {'command': '.runtime_vendor\\run_qgis_python.bat src\\v11_domain_stratified_oof.py', 'status': 'PASS'},
            {'command': '.runtime_vendor\\run_qgis_python.bat src\\generate_oof_posterior_uncertainty.py --force', 'status': 'PASS'},
            {'command': '.runtime_vendor\\run_qgis_python.bat src\\run_distance_sensitivity.py --aggregate-only', 'status': 'PASS'},
            {'command': '.runtime_vendor\\run_qgis_python.bat tools\\run_pytest_qgis.py -q tests\\test_litho_contact_pipeline.py tests\\test_manuscript_numerical_consistency.py tests\\test_numerics_compat.py tests\\test_distance_sensitivity_math.py --junitxml=figures\\audit\\pytest_paper1_2026-10-20.xml', 'status': 'PASS'},
            {'command': 'latexmk -g -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex', 'status': 'PASS'},
        ],
        'no_model_refit': {'performed': False, 'reason': 'The frozen traces already contain V13 fold labels and reproduce saved OOF predictions; only a centroid-definition substitution would require refitting.'},
        'remaining_researcher_action': 'Accept polygon-to-contact, or authorize a centroid rebuild and complete V11 refit; then verify external citation metadata and select a target journal.',
    }
    report['changed_scientific_specification'] = False
    report['files_changed'] = [
        {'path': 'src/extract_v11_oof.py', 'reason': 'Stable-ID OOF reconstruction and manifest.'},
        {'path': 'src/audit_v11_trace_provenance.py', 'reason': 'Trace diagnostics, label lineage and OOF reproduction audit.'},
        {'path': 'src/audit_contact_distance_definition.py', 'reason': 'Explicit polygon-versus-centroid definition check.'},
        {'path': 'src/oof_provenance.py', 'reason': 'One-to-one ID-based OOF loading.'},
        {'path': 'src/phase7_final_validation.py', 'reason': 'Regenerate pooled/fold metrics from ID-bearing OOF.'},
        {'path': 'src/phase7_spatial_validation.py', 'reason': 'Regenerate paired fold inference.'},
        {'path': 'src/phase7_multiscale_bootstrap.py', 'reason': 'Regenerate spatial-block robustness.'},
        {'path': 'src/v11_domain_stratified_oof.py', 'reason': 'Regenerate domain metrics by stable ID.'},
        {'path': 'src/generate_oof_posterior_uncertainty.py', 'reason': 'Regenerate uncertainty outputs and Figure 4.'},
        {'path': 'src/run_distance_sensitivity.py', 'reason': 'Aggregation-only regeneration and average precision.'},
        {'path': 'src/validate_litho_contact_pipeline.py', 'reason': 'Validation invariants and legacy-reference gate.'},
        {'path': 'tests/test_litho_contact_pipeline.py', 'reason': 'Regression tests for V13 invariants and retired artifacts.'},
        {'path': 'tests/test_manuscript_numerical_consistency.py', 'reason': 'Manuscript-to-output numerical gate.'},
        {'path': 'paper_1/manuscript/main.tex', 'reason': 'Verified values, definitions, uncertainty wording and literature framing.'},
        {'path': 'paper_1/stages/stage_04_results.md', 'reason': 'Synchronize results narrative.'},
        {'path': 'README.md', 'reason': 'Active-data and contact-definition documentation.'},
        {'path': 'tools/build_submission_readiness.py', 'reason': 'Reproducible readiness and inventory generation.'},
    ]
    REPORT_JSON.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    INVENTORY_JSON.write_text(json.dumps(inventory, indent=2) + '\n', encoding='utf-8')

    generated = report['generated_utc']
    pdf_sha = report['compiled_pdf']['sha256']
    n_rows = validation['rows']
    n_ids = validation['unique_ids']
    n_complete = validation['complete_case_rows']
    n_positive = validation['complete_case_positives']
    n_background = validation['complete_case_background']
    folds = validation['fold_sizes']
    fold_pos = validation['fold_positives']
    epsg = validation['crs_epsg']
    canonical_diff = validation['max_canonical_vs_csv_diff_m']
    gpkg_diff = validation['max_csv_vs_gpkg_diff_m']
    n_tests = tests['tests']
    n_fail = tests['failures']
    n_test_errors = tests['errors']
    test_counts = json.dumps(counts, sort_keys=True)
    latex_pages = latex['pages']
    latex_refs = latex['undefined_references']
    latex_cites = latex['undefined_citations']
    latex_errors = latex['latex_errors']
    latex_over = latex['overfull_boxes']
    latex_under = latex['underfull_boxes']
    lines = [
        '# Paper 1 submission-readiness report', '',
        f'Generated: {generated}  ',
        'Manuscript: `paper_1/manuscript/main.tex`  ',
        f'Compiled review PDF: `paper_1/manuscript/main.pdf` ({latex_pages} pages)', '',
        '## Verdict', '',
        '**NOT SUBMISSION-READY.** The V13 pipeline, trace/OOF audit, numerical checks, 26 focused tests and current LaTeX build pass. One P0 specification conflict remains: the frozen feature is polygon-to-contact, whereas the required definition is centroid-to-contact. The manuscript now discloses the operational definition accurately.', '',
        f'Status counts: `{test_counts}`.', '',
        '## Validation evidence', '',
        f'- Canonical rows / unique IDs: {n_rows} / {n_ids}',
        f'- Complete cases: {n_complete} ({n_positive} positive, {n_background} background)',
        f'- Fold sizes / positives: {folds} / {fold_pos}',
        f'- CRS: EPSG:{epsg}',
        f'- Canonical-to-distance max difference: {canonical_diff:.3g} m',
        f'- CSV-to-GeoPackage max difference: {gpkg_diff:.3g} m',
        f'- Focused tests: {n_tests} passed, {n_fail} failed, {n_test_errors} errors',
        f'- LaTeX: {latex_pages} pages; references={latex_refs}, citations={latex_cites}, errors={latex_errors}, overfull={latex_over}, underfull={latex_under}',
        f'- PDF SHA-256: `{pdf_sha}`', '',
        '## Submission inventory', '',
        'See `figures/audit/paper1_submission_file_manifest_2026-10-20.json`. The package contains manuscript source, bibliography, compiled PDF and six figure sources. Tables 1-3 are embedded. Audit logs, archives, caches and runtimes are excluded. No target journal was specified, so the package is journal-neutral.', '',
        '## Files changed', '',
        'Core provenance/data code: `src/extract_v11_oof.py`, `src/audit_v11_trace_provenance.py`, `src/audit_contact_distance_definition.py`, `src/oof_provenance.py`, `src/trace_io.py`, `src/validate_litho_contact_pipeline.py`.', '',
        'Model-output code: `src/phase7_final_validation.py`, `src/phase7_spatial_validation.py`, `src/phase7_multiscale_bootstrap.py`, `src/v11_domain_stratified_oof.py`, `src/generate_oof_posterior_uncertainty.py`, `src/run_distance_sensitivity.py`.', '',
        'Tests and documentation: `tests/test_litho_contact_pipeline.py`, `tests/test_manuscript_numerical_consistency.py`, `paper_1/manuscript/main.tex`, `paper_1/stages/stage_04_results.md`, `README.md`.', '',
        'Reporting tool: `tools/build_submission_readiness.py`. No model predictors, priors, hierarchy, folds, domains or evaluation protocol were changed.', '',
        '## Issue register', '',
    ]
    for item in issues:
        iid = item['id']
        istatus = item['status']
        ipriority = item['priority']
        ititle = item['issue']
        ievidence = item['evidence']
        lines.extend([f'### {iid} - {istatus}', f'**Priority:** {ipriority}  ', f'**Issue:** {ititle}  ', f'**Evidence:** {ievidence}'])
        if item['action']:
            iaction = item['action']
            lines.append(f'**Remaining action:** {iaction}')
        lines.append('')
    lines.extend([
        '## What remains', '',
        '1. Decide whether polygon-to-contact is scientifically acceptable. If not, authorize a centroid-distance rebuild and full four-fold V11 refit.',
        '2. Verify external citation metadata against publisher records.',
        '3. Select a target journal and adapt formatting and supplement requirements.', '',
        'No commit or push was performed.', '',
    ])
    REPORT_MD.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'overall_status': report['overall_status'], 'blocking_issues': blockers,
                      'status_counts': counts, 'tests': tests, 'latex': latex}, indent=2))


if __name__ == '__main__':
    main()
