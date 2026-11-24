"""Build the EarthArXiv preprint of Paper 1.

Compiles ``paper_1/manuscript/preprint.tex`` (the journal manuscript plus a
"non-peer-reviewed preprint" note on page 1) and ``supplementary.tex``, joins
them into one PDF, and writes the text for the EarthArXiv submission form.

    python tools/build_preprint.py            # final build; needs the Zenodo DOI
    python tools/build_preprint.py --draft    # allow "DOI to be added"

Outputs in ``paper_1/preprint/``:
    Tasi_rankings_transfer_preprint.pdf   manuscript + supplementary material
    eartharxiv_submission.md              title, abstract, keywords, licence, links
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "paper_1" / "manuscript"
OUT = ROOT / "paper_1" / "preprint"
PDF_NAME = "Tasi_rankings_transfer_preprint.pdf"
REPOSITORY = "https://github.com/vmtasi/copperbelt-mineral-prospectivity"


def compile_tex(name: str) -> Path:
    result = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", name],
        cwd=MANUSCRIPT, capture_output=True, text=True,
    )
    log = (MANUSCRIPT / name).with_suffix(".log").read_text(encoding="latin-1")
    problems = re.findall(r"^!.*|LaTeX Warning: (?:Reference|Citation).*undefined.*", log, re.M)
    if result.returncode != 0 or problems:
        raise SystemExit(f"{name} did not compile cleanly:\n" + "\n".join(problems[:10]))
    return (MANUSCRIPT / name).with_suffix(".pdf")


def latex_to_text(text: str) -> str:
    """Plain text for the web form (handles the constructs used in the abstract)."""
    replacements = [(r"\,", " "), ("---", "—"), ("--", "–"), (r"\%", "%"), ("~", " ")]
    for old, new in replacements:
        text = text.replace(old, new)
    text = re.sub(r"\$([^$]*)\$", r"\1", text)
    text = re.sub(r"\\emph\{([^}]*)\}", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def braced_argument(tex: str, command: str) -> str:
    """Contents of the first ``\\command{...}``, matching nested braces."""
    start = tex.index(command + "{") + len(command) + 1
    depth = 1
    for position in range(start, len(tex)):
        if tex[position] == "{" and tex[position - 1] != "\\":
            depth += 1
        elif tex[position] == "}" and tex[position - 1] != "\\":
            depth -= 1
            if depth == 0:
                return tex[start:position]
    raise ValueError(f"Unbalanced braces after {command}")


def form_text(tex: str, doi: str | None) -> str:
    title = re.search(r"\\title\[[^\]]*\]\{(.*?)\}\n", tex, re.S).group(1)
    abstract = braced_argument(tex, "\\abstract")
    keywords = re.search(r"\\keywords\{(.*?)\}", tex, re.S).group(1)
    data_link = f"https://doi.org/{doi}" if doi else "(Zenodo DOI to be added)"
    return f"""# EarthArXiv submission: Paper 1

Paste these into the EarthArXiv submission form, and upload `{PDF_NAME}`.

**Title**
{latex_to_text(title)}

**Abstract**
{latex_to_text(abstract)}

**Keywords**
{latex_to_text(keywords)}

**Author** (as on the preprint title page)
Van Mutundo Tasi, Independent researcher. Correspondence: vanmutundo@gmail.com
(ORCID 0009-0001-8512-6950, if you choose to add it to your EarthArXiv profile)

**Licence**
CC BY 4.0

**Subject areas (suggested; pick the closest options the form offers)**
Earth Sciences: Geology (Economic Geology); Earth Sciences: Geophysics and
Seismology; Statistics and Probability (Applied Statistics)

**Links to supplementary data and code**
- Code and outputs: {REPOSITORY}
- Posterior samples (Zenodo): {data_link}

**Comments / journal status**
Submitted to Natural Resources Research. Supplementary material is appended
to the PDF.

**Later steps (Springer Nature preprint policy)**
- When you submit to the journal: give the EarthArXiv DOI and its licence
  (CC BY 4.0) in the submission form.
- After the article is published: add the journal DOI to the preprint record.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draft", action="store_true",
                        help="build even though the Zenodo DOI is not yet in the manuscript")
    args = parser.parse_args()

    tex = (MANUSCRIPT / "main.tex").read_text(encoding="utf-8")
    doi_match = re.search(r"10\.5281/zenodo\.\d+", tex)
    if "DOI to be added" in tex and not args.draft:
        raise SystemExit("The manuscript still says 'DOI to be added'. Add the Zenodo DOI first, "
                         "or pass --draft for a trial build.")
    if shutil.which("pdfunite") is None:
        raise SystemExit("pdfunite (poppler) is required to join the PDFs.")

    preprint = compile_tex("preprint.tex")
    supplement = compile_tex("supplementary.tex")
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / PDF_NAME
    # pdfunite prints harmless "recursive dicts" warnings; keep the output quiet.
    subprocess.run(["pdfunite", str(preprint), str(supplement), str(target)], check=True,
                   capture_output=True)
    (OUT / "eartharxiv_submission.md").write_text(
        form_text(tex, doi_match.group(0) if doi_match else None), encoding="utf-8", newline="\n"
    )
    pages = subprocess.run(["pdfinfo", str(target)], capture_output=True, text=True).stdout
    count = re.search(r"Pages:\s+(\d+)", pages)
    print(f"{target} ({count.group(1) if count else '?'} pages)")
    print(OUT / "eartharxiv_submission.md")
    if args.draft:
        print("DRAFT build: the Zenodo DOI is still missing. Do not post this version.")


if __name__ == "__main__":
    main()
