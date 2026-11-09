# RDC.ROCKS acquisition

This workstream retrieves and verifies source documents from the public University of Lubumbashi digital archive at `https://rdc.rocks/`.

The catalogue API, record metadata and file endpoint are public. The acquisition code sends a standard desktop-browser user agent, makes sequential requests, and does not attempt to bypass access controls or rate limits.

## Source identity warning

The archive's S3 object filenames are not reliable document identifiers. Record `3854`, for example, resolves to an object whose filename describes a Sakania railway study, while the PDF itself is the correct one-page Roquil map "Facies et epaisseurs du Groupe R.4". Record identity must therefore be checked from the document content, and all preserved files must be tied to their catalogue record and checksum rather than to the object filename.

## Commands

```powershell
python src/data_acquisition/rdc_rocks/download.py sanitize-probe `
  --source data/raw/rdc_rocks/metadata/file_probe_v1.csv `
  --output data/raw/rdc_rocks/metadata/file_probe_v2_sanitized.csv

python src/data_acquisition/rdc_rocks/download.py download `
  --manifest data/raw/rdc_rocks/metadata/file_probe_v2_sanitized.csv `
  --output-dir data/raw/rdc_rocks/files/pdf `
  --state data/raw/rdc_rocks/metadata/download_state_pdf_v1.jsonl

python src/data_acquisition/rdc_rocks/download.py verify `
  --manifest data/raw/rdc_rocks/metadata/file_probe_v2_sanitized.csv `
  --output-dir data/raw/rdc_rocks/files/pdf
```

TIFF files use the same commands with `--content-type image/tiff` and `--output-dir data/raw/rdc_rocks/files/tiff`.
