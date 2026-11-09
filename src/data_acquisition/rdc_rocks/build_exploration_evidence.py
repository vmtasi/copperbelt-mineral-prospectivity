from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/raw/rdc_rocks/derived/exploration"

ROWS = [
    ("1408", "p070", "historical_exploration_campaign",
     "Uranium potential was investigated in several periods; 1913-1945 discoveries were made during general copper prospecting, with uranium extracted alone or as a by-product.",
     "Documentary chronology, secondary citation", "Historical", "high"),
    ("1408", "p070", "airborne_radiometric",
     "1955-1958 Mission URAKAT: airborne radiometric prospecting (Fairchild campaign).",
     "Documentary chronology, secondary citation", "Historical, Katanga", "high"),
    ("1408", "p071", "airborne_geophysics",
     "1969-1970 Campagne Hunting: airborne radiometric and magnetic survey covering 14,800 km2, approximately the former Gecamines concession; radiometric and magnetic anomaly maps were compiled.",
     "Documentary chronology, secondary citation", "Historical, former Gecamines concession", "high"),
    ("1408", "p071", "integrated_exploration",
     "1982-1985 Association Uranium Zaire (COGEMA-CGEA-GECAMINES): radiometric and magnetic prospecting, VLF electrical logging, detailed geology and tectonics, and 126 cored boreholes in six uranium sectors between Lubumbashi and Kolwezi.",
     "Documentary chronology, secondary citation", "Historical, Lubumbashi-Kolwezi", "high"),
    ("1408", "p025", "drilling",
     "Borehole Lsw-1215 at X 138926.92, Y 221840.09, Z 1335.64; depth 99.4 m; azimuth 0; inclination 90. Local grid, datum not stated.",
     "Direct borehole record transcribed in the thesis", "Luiswishi", "high"),
    ("1408", "p031", "drilling",
     "Borehole Lsw-1216 at X 138726.91, Y 221639.87, Z 1328.51; depth 151 m; azimuth 0; inclination 90. Local grid, datum not stated.",
     "Direct borehole record transcribed in the thesis", "Luiswishi", "high"),
    ("1408", "p075-p078", "borehole_radiometry",
     "Radiometric measurements in counts per minute on samples from both Luiswishi boreholes; downhole profiles plotted against lithology; background 25 CPM.",
     "Direct measurement reported in the thesis", "Luiswishi", "high"),
    ("0100", "p019", "tenure",
     "The 500 km by 50 km Lufilian Arc concession of UMHK/Gecamines was subdivided into three sectors (Sud/Lubumbashi, Centre and Nord).",
     "Documentary statement", "Lufilian Arc", "high"),
    ("0100", "p024", "trenching_and_drilling",
     "The R2, R3 and R4 zones were studied in detail with hundreds of trenches and pits, often duplicated by numerous boreholes; most work was directed by the author with a field team for CHEMAF.",
     "Direct statement by the author", "Kolwezi area (Kabirungu, Kabobo, Kabulungu)", "high"),
    ("0100", "p049", "geological_mapping",
     "CHEMAF concession zones were mapped by traverses; countless trenches, small pits and numerous boreholes were used; the Kabirungu deposit was discovered during this work.",
     "Direct statement by the author", "Kolwezi area", "high"),
    ("0100", "p089", "sampling_and_petrography",
     "Trenches were excavated and logged, formations described and samples collected; about one hundred thin and polished sections were prepared from samples taken during field campaigns.",
     "Direct statement by the author", "Kolwezi area", "high"),
    ("0100", "p003", "whole_rock_geochemistry",
     "TiO2-Al2O3-SiO2 geochemical data were used to interpret a mixed continental and mantle source with felsic, tholeiitic and hydrothermal influences.",
     "Data-based interpretation reported in the thesis", "Kolwezi area", "high"),
    ("0100", "p098", "geochemical_sample_location",
     "A sampled outcrop is located at 10 deg 46 min 27.6 sec S, 25 deg 40 min 29.2 sec E (geographic coordinates as published).",
     "Direct sample record", "Kolwezi area", "high"),
    ("0100", "p036", "drilling",
     "The R2 unit hosts the Cu-Co and U-Ni mineralisation and has been traversed by numerous mining boreholes.",
     "Direct statement by the author", "Kolwezi area", "high"),
    ("1388", "p029", "underground_mapping",
     "Underground geological mapping of the Kamoto Etang Nord deposit (KCC); structural measurements taken in situ with a structural compass (dip and dip direction) and analysed with DIPS.",
     "Direct statement by the author", "Kamoto Etang Nord", "high"),
    ("1388", "p005", "modern_operational_context",
     "Kamoto Copper Company (KCC) provided the field site, supervision and logistical/financial support.",
     "Direct acknowledgement; modern operator context, not historical exploration evidence", "Kamoto", "high"),
    ("0379", "p009", "historical_map_compilation",
     "The 1:100,000 geological map was compiled from the 1:200,000 geological map of Van den Branden (1953) and 1:100,000 topographic maps of the Comite Special du Katanga.",
     "Direct statement in the explanatory notice", "Mitwaba degree square", "high"),
    ("3461", "p024", "ground_radiometric_survey",
     "A detailed radiometric survey was carried out over several days in the main deposits with a portable Scintrex G15-5 digital scintillometer (Geiger counter).",
     "Direct statement by the author; thesis-era survey, not a historical campaign", "Katanga meridional, main Cu-Co-U deposits", "high"),
    ("3461", "p028-p031", "ground_radiometric_survey",
     "150 radioactivity measurements in RSF averaged 570 cps (range 400-750); RSC pockets gave 700-900 cps; CMN measured about 300 cps, barely above 350 cps.",
     "Direct measurements reported in the thesis", "Lwishia/Luiswishi sector", "high"),
    ("3461", "p025", "geochemical_assay",
     "Seven RAT lilas samples gave 3.8-4.3 ppm U (mean 4.1), 16-21 ppm Th and near-zero K.",
     "Direct assay values reported in the thesis", "Lwishia/Luiswishi sector", "high"),
    ("ALL", "-", "absence",
     "No standalone geophysical survey report, raw survey data, assay database, drill log archive or annual exploration report is present among the 118 acquired documents. Exploration evidence is limited to thesis chronologies, borehole descriptions and modern operator acknowledgements.",
     "Our assessment of the acquired corpus", "RDC.ROCKS acquisition set", "high"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "exploration_evidence_v1.csv"
    fields = ["evidence_id", "record_id", "source_page", "evidence_type",
              "direct_observation", "observation_or_inference", "geographic_scope", "confidence"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for i, row in enumerate(ROWS, 1):
            writer.writerow([f"EXP-{i:03d}", *row])
    print(f"{len(ROWS)} evidence rows -> {path}")


if __name__ == "__main__":
    main()
