import csv, json, math, pathlib, re, sys

ROOT = pathlib.Path(r"C:\Users\vanmu\copperbelt-mineral-prospectivity")
RAW = ROOT / "data" / "raw" / "rdc_rocks"
DER = RAW / "derived"
REG = DER / "register"
GEO = DER / "georef"
REG.mkdir(parents=True, exist_ok=True)
(REG / "rows").mkdir(exist_ok=True)
GEO.mkdir(parents=True, exist_ok=True)

MANIFEST = RAW / "metadata" / "acquisition_manifest_v1.csv"


def load_manifest():
    with MANIFEST.open(encoding="utf-8-sig", newline="") as fh:
        return {row["id"]: row for row in csv.DictReader(fh)}


def lexical_ratio(text):
    toks = re.findall(r"[A-Za-z\u00c0-\u00ff']+", text.lower())
    common = set("de la les des et dans est pour du en un une sur par au aux avec qui que ce cette sont a d l".split())
    return sum(1 for t in toks if t in common) / max(1, len(toks))


# ---------------------------------------------------------------- register rows
DOCS = [
 dict(id="100", tier="1",
  title="Étude comparative des horizons contenant la minéralisation cupro-cobaltifère dans le sous-groupe des Mines (R.2) et le sous-groupe de Dipeta (R.3)",
  date="2025", author="Albert Kalau Kaseke; Université de Lubumbashi, Faculté des Sciences et Technologies, Département de Géologie",
  dtype="Doctoral/advanced-degree thesis", coverage="Kolwezi–Kambove–Luiswishi sector, Lualaba / Haut-Katanga, DRC",
  subject="Comparative lithostratigraphy, petrography and geochemistry of Cu-Co-bearing horizons in R.2 (Mines) versus R.3.2/R.3.4 (Dipeta)",
  units="Roan Supergroup; R.1 (RAT); R.2 Sous-groupe des Mines; R.3 Sous-groupe de Dipeta (R.3.2, R.3.4); Mwashya; Nguba; Kundelungu; RAT grise; pyroclastites",
  structures="Faille de Kansuki (upper and lower lips); polyphase D1–D2 Pan-African deformation; anchizonal metamorphism; Lufilian fold-thrust architecture",
  deposits="Kamoto, Musonoi, Luiswishi, Ruashi, Shinkolobwe, Kipushi, Kambove, Fungurume, Kakanda (as context)",
  scale="n/a (not a map)", crs="n/a (not a map)",
  pages="p.3 (abstract); p.251–252 (general conclusion)",
  direct="Basal R.2 and R.3 horizons show chloritic recrystallisation, detrital input and pyroclastic material; upper horizons are grauwackes, lithic sandstones, arkoses and ferruginous shales. The units are structurally sandwiched along the Kansuki fault.",
  interp="Author interprets mixed continental/mantle provenance (TiO2-Al2O3-SiO2), an active-margin setting for Dipeta and arc/back-arc for Mwashya; redox conditions control mineralisation distribution.",
  conf="high", amb="No spatial coordinates; some stratigraphic attributions depend on the author's sampling; R.2/R.3 age contrast relies on published geochronology."),
 dict(id="379", tier="1",
  title="Carte géologique et notice explicative du degré carré de Mitwaba (Shaba, Zaïre)",
  date="1989", author="Mutapile Kashale; Université de Lubumbashi",
  dtype="Geological map sheet + explanatory notice (licence memoir)",
  coverage="Mitwaba degree square, northern Katanga (Shaba)",
  subject="Regional geology and stratigraphy of the Mitwaba sheet",
  units="Katanguien (Roan, Mwashya, Kundelungu); Kibarien; Ubendien (names legible; detailed subdivisions not recoverable)",
  structures="No structure name reliably legible; text fragments mention the Ubendian chain only. No named fault is entered.",
  deposits="None identifiable in the extracted text",
  scale="1:100,000", crs="No CRS stated; degree-square graticule implied",
  pages="p.1 (title page); p.11–13 (regional geology chapter)",
  direct="The work is the explanatory notice for the 1:100,000 Mitwaba degree square and describes Katangian cover over Ubendian/Kibarian basement.",
  interp="Regional synthesis by the author; not independently checked.",
  conf="low", amb="Typeface OCR is heavily degraded (lexical ratio 0.17); unit and structure names must be verified on page images before citation. The map plate itself was not text-extracted."),
 dict(id="957", tier="1",
  title="Contribution à l'étude pétrographique de la série des Mines du Shaba dans le secteur de Kolwezi: les carrières de Mupine et de Kamoto Principal",
  date="1973", author="Tshileo Mukonga; Université Nationale du Zaïre, Campus de Lubumbashi",
  dtype="Licence thesis (petrography/sedimentology)",
  coverage="Mupine and Kamoto Principal quarries, Kolwezi",
  subject="Petrography and sedimentology of the Série des Mines (R.2) ore horizons",
  units="RAT grise; RAT lilas; D.Strat (dolomies stratifiées); RSF (roche siliceuse feuilletée); RSC; BOMZ; SDS (shales dolomitiques supérieurs); CMN; Série des Mines (R.2)",
  structures="Bedding at Kamoto Principal strikes N50°E and dips 75°NW; no named fault captured",
  deposits="Mupine, Kamoto Principal, Musonoi",
  scale="n/a (not a map)", crs="n/a (not a map)",
  pages="p.25–26 (lithological logs); p.78 (conclusions)",
  direct="Measured logs at Mupine and Kamoto Principal: conglomeratic basal unit, cherty bands, D.Strat and RSF horizons; bedding N50°E/75°NW at Kamoto Principal.",
  interp="Sedimentary-environment reconstruction and correlation between the two quarries.",
  conf="medium", amb="Horizon abbreviations are incompletely defined; no coordinates; scan quality is degraded and some numbers are doubtful."),
 dict(id="1043", tier="1",
  title="Carte géologique du Shaba Méridional",
  date="1936", author="Marcel Gysin",
  dtype="Regional geological map", coverage="Southern Shaba: Lubumbashi–Kipushi–Kambove–Likasi corridor and adjoining Copperbelt",
  subject="Regional bedrock geology and copper-deposit distribution",
  units="Kundelungu (undifferentiated); Mwashya; Roan; Conglomérat de base; Muva(?)/Chambishi(?); quartzites and metamorphic schists; granites and related rocks",
  structures="No named fault reliably legible; Wati and Senda are place names, not structures",
  deposits="Legend symbol 'Gisements de cuivre'; localities Kipushi, Kipapila, Mwati, Senda, Lubembe, N'Changa, Kamikuba, Moka, Mufulira, Lufua, Sakania",
  scale="1:500,000", crs="No CRS or datum stated; no legible graticule; 50 km bar scale",
  pages="p.1 (single sheet)",
  direct="Legend categories, copper-deposit symbol and place labels are directly visible.",
  interp="None (map).",
  conf="medium-low", amb="The legend item 'MUVA O&CHAMBISHI' is garbled; sheet bounds and projection are unknown; several toponyms are uncertain."),
 dict(id="1317", tier="1",
  title="Carte géologique du Shaba Méridional",
  date="1936", author="Marcel Gysin (catalogue variant 'Gyslin')",
  dtype="Regional geological map (second scan)", coverage="Southern Shaba (same sheet as RDC_1043)",
  subject="Regional bedrock geology and copper-deposit distribution",
  units="Kundelungu (undifferentiated); Mwashya; Roan; Conglomérat de base; Muva(?)/Chambishi(?); quartzites and metamorphic schists; granites and related rocks",
  structures="No named fault reliably legible",
  deposits="Copper-deposit symbol; Kipushi, Kipapila, Mwati, Senda, Lubembe, N'Changa, Kamikuba, Moka, Mufulira, Lufua, Sakania",
  scale="1:500,000", crs="No CRS or datum stated; graticule labels fragmentary",
  pages="p.1 (single sheet)",
  direct="Same legend and place labels as RDC_1043; scale 1:500,000 legible.",
  interp="None (map).",
  conf="low-medium", amb="Second scan of the same 1936 map as RDC_1043; use the better scan and treat as one independent source. Legend text partly illegible."),
 dict(id="1177", tier="4",
  title="Contribution à l'étude des occurrences cupro-cobaltifères et uranifères de Kolwezi",
  date="1996", author="Chibal Mukachung, Bally; Université de Lubumbashi",
  dtype="Thesis/report (occurrence study)", coverage="Kolwezi, Shaba",
  subject="Cu-Co and uranium occurrences; stratiform mineralisation and its tectonic setting",
  units="Roan moyen (R.2 / Série des Mines); R.1; dolomitic and shaly ore facies",
  structures="Kolwezi nappe and major thrust sheets; folded, faulted and dislocated Roan moyen; mineralised layers persist through overturned, stacked and thrust sheets",
  deposits="Kamoto, Musonoi, Mupine, Kambove; uranium occurrences of the Kolwezi district",
  scale="n/a (not a map)", crs="n/a (not a map)",
  pages="p.4 (introduction); p.40–41 (discussion and general conclusion)",
  direct="Cu-Co and U mineralisation occupies the same stratigraphic layers across thrust sheets; dolomitic and shaly facies; fluid-inclusion temperature cited as <=250 °C.",
  interp="Author favours a syn-sedimentary/syngenetic model over a single hydrothermal source; 'taches' distribution after François (1987).",
  conf="medium", amb="Typeface OCR errors; author name split across fields; no occurrence coordinates; several numeric citations garbled."),
 dict(id="1231", tier="1",
  title="Contribution à l'étude lithostratigraphique, pétrographique et sédimentologique de la Brèche de Fond de Kamoto Principal",
  date="1994", author="Tsikwej Ntambu-A-N Sang; Madi Lugali; Université de Lubumbashi",
  dtype="Licence thesis (lithostratigraphy/petrography)", coverage="Kamoto Principal, Kolwezi",
  subject="Lithostratigraphy, petrography and sedimentology of the basal breccia at Kamoto Principal",
  units="Brèche de Fond de Kamoto Principal; Série des Mines (R.2) — detailed subdivisions not recoverable",
  structures="Not recoverable", deposits="Kamoto Principal",
  scale="n/a (not a map)", crs="n/a (not a map)",
  pages="p.1 (title); p.5–6 (contents, illegible)",
  direct="The document is a dedicated study of the basal breccia at Kamoto Principal (title and cover).",
  interp="None recoverable from the embedded text.",
  conf="low", amb="Embedded text is corrupt (lexical ratio 0.081); do not cite body content from the text layer. Re-OCR the PDF pages before entering stratigraphic detail."),
 dict(id="1300", tier="1",
  title="Groupe du Katanga — échelle stratigraphique",
  date="unknown", author="unknown",
  dtype="Stratigraphic column/table (catalogue says cross-section)", coverage="Katanga (regional)",
  subject="Stratigraphic nomenclature of the Katanga Group",
  units="Roan and Nguba series; fragments: 'Arkoses des plateaux', 'Grès', 'Mwashya sup.', 'Dipeta sup.'; numbered beds 21–24 and 221–232",
  structures="None", deposits="None",
  scale="Not a map; the fragment '1/20000' appears to be a table value, not a map scale", crs="n/a",
  pages="p.1 (single plate)",
  direct="A tabular stratigraphic column with system/series/stage/substage columns.",
  interp="None.",
  conf="low", amb="Only a small fragment of the table is legible (lexical ratio 0.069); full nomenclature cannot be transcribed."),
 dict(id="3129", tier="2",
  title="Stratigraphie de la Série des Mines — Kakanda-Sud (sondage kda 112) / Kakanda-Nord",
  date="unknown", author="R. Coda; R. Oosterbosch (signature; catalogue OCR 'Dosterbach')",
  dtype="Stratigraphic correlation section", coverage="Kakanda Sud and Nord, Lualaba",
  subject="Correlation of the Série des Mines between Kakanda sections",
  units="Série des Mines (R.2); bed numbers 469, 406, 392, 55(?), 26(?)",
  structures="None", deposits="Kakanda (implicit)",
  scale="1:500 (section scale)", crs="n/a (cross-section, no plan coordinates)",
  pages="p.1 (single section)",
  direct="Graphic correlation of two logs with bed numbers and lithological symbols.",
  interp="Correlation proposed by the authors.",
  conf="low", amb="Most lithological labels are unreadable; borehole identifiers need visual verification."),
 dict(id="3848", tier="1",
  title="Stratigraphie du Groupe du Katanga au Katanga méridional (Cahen 1954)",
  date="1954", author="I. Cals, M. Robert, P. Van Den Brande, A. Hacoupy (compilers); after L. Cahen (1954)",
  dtype="Stratigraphic correlation table", coverage="Elisabethville, Sakabinda, Kambove, Lukaili, Kambove–Bunkeya, Plateau du Kundelungu, Anticlinal de Mwashya",
  subject="Lithostratigraphic correlation of Kundelungu and Roan across southern Katanga",
  units="Kundelungu supérieur/moyen/inférieur; Petit Conglomérat; Grand Conglomérat; Calcaire de Kakontwe; Roan (R.1–R.4); Série des Mines; Dipeta; Mwashya; Conglomérat de Mwashya/Mwanshya; C.M.N.; R.S.C.; R.A.T/D.I.; R.S.E.; R.A.VD.I.",
  structures="Anticlinal de Mwashya; Plateau du Kundelungu; Massif de la Kalwanzo (palaeogeographic high)",
  deposits="None explicitly; mineralised R.2 horizons are implied by the Série des Mines",
  scale="n/a (table)", crs="Geographic coordinates per column: 10°20'S–10°45'S, 26°30'E–27°03'E",
  pages="p.1 (single table)",
  direct="Formation thicknesses (e.g. 1500–1800 m Kundelungu supérieur; 400–500 m; 1200–1500 m), lithologies, marker conglomerates and column coordinates.",
  interp="Regional correlations proposed by Cahen (1954) and the compilers.",
  conf="medium", amb="Many thickness figures are ambiguous (e.g. 'IO à 50 m'); column attribution needs visual checking; nomenclature is pre-modern."),
 dict(id="3854", tier="1",
  title="Facies et épaisseurs du Groupe R.4",
  date="unknown (Figure 10 of an unidentified memoir)", author="unknown",
  dtype="Thematic geological map (facies/isopachs)", coverage="Central Copperbelt arc: Kolwezi–Tenke–Fungurume–Kakanda–Kambove–Shinkolobwe–Likasi",
  subject="R.4 facies distribution and thickness",
  units="R.4 (R.4.1, R.4.2, R.4.2.2); facies Kabul, Chabara, Luankonko; R.4 absent zones; underlying R.3",
  structures="Thrust faults ('faille de charriage'); discordances; alluvial-plain limits; facies boundaries",
  deposits="Tenke, Fungurume, Kakanda, Kambove, Shinkolobwe, Kalabi, Kakonge, Kamonga, Mindigi, Kabolela, Katinda, Likasi, Luishia",
  scale="1:500,000", crs="Metric grid eastings 200,000–340,000 m; latitude labels 25°30'–26°30' (partly legible); datum unspecified",
  pages="p.1 (single map plate)",
  direct="Isopach pairs (e.g. 200/100 m), facies polygons, thrust-fault traces and towns.",
  interp="Facies and thickness interpretation by the original author.",
  conf="medium", amb="Grid zone and datum unknown; some isopach values uncertain; the parent memoir is not identified."),
 dict(id="3856", tier="1",
  title="Roan de Tenke–Fungurume",
  date="unknown (Figure 27 of an unidentified memoir)", author="unknown",
  dtype="Detailed geological map", coverage="Tenke–Fungurume, Lualaba",
  subject="Roan geology of the Tenke–Fungurume area",
  units="Sable; Kundelungu; Dipeta; R.G.S.; C.M.N.; R.S.C.; R.S.F.; Roan undifferentiated",
  structures="Not legible", deposits="Tenke, Fungurume (map area)",
  scale="1:50,000", crs="Metric grid 515,000–535,000 E, 320,000–325,000 N; datum unspecified",
  pages="p.1 (single map plate)",
  direct="Formation polygons and grid coordinates.",
  interp="None recoverable.",
  conf="low", amb="Lexical ratio 0.053; legend abbreviations unresolved; likely a figure extracted from a larger report."),
 dict(id="3863", tier="2",
  title="Arc cuprifère du Katanga — Zone Centrale: structures tectoniques principales",
  date="unknown (Figure 17 of an unidentified memoir)", author="unknown",
  dtype="Structural map", coverage="Central Katanga Copperbelt, Kolwezi–Kambove–Likasi corridor",
  subject="Main tectonic structures of the central copper arc",
  units="R.4/R.3 contact; Kalahari; Kundelungu; Roan (legend partly legible)",
  structures="Thrust faults (faille de charriage), reverse faults (chevauchement), back-thrusts (rétrochevauchement), extrusions, excroissances, anticlinal/synclinal axes with plunge; thrust sheets of Lukonde, Kabwe, Lumbele, Kamwangala, Haute Mofra, Mupapala, Bangwe; Luanso back-thrust; faults at Luankonko and Katinda; Monts Dipompa",
  deposits="Not symbolised",
  scale="1:200,000", crs="Metric grid eastings 200,000–300,000 m, northings 260,000–340,000 m; longitudes 26°–27°E; datum unspecified; raw scan is rotated 180°",
  pages="p.1 (single map plate)",
  direct="Structural traces, kinematics symbols, thrust-sheet names and grid.",
  interp="Thin-skinned thrust-stack interpretation by the original author.",
  conf="medium", amb="Several toponyms are garbled; the R.4/R.3 contact versus fault contact must be checked visually."),
 dict(id="4055", tier="2",
  title="Région de Kolwezi — structures tectoniques principales et typologie",
  date="unknown (plate of an unidentified memoir)", author="unknown",
  dtype="Structural/typological map", coverage="Kolwezi region, Lualaba",
  subject="Structural typology of the Kolwezi nappe/carapace system",
  units="Kalahari; Kundelungu (Ks 1+2+3, Ks II, Ki 1.2+1.3+2, Ki II/Gd Cgt); Roan; poudingue de base; socle Kibarien; brèche",
  structures="Nappe front and root; carapaces 1–7; thrust sheets Tombolo, La Senia, La Gule, Kimuneu, Pungume, Kalukundi Ouest/Est/Signal, Kanunka, Kipwishi, Kapiri, Kaku-O, Mulungwishi, Kamzombo, Lukonde, Kabwe, Tondo, Mongo; anticlines of Manika, Mutoshi, Kazemba, Luankonko, Kisanges; synclines of La Pumpi and Kisanfu; Plateau de Manika",
  deposits="No explicit symbol; Kolwezi deposits implied",
  scale="1:250,000", crs="Metric grid implied (coordinate values only partly legible); datum unspecified",
  pages="p.1 (single map plate)",
  direct="Carapace numbering, thrust-sheet names, structural symbols and stratigraphic legend.",
  interp="Nappe/carapace structural model.",
  conf="medium", amb="Several sheet names are garbled (KAM2LOMBO, KAKU'-O, PUNGUIUME); grid values are not readable."),
 dict(id="1388", tier="2",
  title="Contrôle tectonique de la minéralisation cupro-cobaltifère: cas du gisement de Kamoto Étang",
  date="2022", author="Ilunga Ngoie Casino; Université de Lubumbashi",
  dtype="Bachelor's thesis (structural study)", coverage="Kamoto Étang, Kolwezi",
  subject="Tectonic control of Cu-Co mineralisation at Kamoto Étang",
  units="R.2 Sous-groupe des Mines; OBI (RAT grise, D.Strat, RSF); OBS; BOMZ; SDS/SD2a–d/SD3a–b; CMN; dolomite and graphitic shale",
  structures="Synclinal deposit; bedding N10°E–N30°E; fractures/veins N100°E–N130°E; mineralised veins N110°E–N120°E; joints N120°E–N130°E; dips 66° to subvertical; polyphase deformation; no directional faults identified",
  deposits="Kamoto Étang, Kamoto Principal",
  scale="n/a (not a map)", crs="n/a (not a map); stereographic (Schmidt) structural data",
  pages="p.9–10 (introduction, ore horizons); p.65–66 (fault/fracture analysis); p.68 (general conclusion)",
  direct="Measured bedding and fracture orientations; ore minerals chalcopyrite, bornite, pyrite, chalcocite, carrollite, covellite, goethite, malachite, azurite; stratiform syn-diagenetic plus epigenetic vein mineralisation.",
  interp="Polyphase tectonic model; structural control on veining; paragenetic sequence.",
  conf="high", amb="No coordinates; individual faults are not named, only orientation families; the dataset is local to Kamoto Étang."),
 dict(id="1408", tier="1",
  title="Contribution à l'étude géologique des sondages de Luiswishi (LSW-1215 et LSW-1216), Haut-Katanga, RDC: lithostratigraphie, pétrographie, métallographie et radiométrie",
  date="2025", author="Manyong Cham Pascaline; Université de Lubumbashi, Département de Géologie",
  dtype="Thesis (borehole study)", coverage="Luiswishi, Haut-Katanga",
  subject="Lithostratigraphy, petrography, metallography and radiometry of two Luiswishi boreholes",
  units="SDS; Brèche de CMN; SDB; BOMZ; RSC; RSF; D.Strat; RAT grise; RAT lilas; CMN; R.2 (Mines) and R.1",
  structures="Synclinal structure at Luiswishi; mineralisation in rock matrix and fracture fillings",
  deposits="Luiswishi (Kamoto, Kambove, Kipushi, Shinkolobwe, Fungurume as context)",
  scale="n/a (boreholes)", crs="n/a; borehole collar coordinates not in the extracted excerpt",
  pages="p.79–80 (general conclusion)",
  direct="Downhole unit succession for each borehole; sulphides (pyrite, chalcopyrite, bornite) and oxides (goethite, malachite); radiometry 18–28 CPM, mean 22 CPM (background level); synclinal structure.",
  interp="Sedimentary/detrital-chemical origin; radiometric values interpreted against WHO dose guidance.",
  conf="high", amb="Borehole collar coordinates and unit thicknesses were not captured in the excerpt."),
 dict(id="1919", tier="4",
  title="Inventaire des minéralisations au sud du 12° latitude sud dans le Sud-Katanga",
  date="2006", author="Ilunga Nkulu Mukembe Victor; supervisor C.T. Kanzundu; Université de Lubumbashi",
  dtype="Graduate thesis / occurrence inventory", coverage="Southern Katanga, south of 12°S",
  subject="Inventory of mineralisations south of 12°S",
  units="Not captured (front matter only)", structures="Not captured (front matter only)",
  deposits="Not captured (front matter only)",
  scale="n/a (not a map)", crs="n/a",
  pages="p.1–3 only (cover, epigraph, dedication)",
  direct="Title, author, supervisor and academic year are confirmed from the cover.",
  interp="None.",
  conf="high (bibliographic) / none (geological)", amb="Body pages not OCR'd. RDC_1919 and RDC_2755 are two catalogue records of the same thesis (39 vs 37 pages), probably duplicate scans: OCR one only."),
 dict(id="2755", tier="4",
  title="Inventaire des minéralisations au sud du 12° latitude sud dans le Sud-Katanga",
  date="2006", author="Ilunga Nkulu Mukembe Victor; supervisor C.T. Kanzundu; Université de Lubumbashi",
  dtype="Graduate thesis / occurrence inventory (duplicate scan)", coverage="Southern Katanga, south of 12°S",
  subject="Inventory of mineralisations south of 12°S",
  units="Not captured (front matter only)", structures="Not captured (front matter only)",
  deposits="Not captured (front matter only)",
  scale="n/a (not a map)", crs="n/a",
  pages="p.1–3 only (cover, epigraph, dedication)",
  direct="Title, author and academic year confirmed; content identical in scope to RDC_1919.",
  interp="None.",
  conf="high (bibliographic) / none (geological)", amb="Duplicate of RDC_1919 with a different page count; do not count as a second independent source."),
 dict(id="3399", tier="4",
  title="Inventaire des indices et minéralisation du territoire de Sakania (Sud-Est du Katanga)",
  date="2008", author="Mukonkole Mpibwe Seraphin; Université de Lubumbashi, Département de Géologie",
  dtype="Graduate thesis / occurrence inventory", coverage="Sakania territory, south-east Katanga",
  subject="Inventory of mineral showings in Sakania territory",
  units="Not captured", structures="Not captured", deposits="Not captured",
  scale="n/a (not a map)", crs="n/a",
  pages="p.1–3 only (cover, epigraph, acknowledgements)",
  direct="Title, author and academic year (2007–2008) confirmed from the cover.",
  interp="None.",
  conf="high (bibliographic) / none (geological)", amb="Body pages not OCR'd; no occurrence data yet available."),
 dict(id="3461", tier="4",
  title="Inventaire des gîtes des minéraux radioactifs dans le Katanga méridional",
  date="2011", author="Kasongo Wa Kasongo; supervisor P.O. Musengie Matamba; Université de Lubumbashi",
  dtype="Graduate thesis / occurrence inventory", coverage="Southern Katanga",
  subject="Inventory of radioactive-mineral deposits",
  units="Not captured", structures="Not captured", deposits="Not captured",
  scale="n/a (not a map)", crs="n/a",
  pages="p.1–3 only (cover, epigraph, dedication)",
  direct="Title, author, supervisor and academic year (2010–2011) confirmed.",
  interp="None.",
  conf="high (bibliographic) / none (geological)", amb="Body pages not OCR'd; no deposit data yet available."),
 dict(id="3731", tier="4",
  title="Inventaire des indices de minéralisation au Katanga",
  date="2005", author="Frédérick Ngoie Mwikondolole; supervisor C.T. Richard Kasongo Monga; Université de Lubumbashi",
  dtype="Graduate thesis / occurrence inventory", coverage="Katanga",
  subject="Inventory of mineralisation indices",
  units="Not captured", structures="Not captured", deposits="Not captured",
  scale="n/a (not a map)", crs="n/a",
  pages="p.1–3 only (cover, epigraph, in memoriam)",
  direct="Title, author, supervisor and academic year (2004–2005) confirmed.",
  interp="None.",
  conf="high (bibliographic) / none (geological)", amb="Body pages not OCR'd; no index data yet available."),
]

FIELDS = ["record_id","original_filename","original_filename_note","title_verified","publication_date","author_institution",
          "document_type","geographic_coverage","geological_subject","priority_tier",
          "stratigraphic_units","named_structures","deposits_occurrences","map_scale",
          "crs_spatial_reference","page_evidence","direct_observations","interpretations",
          "extraction_source","extraction_confidence","unresolved_ambiguities","text_path"]

man = load_manifest()

def text_path_for(doc_id):
    t = DER / "text" / f"RDC_{int(doc_id):04d}.txt"
    if t.exists():
        return t.relative_to(ROOT).as_posix()
    pages = sorted((DER / "ocr" / "text").glob(f"RDC_{int(doc_id):04d}_p*.txt"))
    if pages:
        return "; ".join(p.relative_to(ROOT).as_posix() for p in pages)
    return ""

rows = []
for d in DOCS:
    m = man.get(d["id"], {})
    fn = m.get("object_filename_untrusted", "")
    fn_toks = {w.lower() for w in re.findall(r"[A-Za-z\u00c0-\u00ff]{5,}", fn.replace("-", " "))}
    title_toks = {w.lower() for w in re.findall(r"[A-Za-z\u00c0-\u00ff]{5,}", d["title"] + " " + d["author"])}
    if re.search(r"scan|tada", fn, re.I):
        fn_note = "generic scan filename; carries no title information"
    elif fn_toks & title_toks:
        fn_note = "matches title/author tokens"
    else:
        fn_note = "does not match title/author; object-store name inherited from another record, not an identity field"
    row = {
        "record_id": d["id"],
        "original_filename": fn,
        "original_filename_note": fn_note,
        "title_verified": d["title"],
        "publication_date": d["date"],
        "author_institution": d["author"],
        "document_type": d["dtype"],
        "geographic_coverage": d["coverage"],
        "geological_subject": d["subject"],
        "priority_tier": d["tier"],
        "stratigraphic_units": d["units"],
        "named_structures": d["structures"],
        "deposits_occurrences": d["deposits"],
        "map_scale": d["scale"],
        "crs_spatial_reference": d["crs"],
        "page_evidence": d["pages"],
        "direct_observations": d["direct"],
        "interpretations": d["interp"],
        "extraction_source": "embedded PDF text layer" if (DER/"text"/f"RDC_{int(d['id']):04d}.txt").exists() else "Windows OCR (fr-FR)",
        "extraction_confidence": d["conf"],
        "unresolved_ambiguities": d["amb"],
        "text_path": text_path_for(d["id"]),
    }
    assert row["text_path"], f"no text for {d['id']}"
    rows.append(row)
    # checkpoint per document
    (REG / "rows" / f"RDC_{int(d['id']):04d}.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=1), encoding="utf-8")
    print("checkpoint", d["id"])

with (REG / "geological_evidence_register_v1.csv").open("w", encoding="utf-8-sig", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader(); w.writerows(rows)

# ------------------------------------------------------- extraction report rows
ext_rows = []
for d in DOCS:
    did = int(d["id"])
    t = DER / "text" / f"RDC_{did:04d}.txt"
    ocr = sorted((DER / "ocr" / "text").glob(f"RDC_{did:04d}_p*.txt"))
    if t.exists():
        text = t.read_text(encoding="utf-8", errors="replace")
        pages_total = len(re.findall(r"=== PAGE \d+ ===", text))
        source, pages_used = "text_layer", pages_total
        chars = len(text)
    else:
        text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in ocr)
        pages_total = int(man.get(d["id"], {}).get("page_count_actual") or 0)
        source, pages_used = "windows_ocr", len(ocr)
        chars = len(text)
    ratio = lexical_ratio(text)
    ext_rows.append({
        "record_id": d["id"],
        "page_count_actual": pages_total,
        "pages_extracted": pages_used,
        "extraction_source": source,
        "characters": chars,
        "french_common_word_ratio": f"{ratio:.3f}",
        "extraction_confidence": d["conf"],
        "usable_for_geological_register": "yes" if d["conf"] not in ("low",) else "limited",
        "note": d["amb"],
    })

with (REG / "document_extraction_report_v1.csv").open("w", encoding="utf-8-sig", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(ext_rows[0].keys()))
    w.writeheader(); w.writerows(ext_rows)

lines = ["# Document-level extraction report (v1)", "",
         "Generated from local text layers and Windows OCR. No PDF or page image was sent to the model.", "",
         "| ID | Pages | Extracted | Source | Chars | French ratio | Confidence | Usable |",
         "|---|---|---|---|---|---|---|---|"]
for r in ext_rows:
    lines.append(f"| {r['record_id']} | {r['page_count_actual']} | {r['pages_extracted']} | {r['extraction_source']} | {r['characters']} | {r['french_common_word_ratio']} | {r['extraction_confidence']} | {r['usable_for_geological_register']} |")
(REG / "document_extraction_report_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print("register rows:", len(rows))

# ------------------------------------------------------------ georef inventory
GEOF = ["candidate_id","record_id","map_title","map_class","scale","georeferenceable",
        "crs_or_projection_evidence","grid_or_graticule","candidate_control_points",
        "control_point_basis","bedrock_lithology","stratigraphic_boundaries",
        "structural_traces","deposit_locations","historical_exploration_extent",
        "source_quality","recommended_method","priority","blockers"]

GEO_ROWS = [
 dict(candidate_id="G01", record_id="1043", map_title="Carte géologique du Shaba Méridional (1936)", map_class="Regional bedrock map",
  scale="1:500,000", georeferenceable="yes, conditional",
  crs_or_projection_evidence="None stated on sheet; no datum or projection text legible",
  grid_or_graticule="50 km bar scale; no legible graticule at OCR resolution",
  candidate_control_points="Lubumbashi, Kipushi, Kambove, Likasi, Kolwezi; additional sheet-margin ticks if present",
  control_point_basis="Toponym geolocation against modern gazetteer; margin ticks to be read visually",
  bedrock_lithology="yes", stratigraphic_boundaries="yes", structural_traces="partial (symbols present, names illegible)",
  deposit_locations="yes (copper-deposit symbol)", historical_exploration_extent="no",
  source_quality="1936 print; OCR 0.123; medium-low", recommended_method="Affine or thin-plate spline from 6-10 town GCPs; expect 1-3 km error",
  priority="P2", blockers="No datum; toponym ambiguity; RDC_1317 is a second scan of the same sheet"),
 dict(candidate_id="G02", record_id="1317", map_title="Carte géologique du Shaba Méridional (1936, second scan)", map_class="Regional bedrock map",
  scale="1:500,000", georeferenceable="yes, conditional",
  crs_or_projection_evidence="None stated", grid_or_graticule="Scale bar legible; graticule fragmentary",
  candidate_control_points="Same towns as RDC_1043", control_point_basis="Toponym geolocation",
  bedrock_lithology="yes", stratigraphic_boundaries="yes", structural_traces="partial",
  deposit_locations="yes", historical_exploration_extent="no",
  source_quality="Second scan; OCR 0.149; low-medium", recommended_method="Use whichever scan is sharper; do not georeference both independently",
  priority="P3", blockers="Duplicate of RDC_1043; use only if sharper"),
 dict(candidate_id="G03", record_id="2376", map_title="Shaba Méridional (A. François)", map_class="Regional geological sketch map",
  scale="1:1,000,000", georeferenceable="yes, conditional",
  crs_or_projection_evidence="None stated", grid_or_graticule="No legible graticule; scale bar only",
  candidate_control_points="Kolwezi, Likasi, Lubumbashi, Kambove, Kipushi, Solwezi",
  control_point_basis="Toponym geolocation",
  bedrock_lithology="yes", stratigraphic_boundaries="yes", structural_traces="yes (observed and inferred faults, anticlines, synclines)",
  deposit_locations="no explicit symbol", historical_exploration_extent="no",
  source_quality="Generalised 1:1M sketch; OCR 0.093; low", recommended_method="Rubber-sheet from town GCPs; suitable only as regional context",
  priority="P3", blockers="Very small scale; generalised geology; unsuitable for deposit-scale work"),
 dict(candidate_id="G04", record_id="379", map_title="Carte géologique et notice explicative du degré carré de Mitwaba", map_class="1:100,000 map sheet + notice",
  scale="1:100,000", georeferenceable="verify map plate present",
  crs_or_projection_evidence="None stated", grid_or_graticule="Degree-square graticule implied",
  candidate_control_points="Sheet corners from the degree square; margin ticks",
  control_point_basis="Read coordinates from the sheet margin visually",
  bedrock_lithology="yes", stratigraphic_boundaries="yes", structural_traces="unknown (not text-extracted)",
  deposit_locations="unknown", historical_exploration_extent="no",
  source_quality="Notice text badly OCR'd (0.17); map plate not assessed", recommended_method="Georeference the map plate only after visual confirmation; use degree-square corners",
  priority="P1", blockers="Map plate may not be in the PDF; notice OCR is unreliable"),
 dict(candidate_id="G05", record_id="3854", map_title="Facies et épaisseurs du Groupe R.4", map_class="Thematic facies/isopach map",
  scale="1:500,000", georeferenceable="yes",
  crs_or_projection_evidence="Metric grid in metres; datum and zone not stated (likely Congo Belge Lambert or UTM 35S)",
  grid_or_graticule="Eastings 200,000-340,000 m; latitude labels 25°30'-26°30' (partly legible)",
  candidate_control_points="Grid intersections; Tenke, Fungurume, Kakanda, Kambove, Shinkolobwe, Likasi",
  control_point_basis="Grid intersections are the primary GCPs; towns are secondary checks",
  bedrock_lithology="partial (R.4 facies only)", stratigraphic_boundaries="yes (R.4 base/top, R.4 absent)",
  structural_traces="yes (thrust faults, discordances)", deposit_locations="no",
  historical_exploration_extent="no",
  source_quality="Clean line map; OCR 0.125; medium", recommended_method="Affine from >=4 grid intersections; validate against towns",
  priority="P1", blockers="Datum/zone must be identified before coordinates are published"),
 dict(candidate_id="G06", record_id="3856", map_title="Roan de Tenke-Fungurume", map_class="Detailed geological map",
  scale="1:50,000", georeferenceable="yes, conditional",
  crs_or_projection_evidence="Metric grid; datum not stated", grid_or_graticule="515,000-535,000 E, 320,000-325,000 N",
  candidate_control_points="Grid intersections; Tenke and Fungurume localities",
  control_point_basis="Grid intersections",
  bedrock_lithology="yes (Roan units)", stratigraphic_boundaries="yes", structural_traces="not legible",
  deposit_locations="partial (Tenke-Fungurume area only)", historical_exploration_extent="no",
  source_quality="Figure; OCR 0.053; low", recommended_method="Affine from grid intersections after legend is read visually",
  priority="P3", blockers="Legend abbreviations unresolved; source memoir unidentified"),
 dict(candidate_id="G07", record_id="3863", map_title="Arc cuprifère du Katanga - Zone Centrale, structures tectoniques principales", map_class="Structural map",
  scale="1:200,000", georeferenceable="yes",
  crs_or_projection_evidence="Metric grid; datum not stated", grid_or_graticule="Eastings 200,000-340,000 m; latitude labels ~26°00'-26°30' (partly legible)",
  candidate_control_points="Grid intersections; Kolwezi, Kambove, Likasi; named thrust sheets as checks",
  control_point_basis="Grid intersections primary; towns secondary",
  bedrock_lithology="partial (R.3/R.4 contact)", stratigraphic_boundaries="yes (R.4-R.3 contact)",
  structural_traces="yes (thrusts, back-thrusts, extrusions, fold axes)", deposit_locations="no",
  historical_exploration_extent="no",
  source_quality="Clean structural line map; OCR 0.299; medium", recommended_method="Affine from grid intersections; digitise structural traces as attributed lines",
  priority="P1", blockers="Several toponyms garbled; contact vs fault must be checked visually"),
 dict(candidate_id="G08", record_id="4055", map_title="Région de Kolwezi - structures tectoniques principales et typologie", map_class="Structural/typological map",
  scale="1:250,000", georeferenceable="yes",
  crs_or_projection_evidence="Metric grid implied; datum not stated", grid_or_graticule="Coordinate values only partly legible",
  candidate_control_points="Kolwezi; carapace and thrust-sheet labels; grid intersections if readable",
  control_point_basis="Grid intersections primary; named lobes as topological checks",
  bedrock_lithology="partial (Kundelungu/Roan/Kibarien)", stratigraphic_boundaries="yes",
  structural_traces="yes (nappe front/root, carapaces 1-7, 17 thrust sheets, fold axes)",
  deposit_locations="no explicit symbol", historical_exploration_extent="no",
  source_quality="Rich structural plate; OCR 0.278; medium", recommended_method="Affine from grid; digitise carapace outlines and thrust-sheet polygons",
  priority="P1", blockers="Grid values unreadable in OCR; margin must be inspected visually"),
 dict(candidate_id="G09", record_id="3129", map_title="Stratigraphie de la Série des Mines - Kakanda", map_class="Stratigraphic correlation section",
  scale="1:500 (section)", georeferenceable="no (section, not a plan map)",
  crs_or_projection_evidence="n/a", grid_or_graticule="n/a",
  candidate_control_points="n/a", control_point_basis="n/a",
  bedrock_lithology="partial (lithological symbols)", stratigraphic_boundaries="yes (bed correlation)",
  structural_traces="no", deposit_locations="no", historical_exploration_extent="no",
  source_quality="OCR 0.103; low", recommended_method="Use as attribute reference for Kakanda stratigraphy, not as a spatial layer",
  priority="P2", blockers="Most labels unreadable; needs visual transcription"),
]

with (GEO / "georeferencing_candidates_v1.csv").open("w", encoding="utf-8-sig", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=GEOF)
    w.writeheader(); w.writerows(GEO_ROWS)

# --------------------------------------------------------------- research gaps
gaps = """# Research-gap report (v1)

Scope: independent RDC.ROCKS acquisition workstream. No V14, Stage 3-5 or manuscript files were touched.

## What the evidence now supports

- A coherent Roan stratigraphic vocabulary for the Kolwezi-Kambove-Luiswishi corridor:
  R.1 (RAT), R.2 (Sous-groupe des Mines: RAT grise, D.Strat, RSF, RSC, BOMZ, SDS/SD2-SD3, CMN), R.3 (Dipeta), Mwashya, Nguba, Kundelungu.
- Direct structural measurements at Kamoto Etang: bedding N10-N30E, fractures/veins N100-N130E, mineralised veins N110-N120E, steep dips, polyphase deformation (RDC_1388).
- Regional structural architecture: thrust sheets, back-thrusts, extrusions, carapaces 1-7 and named lobes (RDC_3863, RDC_4055).
- Deposit-scale lithostratigraphy at Kamoto Principal, Mupine and Luiswishi (RDC_957, RDC_1231, RDC_1408).
- A 1954 correlation framework with thicknesses and geographic coordinates (RDC_3848).

## Gaps that still limit the evidence base

1. **No historical exploration records (priority 3).** The selected 118 documents contain no tenure, assay, drilling-campaign or annual exploration reports. Reporting intensity and observation bias therefore cannot yet be separated from geology.
2. **Occurrence inventories are front matter only.** RDC_1919/2755, 3399, 3461 and 3731 have only covers, epigraphs and dedications OCR'd. None of the actual occurrence lists has been captured. RDC_1919 and RDC_2755 are duplicate records of the same thesis.
3. **86 image-only PDFs remain un-OCR'd.** Additional bedrock, structural and occurrence evidence may exist in them; they are unassessed, not absent.
4. **Low-confidence extractions.** RDC_1231 (lexical ratio 0.081), RDC_1300 (0.069), RDC_3856 (0.053), RDC_2376 (0.093) and RDC_379 (0.170) cannot support feature-level assertions without re-OCR or visual transcription.
5. **No map declares a CRS or datum.** All georeferencing candidates carry metre grids or town/graticule control only. No coordinates should be published until the grid zone and datum are established.
6. **Geographic bias.** Evidence clusters on Kolwezi-Kamoto-Luiswishi-Kambove. Mitwaba and Sakania are represented only by a poorly OCR'd notice and an un-OCR'd inventory; Tenke-Fungurume has one 1:50,000 map of low legibility.
7. **No fault-name inventory.** Regional maps name thrust sheets and lobes, but there is no consistent fault database with geometry, kinematics and cross-cutting relationships.
8. **No geochemistry or geophysics beyond one radiometric transect.** RDC_1408 reports 18-28 CPM (background level); no assay or geophysical coverage was acquired.
9. **Duplicate sources.** TIFF 0853 and 0866 are byte-identical; RDC_1043 and RDC_1317 are two scans of the same 1936 map; RDC_1919/2755 duplicate a thesis. Independent-source counts must be adjusted.
10. **Catalogue object filenames are unreliable.** Many `object_filename_untrusted` values name a completely different work (for example RDC_3863 is stored as "etude-des-tassement-des-sols..."). They are recorded but flagged in the register and must never be used as document identity.

## Next concrete steps (in order)

1. OCR RDC_1919 pages 4-10 only (skip duplicate RDC_2755), then RDC_3399, RDC_3461 and RDC_3731 pages 4-10; checkpoint per page.
2. Visually confirm whether the RDC_379 PDF contains the 1:100,000 map plate; if yes, georeference it first (best scale for Mitwaba).
3. Establish the grid datum for RDC_3854, RDC_3863 and RDC_4055 by reading sheet margins; these three give the best structure/facies coverage.
4. Georeference RDC_3854 (isopachs/facies) and RDC_3863 (structures) first; then RDC_4055 (carapaces); then the 1936 map.
5. Only after georeferencing QA, derive structural traces and stratigraphic boundaries.
6. Acquire historical exploration/tenure records to address priority 3.

## Constraint respected

No spatial derivative, no georeferenced raster and no reinterpretation was produced in this batch. The register records observations and the original authors' interpretations in separate columns.
"""
(REG / "research_gaps_v1.md").write_text(gaps, encoding="utf-8")
print("georef rows:", len(GEO_ROWS))
