from __future__ import annotations

import csv
import difflib
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/raw/rdc_rocks/derived/occurrences"
STAGE4 = ROOT / "data/processed/stage4/stage4_occurrences_v1.csv"

DOCS = {
    "3731": {
        "filename": "RDC_3731.pdf",
        "title": "Inventaire des indices de mineralisation au Katanga",
        "author": "Frederick Ngoie Mwikondolole",
        "date": "2004-2005",
        "institution": "Universite de Lubumbashi",
    },
    "3399": {
        "filename": "RDC_3399.pdf",
        "title": "Inventaire des indices et mineralisation du territoire de Sakania (Sud-Est du Katanga)",
        "author": "Mukonkole Mpibwe Seraphin",
        "date": "2007-2008",
        "institution": "Universite de Lubumbashi",
    },
    "1919": {
        "filename": "RDC_1919.pdf",
        "title": "Inventaire des mineralisations au sud du 12e latitude sud dans le sud-Katanga",
        "author": "Victor Ilunga Nkulu Mukembe",
        "date": "2005-2006",
        "institution": "Universite de Lubumbashi",
    },
}

# page, name, classification as published, published location, format, host unit, commodity, uncertainty, notes
R3731 = [
    (23, "Kipushi", "Fer, element siderophile, Katanguien", "place name only", "place_name", "contact Kaponda-Kakontwe; calcaire de Kakontwe", "Fe", "No numeric coordinates; named locality only", "Siderite in hydrothermal/mesothermal deposits along the contact fault."),
    (23, "Ouest de Kipushi (frontiere zambienne)", "Fer, element siderophile, Katanguien", "west of Kipushi along the Zambian border", "descriptive_area", "base du Kundelungu", "Fe", "Area description only", "Residual hematite/magnetite deposits."),
    (23, "Est de Kambove", "Fer, element siderophile, Katanguien", "east of Kambove", "descriptive_area", "base du Kundelungu", "Fe", "Area description only", "Same residual type as west of Kipushi."),
    (23, "Kasumbalesa", "Fer, element siderophile, Katanguien", "northern part of syncline between Luina and Konkola domes", "descriptive_area", "base du Roan / base du Mwashya, gres", "Fe", "No numeric coordinates; structural position given", "Sedimentary iron concentration; railway and Lubumbashi-Zambia road cross the deposit."),
    (24, "Musonoi", "Fer, element siderophile, Katanguien", "place name only", "place_name", "RAT", "Fe", "No numeric coordinates", "Metamorphogenic hematite/magnetite in RAT."),
    (24, "Kamoto", "Fer, element siderophile, Katanguien", "place name only", "place_name", "RAT", "Fe", "No numeric coordinates", "Metamorphogenic hematite/magnetite in RAT."),
    (24, "Kakanda", "Fer, element siderophile, Katanguien", "place name only", "place_name", "calcaires de Kakontwe", "Fe", "No numeric coordinates", "Melanterite in Kakontwe limestone."),
    (24, "Kipushi", "Nickel, element siderophile, Katanguien", "place name only", "place_name", "faille de Kipushi", "Ni", "No numeric coordinates", "Bravoite in hydrothermal/mesothermal deposits of the Kipushi fault."),
    (24, "Shinkolobwe", "Nickel, element siderophile, Katanguien", "place name only", "place_name", "noyau de l'anticlinal", "Ni", "No numeric coordinates", "Garnierite, nepouite, vaesite, millerite, niccolite in the anticline core."),
    (24, "Kasompi", "Nickel, element siderophile, Katanguien", "place name only", "place_name", "not stated", "Ni", "No numeric coordinates", "Garnierite occurrence."),
    (25, "Shinkolobwe", "Cobalt, element siderophile, Katanguien", "place name only", "place_name", "not stated", "Co", "No numeric coordinates", "Cattierite, linneite, carrollite, siegenite."),
    (25, "Musonoi", "Cobalt, element siderophile, Katanguien", "place name only", "place_name", "sous-groupe des mines: SD, up to BOMZ", "Co", "No numeric coordinates", "Sedimentary-type impregnations in lower mine-subgroup beds."),
    (25, "Luishia Principal", "Cobalt, element siderophile, Katanguien", "about 70 km NW of Lubumbashi (sector)", "descriptive_distance", "sous-groupe des mines (incomplete: no BOMZ, upper SD, CMN)", "Co", "Sector distance only; individual deposit not coordinated", "Named as one of the Luishia sector deposits."),
    (25, "Kasongwe", "Cobalt, element siderophile, Katanguien", "Luishia sector", "descriptive_area", "sous-groupe des mines", "Co", "Sector location only", "Listed among Luishia sector deposits."),
    (25, "Karoano", "Cobalt, element siderophile, Katanguien", "Luishia sector", "descriptive_area", "sous-groupe des mines (complete but overturned)", "Co", "Sector location only", "Heterogenite and carrollite; lower orebody structurally above upper orebody."),
    (25, "Kamwale", "Cobalt, element siderophile, Katanguien", "Luishia sector", "descriptive_area", "sous-groupe des mines", "Co", "Sector location only", "Listed among Luishia sector deposits."),
    (25, "Shandwe", "Cobalt, element siderophile, Katanguien", "Luishia sector", "descriptive_area", "sous-groupe des mines", "Co", "Sector location only", "Listed among Luishia sector deposits."),
    (25, "Varseur", "Cobalt, element siderophile, Katanguien", "Luishia sector", "descriptive_area", "sous-groupe des mines", "Co", "Sector location only", "Listed among Luishia sector deposits."),
    (25, "Konsiki", "Cobalt, element siderophile, Katanguien", "place name only (Luishia area)", "place_name", "sous-groupe des mines (complete but overturned)", "Co", "No numeric coordinates", "Heterogenite and carrollite."),
    (26, "Lukuni", "Cobalt, element siderophile, Katanguien", "about 22 km NW of Lubumbashi", "descriptive_distance", "sous-groupe des mines, RAT lilas to CMN", "Co", "Distance only; no bearing precision", "Complete normal mine-subgroup series; beds strike SW-NE and dip SSE; bounded west by a N-S fault."),
    (26, "Kasombo", "Cobalt, element siderophile, Katanguien", "Lupoto sector, Kisanga anticline, 24 km W of Lubumbashi and 20 km N of Kipushi", "descriptive_distance", "sous-groupe des mines, CMN to RAT grises", "Co", "Two distance references; no numeric coordinates", "Lupoto anticline; series of 14 thrust sheets; lower orebody NW, upper orebody SE."),
    (26, "Ruashi", "Cobalt, element siderophile, Katanguien", "about 10 km N of Lubumbashi", "descriptive_distance", "Roan moyen / Roan inferieur, along axial fault", "Co", "Distance only", "Two sandstone-dolomite complexes separated by shales; accessory heterogenite and asbolane."),
    (27, "Etoile", "Cobalt, element siderophile, Katanguien", "12 km NE of Lubumbashi, Ruashi anticline", "descriptive_distance", "RAT and basal SD", "Co", "Distance only", "Etoile polygon includes Etoile, Ruashi and Kimbeyembe deposits; heterogenite in bedding joints, below 80 m piezometric level."),
    (27, "Kimpe-Kipapila", "Cobalt, element siderophile, Katanguien", "90 km SE of Lubumbashi, north of Luishia massif", "descriptive_distance", "shales greseux", "Co", "Distance only", "Heterogenite and asbolane."),
    (27, "Musonoi", "Platines et platinoides (MGP), Katanguien", "place name only", "place_name", "lower mine-subgroup strata; RSF and RAT grises", "PGE", "No numeric coordinates", "PGE as chalcogenides."),
    (28, "Ruwe (Mutoshi)", "Platines et platinoides (MGP), Katanguien", "place name only", "place_name", "lower mine-subgroup strata; chlorite layer", "PGE", "No numeric coordinates", "Powdery descloizite visible at surface."),
    (28, "Shinkolobwe", "Platines et platinoides (MGP), Katanguien", "place name only", "place_name", "chlorite layer", "PGE", "No numeric coordinates", "Thoreau and Dutrieu (1933) reported a chlorite layer mineralised in Au and PGE; native platinum."),
    (28, "Ruwe", "Or, element siderophile, Katanguien", "place name only", "place_name", "sous-groupe des mines (supergene)", "Au", "No numeric coordinates", "Gold interpreted as supergene replacement of Cu in sulphides."),
    (28, "Swambo", "Or, element siderophile, Katanguien", "place name only", "place_name", "sous-groupe des mines (supergene)", "Au", "No numeric coordinates", "Same supergene interpretation as Ruwe."),
    (28, "Kipese", "Or, element siderophile, Katanguien", "37 km SW of Likasi, 7 km from Shinkolobwe", "descriptive_distance", "Roan megafragment; complete mine subgroup RAT-CMN", "Au", "Distance only", "Disseminated gold; highest concentrations in RSF, D. strat, RAT grise and RSF/RSC contact."),
    (29, "Kanyaka", "Or, element siderophile, Katanguien", "20 km S of Lubumbashi, Muniama river plain", "descriptive_distance", "alluvial gravels", "Au", "Distance only", "Alluvial deposit associated with two gravel beds."),
    (29, "Mutoshi", "Or, element siderophile, Katanguien", "place name only", "place_name", "residual/eluvial placers after sulphide alteration", "Au", "No numeric coordinates", "Gold in iron caps and oxidation zone; also Pt, Pd, V reported."),
    (32, "Musoshi", "Cu-Co-U, gites dans les RAT", "place name only", "place_name", "top of argillaceous schist, sandstone and feldspathic quartzite complex", "Cu-Co-U", "No numeric coordinates", "Stratigraphic position stated."),
    (32, "Kisenda", "Cu-Co-U, gites dans les RAT", "place name only", "place_name", "top of feldspathic quartzites, arkose and conglomerates; base of argillaceous complex", "Cu-Co-U", "No numeric coordinates", "Stratigraphic position stated."),
    (32, "Lubembe", "Cu-Co-U, gites dans les RAT", "place name only", "place_name", "top of feldspathic quartzites, arkose and conglomerates; base of argillaceous complex", "Cu-Co-U", "No numeric coordinates", "Stratigraphic position stated."),
    (32, "Etoile", "Cu-Co-U, sous-groupe des mines", "place name only", "place_name", "sous-groupe des mines", "Cu-Co-U", "No numeric coordinates", "Listed among the main deposits of the Lubumbashi-Kolwezi fold arc."),
    (32, "Rwashi", "Cu-Co-U, sous-groupe des mines", "place name only", "place_name", "sous-groupe des mines", "Cu-Co-U", "No numeric coordinates", "Listed among the main deposits of the Lubumbashi-Kolwezi fold arc."),
    (32, "Luiswishi", "Cu-Co-U, sous-groupe des mines", "place name only", "place_name", "sous-groupe des mines", "Cu-Co-U", "No numeric coordinates", "Listed among the main deposits of the Lubumbashi-Kolwezi fold arc."),
    (32, "Lukuni", "Cu-Co-U, sous-groupe des mines", "place name only", "place_name", "sous-groupe des mines", "Cu-Co-U", "No numeric coordinates", "Listed among the main deposits of the Lubumbashi-Kolwezi fold arc."),
    (32, "Kasonta", "Cu-Co-U, sous-groupe des mines", "place name only", "place_name", "sous-groupe des mines", "Cu-Co-U", "No numeric coordinates", "Listed among the main deposits of the Lubumbashi-Kolwezi fold arc."),
    (32, "Luishia", "Cu-Co-U, sous-groupe des mines", "place name only", "place_name", "sous-groupe des mines", "Cu-Co-U", "No numeric coordinates", "Listed among the main deposits of the Lubumbashi-Kolwezi fold arc."),
    (32, "Kamatanda", "Cu-Co-U, sous-groupe des mines", "place name only", "place_name", "sous-groupe des mines", "Cu-Co-U", "No numeric coordinates", "Listed among the main deposits of the Lubumbashi-Kolwezi fold arc."),
    (32, "Likasi", "Cu-Co-U, sous-groupe des mines", "place name only", "place_name", "sous-groupe des mines", "Cu-Co-U", "No numeric coordinates", "Listed among the main deposits of the Lubumbashi-Kolwezi fold arc."),
    (32, "Kambove", "Cu-Co-U, sous-groupe des mines", "place name only", "place_name", "sous-groupe des mines", "Cu-Co-U", "No numeric coordinates", "Listed among the main deposits of the Lubumbashi-Kolwezi fold arc."),
    (32, "Kambove Ouest", "Cu-Co-U, sous-groupe des mines", "place name only", "place_name", "sous-groupe des mines", "Cu-Co-U", "No numeric coordinates", "Listed among the main deposits of the Lubumbashi-Kolwezi fold arc."),
    (32, "Fungurume", "Cu-Co-U, sous-groupe des mines", "place name only", "place_name", "sous-groupe des mines", "Cu-Co-U", "No numeric coordinates", "Listed among the main deposits of the Lubumbashi-Kolwezi fold arc."),
    (32, "Musonoi", "Cu-Co-U, sous-groupe des mines", "place name only", "place_name", "sous-groupe des mines", "Cu-Co-U", "No numeric coordinates", "Listed among the main deposits of the Lubumbashi-Kolwezi fold arc."),
    (32, "Mutoshi (Ruwe)", "Cu-Co-U, sous-groupe des mines", "place name only", "place_name", "sous-groupe des mines", "Cu-Co-U", "No numeric coordinates", "Listed among the main deposits of the Lubumbashi-Kolwezi fold arc."),
    (32, "Shituru", "Cu-Co-U, above the mine subgroup", "place name only", "place_name", "above mine subgroup, below siliceous oolite (Roan)", "Cu-Co-U", "No numeric coordinates", "Stratigraphically above the mine subgroup and below the siliceous oolite."),
    (32, "Kimbwe", "Cu-Co-U, above the mine subgroup", "place name only", "place_name", "above mine subgroup, below siliceous oolite (Roan)", "Cu-Co-U", "No numeric coordinates", "Stratigraphically above the mine subgroup and below the siliceous oolite."),
    (32, "Kasompi", "Cu-Co-U, above the mine subgroup", "place name only", "place_name", "above mine subgroup, below siliceous oolite (Roan)", "Cu-Co-U", "No numeric coordinates", "Stratigraphically above the mine subgroup and below the siliceous oolite."),
    (33, "Kipushi", "Cu-Pb-Zn, calcaire de Kakontwe", "place name only", "place_name", "calcaire de Kakontwe; breche de charriage", "Cu-Zn-Pb-Ag-Ge", "No numeric coordinates", "Anticline cut along its length by a thick thrust breccia; deposit at the Kakontwe contact; hydrothermal/mesothermal."),
    (33, "Lombe", "Zn-Pb, calcaire de Kakontwe", "about 70 km W of Kipushi, north flank of Lombe anticline", "descriptive_distance", "calcaire de Kakontwe", "Zn-Pb", "Distance only", "Polymetallic belt around the southern Katanga iron zone."),
    (33, "Kengere", "Pb-Zn-Ag-Cu, calcaire de Kakontwe", "50 km S of Kolwezi", "descriptive_distance", "fault zone in calcaire de Kakontwe", "Pb-Zn-Ag-Cu", "Distance only", "Fault-hosted Pb-Zn."),
    (33, "Lukila", "Cu-Pb-Zn, calcaire de Kakontwe", "Kakontwe limestone belt", "descriptive_area", "calcaire de Kakontwe", "Cu-Pb-Zn", "Belt location only", "Listed in the Kakontwe limestone deposit belt."),
    (33, "Tantara", "Cu-Pb-Zn, calcaire de Kakontwe", "Kakontwe limestone belt", "descriptive_area", "cavities in calcaire de Kakontwe", "Cu-Pb-Zn (Sr, Co)", "Belt location only", "Also spelled Tangara; essentially a copper deposit with dioptase, plancheite, shattuckite, brochantite; strontium and cobalt-bearing calcite."),
    (33, "Sampwe", "Cu-Pb-Zn, calcaire de Kakontwe", "Kakontwe limestone belt", "descriptive_area", "calcaire de Kakontwe", "Cu-Pb-Zn", "Belt location only", "Listed in the Kakontwe limestone deposit belt."),
    (33, "Kirundu", "Cu-Pb-Zn, calcaire de Kakontwe", "Kakontwe limestone belt", "descriptive_area", "calcaire de Kakontwe", "Cu-Pb-Zn", "Belt location only", "Listed in the Kakontwe limestone deposit belt."),
    (33, "Tenke", "Cu-Pb-Zn, calcaire de Kakontwe", "Kakontwe limestone belt", "descriptive_area", "calcaire de Kakontwe", "Cu-Pb-Zn", "Belt location only", "Listed in the Kakontwe limestone deposit belt."),
    (34, "Mulunguishi", "Zn-Pb hydrothermal-epithermal", "place name only", "place_name", "not stated", "Zn-Pb", "No numeric coordinates", "Cited to Oosterbosch (1968), carte metallogenique du Shaba."),
    (34, "Kapuma", "Zn-Pb hydrothermal-epithermal", "place name only", "place_name", "not stated", "Zn-Pb", "No numeric coordinates", "Cited to Oosterbosch (1968)."),
    (34, "Muita", "Zn-Pb hydrothermal-epithermal", "place name only", "place_name", "not stated", "Zn-Pb", "No numeric coordinates", "Cited to Oosterbosch (1968)."),
    (35, "Kalongwe", "Uranium, Katanguien", "southern border of the copper arc, western end", "descriptive_area", "not stated", "U", "Regional position only", "Uraniferous occurrences along the southern border of the copper arc from Kalongwe to Luiswishi."),
    (35, "Munda", "Uranium, Katanguien", "southern border of the copper arc", "descriptive_area", "not stated", "U", "Regional position only", "Listed between Kalongwe and Kasompi."),
    (35, "Kasompi", "Uranium, Katanguien", "southern border of the copper arc", "descriptive_area", "not stated", "U", "Regional position only", "Listed between Munda and Swambo."),
    (35, "Swambo", "Uranium, Katanguien", "west of Shinkolobwe, near Mindigi mine", "descriptive_area", "lower orebody; bedding joints", "U", "Relative location only", "Uranium in veinlets along bedding joints and disseminated in wall rock."),
    (35, "Shinkolobwe", "Uranium, Katanguien", "15 km S of Kambove", "descriptive_distance", "Kundelungu schists, mine-subgroup formations, carbonate breccias", "U", "Distance only", "Mining polygon elongated E-W; pitchblende in veins; uraninite and secondary uranium minerals."),
    (35, "Luishia", "Uranium, Katanguien", "southern border of the copper arc", "descriptive_area", "not stated", "U", "Regional position only", "Listed along the southern border of the copper arc."),
    (35, "Luiswishi", "Uranium, Katanguien", "southern border of the copper arc, south-east end", "descriptive_area", "RAT grises / RAT lilas redox front", "U", "Regional position only", "Stratiform uranium pre-concentrated during Roan sedimentation; redox front between lilac and grey RAT."),
    (35, "Kakanda-Sud", "Uranium, Katanguien", "northern copper-rich facies", "descriptive_area", "not stated", "U", "Regional position only", "May form exploitable concentrations."),
    (35, "Kambove-Ouest", "Uranium, Katanguien", "northern copper-rich facies", "descriptive_area", "not stated", "U", "Regional position only", "May form exploitable concentrations."),
    (36, "Kamoto", "Uranium, Katanguien", "8 km W of Kolwezi", "descriptive_distance", "first 40 m of lower Roan; Kamoto thrust sheet", "U", "Distance only", "Stratiform uranium in two orebodies; kamotoite, shabaite, francoisite."),
    (37, "Musonoi", "Uranium, Katanguien", "centre of Kolwezi fragment", "descriptive_area", "base dolomitic schists", "U", "Relative location only", "Uraninite, becquerelite, curite, kasolite."),
    (39, "Kasaji-Kisenge", "Or, Kibarien", "Kasaji-Kisenge region", "descriptive_area", "Archean Sandoa-Kapanga complex; migmatites, microschists, amphibolites, quartzites, metagabbros, granitoids", "Au", "Region only", "Primary gold in veins and dykes; author relates mineralisation genetically to metagabbro."),
    (39, "Sandoa-Kapanga", "Or, Kibarien", "along the Lulua river", "descriptive_area", "Archean Sandoa-Kapanga complex", "Au", "Region only", "Primary and secondary gold."),
    (39, "Musefu", "Or, Kibarien", "along the Katanga river (Lulua tributary)", "descriptive_area", "not stated", "Au", "Region only", "Important gold veins along the Katanga river."),
    (39, "Manono-Mitwaba", "Or, Kibarien", "Manono-Mitwaba sector", "descriptive_area", "pegmatite veins", "Au", "Sector only", "Auriferous pegmatite veins (Cahen, 1954)."),
    (39, "Riviere Kasai", "Or, Kibarien", "along the Kasai river", "descriptive_area", "not stated", "Au", "Region only", "Occurrences observed along the Kasai river."),
    (40, "Busanga", "Etain (Sn), Kibarien", "Kibarian chain", "descriptive_area", "quartz veins", "Sn", "Regional position only", "Primary cassiterite in quartz veins."),
    (40, "Mitwaba", "Etain (Sn), Kibarien", "Kibarian chain", "descriptive_area", "granite margin; upper Kibarian country rocks", "Sn", "Regional position only", "Cassiterite in quartz veins around a granite massif; fan-shaped veins; sulphides and yttrocrasite."),
    (40, "Mwika", "Etain (Sn), Kibarien", "Kibarian chain", "descriptive_area", "pegmatites", "Sn", "Regional position only", "Cassiterite in pegmatites."),
    (40, "Manono", "Etain (Sn), Kibarien", "Kibarian chain", "descriptive_area", "pegmatite laccoliths", "Sn", "Regional position only", "Two pegmatite laccoliths, each about 5 km long and 400 m wide."),
    (41, "Lula", "Etain (Sn), Kibarien", "place name only", "place_name", "tourmalinised and silicified microschists", "Sn", "No numeric coordinates", "Mineralised alignment in microschists; a more northerly alignment comprises quartz veins with golden mica."),
    (41, "Mandwe", "Etain (Sn), Kibarien", "place name only", "place_name", "stanniferous greisen pegmatite veins", "Sn", "No numeric coordinates", "Black cassiterite with columbite."),
    (41, "Muika", "Etain (Sn), Kibarien", "place name only", "place_name", "pegmatite lenses; greisen", "Sn", "No numeric coordinates", "Pegmatite with microcline, oligoclase, albite, quartz, white mica; columbo-tantalite in one working (plage IX)."),
    (40, "Shiombio", "Etain indicator minerals, Kibarien", "around the Shiombio granite massif", "descriptive_area", "tourmalinised granite margin", "Sn (scheelite, beryl)", "Massif only", "Strongly tourmalinised zones contain scheelite and beryl but no cassiterite."),
    (42, "Kasekelesa", "Manganese, Kibarien", "SW of Kolwezi", "descriptive_area", "Kibarian sandstone-quartzite beds", "Mn", "Relative location only", "Psilomelane layers/masses with hollandite and braunite; replacement deposit; some ore carries Pb and Ba."),
    (42, "Kisenge", "Manganese, Kibarien", "upper Lualaba region", "descriptive_area", "Lukoshi formation schists", "Mn", "Region only", "Masses replacing metamorphic schists and veinlets; polianite and hollandite."),
    (43, "Manono", "Niobium-tantalum, Kibarien", "place name only", "place_name", "mineralised pegmatites", "Nb-Ta", "No numeric coordinates", "Columbo-tantalite in pegmatites."),
    (43, "Kyato", "Niobium-tantalum, Kibarien", "place name only", "place_name", "mineralised pegmatites", "Nb-Ta", "No numeric coordinates", "Columbo-tantalite in pegmatites."),
    (43, "Bukena", "Niobium-tantalum, Kibarien", "place name only", "place_name", "mineralised pegmatites", "Nb-Ta", "No numeric coordinates", "Columbo-tantalite in pegmatites."),
    (43, "Mwika", "Niobium-tantalum, Kibarien", "place name only", "place_name", "rare-earth pegmatites", "Nb-Ta", "No numeric coordinates", "Columbo-tantalite in pegmatites."),
    (43, "Katumba", "Niobium-tantalum, Kibarien", "place name only", "place_name", "mineralised pegmatites", "Nb-Ta", "No numeric coordinates", "Columbo-tantalite in pegmatites."),
    (43, "Busanga", "Niobium-tantalum, Kibarien", "place name only", "place_name", "quartz veins", "Nb-Ta", "No numeric coordinates", "Vein-type columbo-tantalite."),
    (43, "Kaita", "Niobium-tantalum, Kibarien", "place name only", "place_name", "quartz veins", "Nb-Ta", "No numeric coordinates", "Vein-type columbo-tantalite."),
    (43, "Kalima", "Niobium-tantalum, Kibarien", "place name only", "place_name", "quartz veins", "Nb-Ta", "No numeric coordinates", "Vein-type columbo-tantalite."),
    (43, "Kamunka", "Niobium-tantalum, Kibarien", "Malemba-Manono area", "descriptive_area", "both pegmatite and vein types", "Nb-Ta", "Area only", "Region contains both deposit types."),
    (43, "Malemba", "Niobium-tantalum, Kibarien", "place name only", "place_name", "pegmatites", "Nb-Ta", "No numeric coordinates", "Columbo-tantalite in pegmatites."),
    (43, "Kanua", "Niobium-tantalum, Kibarien", "Malemba area", "descriptive_area", "rare-earth pegmatites", "Nb-Ta", "Area only", "Columbo-tantalite in pegmatites."),
    (44, "Malemba-Nkulu", "Tungsten-molybdenum, Kibarien", "Malemba-Nkulu area", "descriptive_area", "apical parts of granitic massifs; quartz veins", "W-Mo", "Area only", "Wolframite, huebnerite, scheelite, molybdenite; also in alluvial deposits."),
    (45, "Malemba-Nkulu (tin-granite veins)", "Bismuth, Kibarien", "Malemba-Nkulu area", "descriptive_area", "tin-bearing granite veins", "Bi", "Area only", "Native bismuth associated with arsenopyrite, wolframite, cassiterite, beryl, tourmaline."),
    (45, "Malemba-Nkulu (alluvial)", "Beryllium, Kibarien", "Malemba-Nkulu area", "descriptive_area", "alluvial deposits", "Be", "Area only", "Beryl varieties in alluvial crystals and rolled pebbles."),
    (46, "Mulombozi-Sambala", "Or, Ubendien", "SW of Mpala, along the Tanganyika coast", "descriptive_area", "granites and basic rocks; eluvium and quartz veins", "Au", "Area only", "Two mineralised veins recognised: Kamalingela and Mutotolwa; deposit described as poorly known."),
    (46, "Lukonzolwa", "Or, Ubendien", "three groups: Kiyimbi-Mudapale (N), Luama, Lugumba (S)", "descriptive_area", "granitic and microgranitic rocks with rhyolites", "Au", "Area only", "Alluvial gold derived from primary deposits, notably Kamalingela."),
    (46, "Kabobo", "Or, Ubendien", "between the 5th parallel S and the sources of the Lwama, Tanganyika margin", "descriptive_area", "magmatic and volcano-sedimentary rocks; graphitic pyritic schists; gneissic granite", "Au", "Area only", "Ancient and recent alluvium rich in gold; dense pegmatite/aplite network on the Tanganyika-Luena ridge."),
    (47, "Pola-Tembwe", "Or, Ubendien", "between the foot of the Muhila mountains and the Tanganyika shore", "descriptive_area", "micaschists, phyllites, amphibolites, chlorite schists, gneiss, granite, basic rocks", "Au", "Area only", "Gold in quartz veins, weakly mineralised, with pyrite and chalcopyrite; alluvial concentrations."),
    (47, "Kimunga", "Or, Ubendien", "Monjoie basin and its tributary", "descriptive_area", "granitic massif south; micaschist/sandstone-quartzite alternation north", "Au", "Area only", "Dense stockwork of veins and veinlets."),
    (48, "Moba", "Cuivre, Ubendien", "NE of the Katanga copper-cobalt province", "descriptive_area", "quartz veins near gabbro masses and dolerite dykes", "Cu", "Area only", "Chalcocite and copper carbonates; stratigraphic position of mineralised beds not precisely defined."),
    (48, "Bassin de la Luena", "Charbon, serie de couverture (Lukuga)", "south of Bukama", "descriptive_area", "upper Lukuga, Permo-Carboniferous", "Coal", "Area only", "Eight mining polygons: Kisulu, Luena, Kalule Nord, Kalule, Kamienga, Kawawa, Sankosha, Shina."),
    (50, "Makala", "Charbon, serie de couverture (Lukuga)", "15 km from Kalemie, Lukuga basin", "descriptive_distance", "Lukuga series, Permo-Carboniferous", "Coal", "Distance only", "Principal deposit of the northern Lukuga coal basin."),
]

# page, name, classification, coordinates as published, format, host, commodity, uncertainty, notes, confidence
R3399 = [
    (28, "Kasumbalesa", "Fer, gisement, base du Roan", "northern part of syncline between Luina and Konkola domes; railway and Lubumbashi-Zambia road cross the deposit", "descriptive_area", "base Roan conglomerate, arkosic conglomerate, quartzite and sandstone (Formation Ferrifere de Kasumbalesa)", "Fe", "No numeric coordinates in this document; structural and road description only", "Hematite-rich lenticular conglomerates and sandstones; Gysin (1936), Oosterbosch (1962) and Balabala (1982) interpretations.", "high"),
    (29, "Kimpe (Kipapila)", "Cu-Co, facies Katangais", "12 deg 03-08 min S; 27 deg 50-56 min E", "lat_lon_range_dms (datum not stated)", "lower orebody: RAT grise, D strat and RSF of the mine subgroup", "Cu-Co", "Published 5 x 6 minute bounding box, not a deposit point", "50 km NE of Musoshi and about 112 km SE of Lubumbashi; altitude 1190-1200 m; malachite worked by local people before prospecting (Musenge, 1996).", "high"),
    (30, "Mabaya", "Cu-Fe-Mn-Co, indice/gisement", "90 km SE of Lubumbashi on the Lubumbashi-Kasumbalesa road; bounded by Luanshiula river (N), high-voltage line and Kimpe road (S), Musoshi river (W) and Kangiri river (E)", "descriptive_area", "feldspathic quartzites, argillaceous shales and SD of the lower Roan", "Cu-Fe-Mn-Co", "No numeric coordinates; river and road boundaries only", "Boreholes at Mabaya intersected non-negligible copper mineralisation in RAT-like, D strat and RSF-type detrital rocks (Makoka, 1992).", "high"),
    (31, "Kapapa", "Cu, indice", "12 deg 00-08 min S; 27 deg 46-49 min E", "lat_lon_range_dms (datum not stated)", "middle part of the mine subgroup; mineralised sandstone near the Kapapa river", "Cu", "Published 8 x 3 minute bounding box", "22 km NW of Musoshi on the south flank of the Lupoto anticline against the Luena dome; altitude 1290-1450 m; mapped by CSK then UMHK, then Gysin 1929-1932; only malachite-stained sandstone about 10 m from intact UMHK trenches; no interesting mineralisation reported.", "high"),
    (32, "Musoshi", "Cu (Co), stratiform gisement", "12 deg 11-14 min S; 27 deg 40-45 min E", "lat_lon_range_dms (datum not stated)", "shales, microsandstones and argillaceous-micaceous rocks at the top of the lower Roan", "Cu (Co)", "Published 3 x 5 minute bounding box", "75 km SE of Lubumbashi; altitude 1225-1450 m; NW extension of the Zambian Konkola deposit; hypogene sulphides from 100-150 m depth; vertical stratigraphic zoning.", "high"),
    (33, "Kinsenda", "Cu, stratiform gisement", "12 deg 13-16 min S; 27 deg 56-59 min E", "lat_lon_range_dms (datum not stated)", "Roan arenites and rudites; Upper, Middle and Lower Ore Zones (UOZ, MOZ, LOZ)", "Cu", "Published 3 x 3 minute bounding box", "110 km SE of Lubumbashi; altitude about 1250 m; dip 20-25 deg SW; reserves 37,600 t at 6.7% Cu (Ilunga, 1988); exploited by underground galleries; recent drilling by SODIMICO (1987).", "high"),
    (35, "Lubembe", "Cu, gisement", "approximately 12 deg 22 min S; 28 deg 10 min E (OCR damaged)", "degrees_minutes (partly illegible)", "lower Mutonda formation: feldspathic sandstones and quartzites, conglomeratic arkoses", "Cu", "Approximate; OCR damaged; coordinate remains unresolved", "20 km SE of Kinsenda on the west flank of the western Lubembe syncline; altitude 1250 m; discovered by UMHK geologists; NW extension of the Kasaria-Luansobe mineralised zone.", "medium"),
    (36, "Mokambo", "Cu, gisement", "approximately 12 deg 16-27 min S; 28 deg 19-26 min E (OCR damaged)", "lat_lon_range_dms (partly illegible)", "lower Roan quartzites, quartzo-feldspathic sandstones, pelite-quartz-mica unit and cherty dolomite", "Cu", "OCR damaged; latitude and longitude partly unresolved", "Granite massif of about 51 km2; 63 km NW of Sakania and about 130 km SE of Lubumbashi; malachite impregnations; copper phosphate concretions in cherty dolomite.", "medium"),
    (36, "Lonshi", "Cu (Zn-Pb, Co), gisement", "13 deg 10-11 min S; 28 deg 55-57 min E", "lat_lon_range_dms (datum not stated)", "feldspathic, quartzitic and sandstone lower Roan; conglomeratic shales possibly lower Kundelungu", "Cu (Zn-Pb, Co)", "Published 1 x 2 minute bounding box", "130 km SE of Kinsenda; altitude 1200-1303 m between Musendeka summit (S) and Sabwe hill (N); recent drilling; currently mined by COMISA; origin not established.", "high"),
    (37, "Kasabi", "Cu-Co, indice", "within 12 deg 00-13 deg 10 min S; 27 deg 30-28 deg 58 min E", "regional_bounding_box", "not defined", "Cu-Co", "Regional box only; individual location not published", "Listed among the mineralisation indices of Kombo chiefdom, Sakania territory; lithology not yet defined.", "high"),
    (37, "Kamikuba", "Cu-Co, indice", "within 12 deg 00-13 deg 10 min S; 27 deg 30-28 deg 58 min E", "regional_bounding_box", "not defined", "Cu-Co", "Regional box only; individual location not published", "Listed among the mineralisation indices of Kombo chiefdom, Sakania territory; lithology not yet defined.", "high"),
    (37, "Kipula", "Cu-Co, indice", "within 12 deg 00-13 deg 10 min S; 27 deg 30-28 deg 58 min E", "regional_bounding_box", "not defined", "Cu-Co", "Regional box only; individual location not published", "Listed among the mineralisation indices of Kombo chiefdom, Sakania territory; lithology not yet defined.", "high"),
    (37, "Kombo", "Cu-Co, indice", "within 12 deg 00-13 deg 10 min S; 27 deg 30-28 deg 58 min E", "regional_bounding_box", "not defined", "Cu-Co", "Regional box only; individual location not published", "Listed among the mineralisation indices of Kombo chiefdom, Sakania territory; lithology not yet defined.", "high"),
    (37, "Katshili", "Cu-Co, indice", "within 12 deg 00-13 deg 10 min S; 27 deg 30-28 deg 58 min E", "regional_bounding_box", "not defined", "Cu-Co", "Regional box only; individual location not published", "Listed among the mineralisation indices of Kombo chiefdom, Sakania territory; lithology not yet defined.", "high"),
    (37, "Kalinagdinga", "Cu-Co, indice", "within 12 deg 00-13 deg 10 min S; 27 deg 30-28 deg 58 min E", "regional_bounding_box", "not defined", "Cu-Co", "Regional box only; individual location not published", "Listed among the mineralisation indices of Kombo chiefdom, Sakania territory; lithology not yet defined.", "high"),
    (37, "Kipalushia", "Cu-Co, indice", "within 12 deg 00-13 deg 10 min S; 27 deg 30-28 deg 58 min E", "regional_bounding_box", "not defined", "Cu-Co", "Regional box only; individual location not published", "Listed among the mineralisation indices of Kombo chiefdom, Sakania territory; lithology not yet defined.", "high"),
    (37, "Mwati", "Cu-Co, indice", "within 12 deg 00-13 deg 10 min S; 27 deg 30-28 deg 58 min E", "regional_bounding_box", "not defined", "Cu-Co", "Regional box only; individual location not published", "Listed among the mineralisation indices of Kombo chiefdom, Sakania territory; lithology not yet defined.", "high"),
    (37, "Mupitanshi", "Cu-Co, indice", "within 12 deg 00-13 deg 10 min S; 27 deg 30-28 deg 58 min E", "regional_bounding_box", "not defined", "Cu-Co", "Regional box only; individual location not published", "Listed among the mineralisation indices of Kombo chiefdom, Sakania territory; lithology not yet defined.", "high"),
    (37, "Luftwa", "Cu-Co, indice", "within 12 deg 00-13 deg 10 min S; 27 deg 30-28 deg 58 min E", "regional_bounding_box", "not defined", "Cu-Co", "Regional box only; individual location not published", "Listed among the mineralisation indices of Kombo chiefdom, Sakania territory; lithology not yet defined.", "high"),
    (37, "Sabwe", "Cu-Co, indice", "within 12 deg 00-13 deg 10 min S; 27 deg 30-28 deg 58 min E", "regional_bounding_box", "not defined", "Cu-Co", "Regional box only; individual location not published", "Listed among the mineralisation indices of Kombo chiefdom, Sakania territory; lithology not yet defined.", "high"),
]

# page, name, classification, coordinates as published, format, host, commodity, uncertainty, notes, confidence
R1919 = [
    (27, "Lonshi", "Cu, gisement", "13 deg 10-11 min S; 28 deg 55-57 min E", "lat_lon_range_dms (datum not stated)", "Dipeta feldspathic, quartzitic and sandstone formations; conglomeratic shales possibly Nguba", "Cu", "Published 1 x 2 minute bounding box", "130 km SE of Kisenda; altitude 1200-1303 m between Musendeka summit and Sabwe hill. Host assignment conflicts with RDC_3399, which places the mineralisation in lower Roan and possibly lower Kundelungu.", "high"),
    (27, "Mokambo (Mokombo)", "Cu, gisement", "12 deg 16-27 min S; 28 deg 19-26 min E", "lat_lon_range_dms (datum not stated)", "Dipeta quartzites; quartzo-feldspathic sandstones and pelite-quartz-mica unit", "Cu", "Published 11 x 7 minute bounding box", "63 km NW of Sakania and about 130 km SE of Lubumbashi; this source resolves the coordinate string damaged in RDC_3399. Host assignment differs from RDC_3399 (lower Roan).", "high"),
    (28, "Lubembe", "Cu, gisement", "12 deg 19-22 min 05 sec S; 28 deg 10 min E", "lat_lon_range_dms (datum not stated)", "lower Mutonda formation: feldspathic sandstones and quartzites, conglomeratic arkoses", "Cu", "Published 3-minute latitude range and single longitude", "20 km SE of Kisenda on the west flank of the western Lubembe syncline; altitude 1250 m; discovered by UMHK geologists. This source gives a cleaner coordinate string than RDC_3399.", "high"),
    (29, "Kisenda (Kinsenda)", "Cu, gisement", "12 deg 13-16 min S; 27 deg 56-59 min E", "lat_lon_range_dms (datum not stated)", "Roan sandstones; Upper, Middle and Lower Ore Zones (UOZ, MOZ, LOZ)", "Cu", "Published 3 x 3 minute bounding box", "110 km SE of Lubumbashi; altitude about 1250 m; reserves 37,600 t at 6.7% Cu (Ilunga, 1988); underground galleries. Orthographic variant of Kinsenda.", "high"),
    (30, "Mabaya", "Cu-Fe-Mn-Co, gisement", "12 deg 08-11 min S; 27 deg 43-45 min E", "lat_lon_range_dms (datum not stated)", "feldspathic quartzites, argillaceous shales and dolomitic shales of the lower Roan", "Cu-Fe-Mn-Co", "Published 3 x 2 minute bounding box", "Western border of the Luina dome, about 15 km from Musoshi; altitude 1300-1400 m; boreholes intersected Cu mineralisation in RAT-like, D strat and RSF rocks (Makoka, 1992).", "high"),
    (30, "Kimpe-Kipapila", "Cu-Co, gisement", "12 deg 03-08 min S; 27 deg 50-56 min E", "lat_lon_range_dms (datum not stated)", "RAT, D strat, RSF and SD; breccia zones", "Cu-Co", "Published 5 x 6 minute bounding box", "North of the Luina massif, 90 km SE of Lubumbashi; altitude 1190-1200 m; briefly exploited in the 1940s; reported potential of 3 tonnes at 5% Cu (Kipata, 1999).", "high"),
    (34, "Kasumbalesa", "Fe, gisement", "12 deg 06-14 min S; 27 deg 42-54 min E", "lat_lon_range_dms (datum not stated)", "base Roan feldspathic sandstones and arkosic conglomerate; oligiste-haematite mass", "Fe", "Published 8 x 12 minute bounding box", "About 90 km from Lubumbashi; NW-trending hill chain; iron metal 60-70%; production of more than 616,000 t between 1917 and 1949 (SGZ, 1974). Iron indications also reported at Kinsenda, Mabaya, Lubembe and Lonshi.", "high"),
    (32, "Kanyaka", "Au, alluvial gisement", "about 20 km S of Lubumbashi, Muniama river plain (Kanyaka, Kintu and Mupepele)", "descriptive_distance", "alluvial gravels over pedological cover; quartz-rich gravel", "Au", "Distance only; no numeric coordinates", "Two gravel beds 0.2-1 m thick; mineralised flat 10-150 m wide; grade near the river 8.481 g/t; reserves 168.46 kg Au in 45,492.5 t of gravel; intense artisanal exploitation at the time of writing.", "high"),
    (31, "Beya", "Cu, occurrence in the Grand Conglomerat", "place name only", "place_name", "Grand Conglomerat", "Cu", "No numeric coordinates", "Listed among stratiform copper occurrences in the Grand Conglomerat.", "high"),
    (31, "Katchili", "Cu, occurrence in the Grand Conglomerat", "place name only", "place_name", "Grand Conglomerat", "Cu", "No numeric coordinates", "Listed among stratiform copper occurrences in the Grand Conglomerat; probably the same name as Katshili in RDC_3399.", "high"),
    (31, "Kamituba", "Cu, occurrence in the Grand Conglomerat", "place name only", "place_name", "Grand Conglomerat", "Cu", "No numeric coordinates", "Listed among stratiform copper occurrences in the Grand Conglomerat.", "high"),
    (31, "Kasabi", "Cu, occurrence in the Grand Conglomerat", "place name only", "place_name", "Grand Conglomerat", "Cu", "No numeric coordinates", "Listed among stratiform copper occurrences in the Grand Conglomerat; also listed as a Cu-Co index in RDC_3399.", "high"),
    (31, "Kipula", "Cu, occurrence in the Grand Conglomerat", "place name only", "place_name", "Grand Conglomerat", "Cu", "No numeric coordinates", "Listed among stratiform copper occurrences in the Grand Conglomerat; also listed as a Cu-Co index in RDC_3399.", "high"),
    (31, "Kalindinga", "Cu, occurrence in the Grand Conglomerat", "place name only", "place_name", "Grand Conglomerat", "Cu", "No numeric coordinates", "Listed among stratiform copper occurrences in the Grand Conglomerat; probably the same name as Kalinagdinga in RDC_3399.", "high"),
    (31, "Kibutulu", "Cu, occurrence in the Grand Conglomerat", "place name only", "place_name", "Grand Conglomerat", "Cu", "No numeric coordinates", "Listed among stratiform copper occurrences in the Grand Conglomerat.", "high"),
    (31, "Kapalasha", "Cu, occurrence in the Grand Conglomerat", "place name only", "place_name", "Grand Conglomerat", "Cu", "No numeric coordinates", "Listed among stratiform copper occurrences in the Grand Conglomerat.", "high"),
    (35, "Musoshi", "Fe, occurrence", "place name only", "place_name", "Roan; top of the schist-sandstone complex", "Fe", "No numeric coordinates", "Disseminated iron occurrences that continue into the Zambian Copperbelt; pyrite in the cementation zone; haematite south-west of the Luina massif.", "high"),
]


def norm(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = value.lower().replace("'", " ")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def write_occurrences(rows: list[dict]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "rdc_occurrences_v1.csv"
    fields = [
        "occurrence_id", "source_record_id", "source_filename", "source_page",
        "document_title", "original_name", "original_classification",
        "coordinates_as_published", "coordinate_format", "host_unit", "commodity",
        "positional_uncertainty", "extraction_confidence", "notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def reconcile(rows: list[dict]) -> Path:
    aliases = {
        "rwashi": "ruashi",
        "musonoi": "musonie",
        "kisenda": "kinsenda",
        "kasongwe": "kansongwe",
        "kimpe kipapila": "kipapila",
        "kisenda kinsenda": "kinsenda",
    }
    with STAGE4.open("r", encoding="utf-8-sig", newline="") as handle:
        usgs = list(csv.DictReader(handle))
    index = []
    for rec in usgs:
        label = (rec.get("Name") or "") + " " + (rec.get("Name_other") or "")
        index.append((norm(label), rec))

    out = []
    for row in rows:
        raw_name = row["original_name"]
        variants = [raw_name]
        if "(" in raw_name:
            variants.append(raw_name.split("(")[0])
            variants.append(raw_name.split("(")[1].rstrip(")"))
        targets = []
        for variant in variants:
            target = norm(variant)
            target = re.sub(r"\b(ouest|est|nord|sud|principal|mine|deposit|gisement)\b", "", target).strip()
            target = aliases.get(target, target)
            if target and target not in targets:
                targets.append(target)
        alias = targets[0] if len(targets) > 1 and targets[0] != norm(raw_name) else None
        best_score, best_rec, basis = 0.0, None, ""
        for target in targets:
            if len(target) < 4:
                continue
            for label_norm, rec in index:
                tokens = [t for t in label_norm.split() if len(t) >= 4]
                if target == label_norm:
                    score, reason = 1.0, "exact name"
                elif target in tokens:
                    score, reason = 0.95, "exact token"
                elif len(target) >= 6 and target in label_norm:
                    score, reason = 0.9, "token containment"
                else:
                    score = max((difflib.SequenceMatcher(None, target, t).ratio() for t in tokens), default=0.0)
                    reason = "fuzzy token"
                if score > best_score:
                    best_score, best_rec, basis = score, rec, reason
        if best_rec is None or best_score < 0.85:
            confidence, basis = "none", "no candidate >= 0.85"
        elif best_score >= 0.95:
            confidence = "high"
        elif best_score >= 0.90:
            confidence = "medium"
        else:
            confidence = "medium"
        if alias and best_rec is not None and best_score >= 0.85:
            basis = "alias name / orthographic variant"
        out.append({
            "occurrence_id": row["occurrence_id"],
            "source_record_id": row["source_record_id"],
            "original_name": row["original_name"],
            "commodity": row["commodity"],
            "usgs_name": (best_rec.get("Name") if best_rec else ""),
            "usgs_name_other": (best_rec.get("Name_other") if best_rec else ""),
            "usgs_latitude": (best_rec.get("Latitude") if best_rec else ""),
            "usgs_longitude": (best_rec.get("Longitude") if best_rec else ""),
            "match_basis": basis,
            "match_score": f"{best_score:.3f}" if best_rec else "",
            "match_confidence": confidence,
            "geometric_comparison": "not possible: source publishes no numeric coordinates",
            "notes": "Candidate name match only; not merged and not treated as a model positive.",
        })
    path = OUT / "occurrence_usgs_reconciliation_v1.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    return path


def main() -> None:
    rows = []
    for doc_id, spec in DOCS.items():
        source = {"3731": R3731, "3399": R3399, "1919": R1919}[doc_id]
        for i, r in enumerate(source, 1):
            if len(r) == 10:
                page, name, cls, coords, fmt, host, commodity, unc, notes, conf = r
            else:
                page, name, cls, coords, fmt, host, commodity, unc, notes = r
                conf = "high"
            rows.append({
                "occurrence_id": f"OCC-{doc_id}-{i:03d}",
                "source_record_id": doc_id,
                "source_filename": spec["filename"],
                "source_page": f"p{page:03d}",
                "document_title": spec["title"],
                "original_name": name,
                "original_classification": cls,
                "coordinates_as_published": coords,
                "coordinate_format": fmt,
                "host_unit": host,
                "commodity": commodity,
                "positional_uncertainty": unc,
                "extraction_confidence": conf,
                "notes": notes,
            })
    path = write_occurrences(rows)
    rec = reconcile(rows)
    counts = {}
    for row in rows:
        counts[row["commodity"]] = counts.get(row["commodity"], 0) + 1
    print(f"occurrences: {len(rows)} -> {path}")
    print(f"reconciliation: {len(rows)} -> {rec}")
    print("commodity counts:", counts)


if __name__ == "__main__":
    main()
