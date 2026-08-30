import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "C:/Users/vanmu/OneDrive/Documents/Checklist ver3.xlsx";
const outputDir = "outputs/checklist-fr";
const outputPath = `${outputDir}/Checklist ver3 - Français.xlsx`;

const translations = new Map([
  ["COOMEAD DUE DILIGENCE CHECKLIST — SUMMARY TRACKER", "LISTE DE CONTRÔLE DE DUE DILIGENCE DE COOMEAD — SUIVI SYNTHÉTIQUE"],
  ["No.", "N°"], ["Checklist Item", "Élément de contrôle"], ["Status", "Statut"], ["Notes", "Notes"],
  ["1. CORPORATE AND LEGAL STRUCTURE", "1. STRUCTURE SOCIÉTAIRE ET JURIDIQUE"],
  ["2. MINING RIGHTS AND CONCESSIONS", "2. DROITS MINIERS ET CONCESSIONS"],
  ["3. REGULATORY COMPLIANCE", "3. CONFORMITÉ RÉGLEMENTAIRE"],
  ["☑ Verified", "☑ Vérifié"], ["◐ In Progress", "◐ En cours"], ["☐ Not Started", "☐ Non commencé"],
  ["Obtain certified copy of COOMEAD Statutes (Articles of Association) dated 10 August 2025", "Obtenir une copie certifiée conforme des statuts de COOMEAD (acte constitutif) datés du 10 août 2025"],
  ["Confirm RCCM Registration: CD/BN/RCCM/22-B-050 is current and in good standing", "Confirmer que l’immatriculation au RCCM : CD/BN/RCCM/22-B-050 est à jour et en règle"],
  ["Verify National Identification Number: 19-B0500-N14051F is active", "Vérifier que le numéro national d’identification : 19-B0500-N14051F est actif"],
  ["Obtain the Notarial Act (Acte Notarié) dated 23 January 2026 by Notary Kambale Lukula Patrick, Beni", "Obtenir l’acte notarié daté du 23 janvier 2026, établi par le notaire Kambale Lukula Patrick, Beni"],
  ["Verify authenticity of notarization and legal registration", "Vérifier l’authenticité de la légalisation notariale et de l’enregistrement légal"],
  ["Obtain original Ministerial Approval (Arrêté Ministériel) N°0838/CAB.MIN.MINES/01/2015 dated 29 September 2015", "Obtenir l’original de l’arrêté ministériel N°0838/CAB.MIN.MINES/01/2015 daté du 29 septembre 2015"],
  ["Verify Ministerial Approval remains valid and has not been suspended, revoked, or modified", "Vérifier que l’arrêté ministériel demeure valide et n’a pas été suspendu, révoqué ou modifié"],
  ["Confirm registration with relevant provincial mining authorities", "Confirmer l’enregistrement auprès des autorités minières provinciales compétentes"],
  ["Verify complete shareholder register (5 shareholders, 450 of 1,000 shares subscribed)", "Vérifier le registre complet des actionnaires (5 actionnaires, 450 actions souscrites sur 1 000)"],
  ["Verify capital paid-up status: USD 22,500 subscribed of USD 50,000 authorized", "Vérifier la libération du capital : 22 500 USD souscrits sur 50 000 USD autorisés"],
  ["Obtain sworn declarations of ultimate beneficial ownership from each shareholder", "Obtenir de chaque actionnaire une déclaration sous serment sur la propriété effective ultime"],
  ["There are no UBOs as the shareholders are the ultimate beneficial owners ", "Il n’existe pas de bénéficiaires effectifs ultimes distincts, les actionnaires étant eux-mêmes les bénéficiaires effectifs ultimes."],
  ["Verify no pledge, encumbrance, or third-party rights over any shares", "Vérifier l’absence de nantissement, de charge ou de droits de tiers sur les actions"],
  ["Confirm any historical share transfers were properly documented and registered", "Confirmer que tout transfert historique d’actions a été correctement documenté et enregistré"],
  ["Obtain Procès-Verbal of General Assembly authorizing the Joint Venture / Investment with HELIOS", "Obtenir le procès-verbal de l’assemblée générale autorisant la coentreprise / l’investissement avec HELIOS"],
  ["Verify PCA has been properly authorized to sign transaction documents on behalf of COOMEAD", "Vérifier que le PCA a été dûment autorisé à signer les documents de transaction au nom de COOMEAD"],
  ["Confirm the transaction does not require unanimous shareholder approval", "Confirmer que la transaction ne requiert pas l’approbation unanime des actionnaires"],
  ["Conduct politically exposed persons (PEP) screening", "Effectuer un contrôle des personnes politiquement exposées (PPE)"],
  ["Confirm no criminal records or pending prosecutions", "Confirmer l’absence de casier judiciaire ou de poursuites en cours"],
  ["Review any personal guarantees provided by shareholders", "Examiner toute garantie personnelle fournie par les actionnaires"],
  ["Obtain and verify authenticity of Cadastre Minier (CAMI) extracts for ZEA 489", "Obtenir et vérifier l’authenticité des extraits du Cadastre Minier (CAMI) pour la ZEA 489"],
  ["Obtain and verify authenticity of Cadastre Minier (CAMI) extracts for ZEA 490", "Obtenir et vérifier l’authenticité des extraits du Cadastre Minier (CAMI) pour la ZEA 490"],
  ["Obtain and verify authenticity of Cadastre Minier (CAMI) extracts for ZEA 491", "Obtenir et vérifier l’authenticité des extraits du Cadastre Minier (CAMI) pour la ZEA 491"],
  ["Obtain and verify authenticity of Cadastre Minier (CAMI) extracts for ZEA 494", "Obtenir et vérifier l’authenticité des extraits du Cadastre Minier (CAMI) pour la ZEA 494"],
  ["Obtain original certification from CAMI confirming COOMEAD's rights over all four ZEAs", "Obtenir la certification originale du CAMI confirmant les droits de COOMEAD sur les quatre ZEA"],
  ["Mining rights are issued by the Ministry of Mines through research or exploitation permits, and cooperatives may apply for either as needed", "Les droits miniers sont délivrés par le ministère des Mines au moyen de permis de recherche ou d’exploitation, et les coopératives peuvent demander l’un ou l’autre selon leurs besoins."],
  ["Verify Notification Letter N°CAB.MIN/MINES/01/0668/2016 dated 29 April 2016", "Vérifier la lettre de notification N°CAB.MIN/MINES/01/0668/2016 datée du 29 avril 2016"],
  ["Verify Notification Letter N°0591/SG.MINES/2015 dated 8 October 2015", "Vérifier la lettre de notification N°0591/SG.MINES/2015 datée du 8 octobre 2015"],
  ["Confirm no lapse in registration or annual filings", "Confirmer l’absence d’interruption de l’enregistrement ou des déclarations annuelles"],
  ["Obtain updated cadastral maps showing current status", "Obtenir des cartes cadastrales à jour indiquant le statut actuel"],
  ["Verify no overlapping claims with other cooperatives or industrial mining companies", "Vérifier l’absence de revendications qui se chevauchent avec d’autres coopératives ou sociétés minières industrielles"],
  ["Confirm ZEAs are still classified as artisanal exploitation zones by the Ministry of Mines", "Confirmer que les ZEA sont toujours classées comme zones d’exploitation artisanale par le ministère des Mines"],
  ["Verify no re-classification to industrial mining or protected areas", "Vérifier l’absence de reclassement en zone minière industrielle ou en zone protégée"],
  ["Obtain SAEMAPE/SAESSCAM Procès-Verbal dated 20 May 2016 confirming site inspection", "Obtenir le procès-verbal SAEMAPE/SAESSCAM daté du 20 mai 2016 confirmant l’inspection du site"],
  ["Verify current SAEMAPE installation status on the ZEAs", "Vérifier le statut actuel de l’installation de la SAEMAPE sur les ZEA"],
  ["Obtain Governor of Ituri Récépissé N°01/APM/006/CAB/Gouv./PI/2016 dated 23 May 2016", "Obtenir le récépissé du gouverneur de l’Ituri N°01/APM/006/CAB/Gouv./PI/2016 daté du 23 mai 2016"],
  ["Verify current standing with Governor's office and provincial authorities", "Vérifier la situation actuelle auprès du cabinet du gouverneur et des autorités provinciales"],
  ["Verify claims to additional sites: MAYUANO, ELAKE, BAKOLO MBOKA, ALIMA, BELA, PANGITE, BIAKATO area", "Vérifier les revendications sur les sites supplémentaires : zone de MAYUANO, ELAKE, BAKOLO MBOKA, ALIMA, BELA, PANGITE et BIAKATO"],
  ["COOMEAD has settled all customary land fees relating to the identified sites. The outstanding steps are the submission and processing of the application with the Mining Cadastre (CAMI), followed by the Ministry of Mines, to secure the formal mining title.", "COOMEAD a réglé tous les frais fonciers coutumiers relatifs aux sites identifiés. Les étapes restantes sont le dépôt et le traitement de la demande auprès du Cadastre Minier (CAMI), puis du ministère des Mines, afin d’obtenir le titre minier officiel."],
  ["Obtain original Protocol d'Accord with ALIMA community landowners dated 26 May 2016", "Obtenir le protocole d’accord original avec les propriétaires fonciers de la communauté d’ALIMA, daté du 26 mai 2016"],
  ["Verify all signatories to the community protocol", "Vérifier tous les signataires du protocole communautaire"],
  ["Confirm USD 5,000 total payment obligation to community was fulfilled", "Confirmer que l’obligation totale de paiement de 5 000 USD à la communauté a été remplie"],
  ["Verify USD 1,500 initial payment (10-day payment obligation) was made", "Vérifier que le paiement initial de 1 500 USD (obligation de paiement sous 10 jours) a été effectué"],
  ["Obtain any renewal or updated community agreements", "Obtenir tout renouvellement ou accord communautaire mis à jour"],
  ["There is no formal requirement to renew the community agreement. COOMEAD maintains ongoing engagement with the local community and customary authorities through regular consultations and continued relationship management. As part of its community engagement activities, the company participates in local cultural and traditional ceremonies by providing customary in-kind contributions, including livestock (such as goats and chickens), cassava flour, spices, and traditional beverages, in accordance with local practices.", "Il n’existe aucune obligation formelle de renouveler l’accord communautaire. COOMEAD maintient un dialogue continu avec la communauté locale et les autorités coutumières par des consultations régulières et une gestion soutenue des relations. Dans le cadre de ses activités d’engagement communautaire, l’entreprise participe aux cérémonies culturelles et traditionnelles locales en fournissant des contributions coutumières en nature, notamment du bétail (tel que des chèvres et des poulets), de la farine de manioc, des épices et des boissons traditionnelles, conformément aux pratiques locales."],
  ["Confirm no land disputes with local populations", "Confirmer l’absence de litiges fonciers avec les populations locales"],
  ["Obtain evidence of ongoing community relationship and payments", "Obtenir des preuves de la relation continue avec la communauté et des paiements effectués"],
  ["Verify no existing sub-concessions or leases to third parties", "Vérifier l’absence de sous-concessions ou de baux existants au profit de tiers"],
  ["Confirm no pending contract disputes with other operators", "Confirmer l’absence de litiges contractuels en cours avec d’autres opérateurs"],
  ["Check for any pledge, mortgage, or security interest over the ZEAs", "Vérifier l’existence de tout nantissement, hypothèque ou sûreté sur les ZEA"],
  ["COOMEAD does not have any financial obligation tied to its assets", "COOMEAD n’a aucune obligation financière liée à ses actifs."],
  ["Verify no restrictions on partnership with foreign investors", "Vérifier l’absence de restrictions au partenariat avec des investisseurs étrangers"],
  ["Confirm no exclusive off-take agreements with existing buyers", "Confirmer l’absence d’accords exclusifs d’enlèvement avec les acheteurs existants"],
  ["Review any historical disputes with SAEMAPE or Ministry of Mines", "Examiner tout litige historique avec la SAEMAPE ou le ministère des Mines"],
  ["No historical disputes with SAEMAPE or the Ministry of Mines have been recorded. Minor issues have occasionally arisen due to delays in statutory payments; however, these have been resolved promptly upon settlement of the outstanding amounts and have not resulted in any material disputes.", "Aucun litige historique avec la SAEMAPE ou le ministère des Mines n’a été recensé. Des problèmes mineurs sont parfois survenus en raison de retards de paiements réglementaires ; ils ont toutefois été rapidement résolus après règlement des montants dus et n’ont entraîné aucun litige important."],
  ["Check for any environmental orders or restrictions on operations", "Vérifier l’existence de toute injonction environnementale ou restriction des opérations"],
  ["No formal environmental restrictions noted; only internal safety and mitigation measures in place.", "Aucune restriction environnementale formelle n’a été relevée ; seules des mesures internes de sécurité et d’atténuation sont en place."],
  ["Verify no sanctions applied by ITSCI or industry bodies", "Vérifier l’absence de sanctions imposées par l’ITSCI ou des organismes sectoriels"],
  ["No sanctions identified against COOMEAD", "Aucune sanction n’a été identifiée à l’encontre de COOMEAD."],
  ["Confirm concessions are not in protected areas, forest reserves, or conservation zones", "Confirmer que les concessions ne se situent pas dans des aires protégées, réserves forestières ou zones de conservation"],
  ["Concessions confirmed in ZEAs of Mambasa/Ituri; no overlap with protected or consrvation zones", "Concessions confirmées dans les ZEA de Mambasa/Ituri ; aucun chevauchement avec des zones protégées ou de conservation."],
  ["Investigate reported additional concessions in BABILA BABOMBI (BABILA TETURI grouping)", "Examiner les concessions supplémentaires signalées à BABILA BABOMBI (groupement BABILA TETURI)"],
  ["Investigate reported additional concessions in BATANGI BAU (KIPABASHI locality, North Kivu)", "Examiner les concessions supplémentaires signalées à BATANGI BAU (localité de KIPABASHI, Nord-Kivu)"],
  ["Determine if these additional concessions are included in the transaction scope or excluded", "Déterminer si ces concessions supplémentaires sont incluses ou exclues du périmètre de la transaction"],
  ["Additional concessions are yet to be part of the transaction as the submission and processing of the application with the Mining Cadastre (CAMI) and the Ministry of Mines, to secure the formal mining title are yet to be completed though customary land fees were paid already", "Les concessions supplémentaires ne font pas encore partie de la transaction, car le dépôt et le traitement de la demande auprès du Cadastre Minier (CAMI) et du ministère des Mines, afin d’obtenir le titre minier officiel, ne sont pas encore achevés, bien que les frais fonciers coutumiers aient déjà été payés."],
  ["Verify payment of all applicable surface rights (droits superficiaires)", "Vérifier le paiement de tous les droits superficiaires applicables"],
  ["Confirm payment of annual concession fees", "Confirmer le paiement des redevances annuelles de concession"],
  ["Review any Notice of Non-Compliance issued by mining authorities", "Examiner tout avis de non-conformité émis par les autorités minières"],
  ["Verify current compliance with Law N°007/2002 (Mining Code as amended)", "Vérifier la conformité actuelle avec la loi N°007/2002 (Code minier tel que modifié)"],
  ["COOMEAD holds valid ZEA permits and operates under Mining Code requirements; compliant in registration, installation, traceability, and reporting.", "COOMEAD détient des permis ZEA valides et opère conformément aux exigences du Code minier ; elle est en règle en matière d’enregistrement, d’installation, de traçabilité et de déclaration."],
  ["Confirm compliance with Decree N°038/2003 (Mining Regulations)", "Confirmer la conformité avec le décret N°038/2003 (Règlement minier)"],
  ["Compliance obligations acknowledged; implementation partial due to missing rehabilitation, technical studies", "Les obligations de conformité sont reconnues ; leur mise en œuvre est partielle en raison de l’absence de réhabilitation et d’études techniques."],
  ["Verify compliance with 2018 Mining Code amendments (Law N°18/001)", "Vérifier la conformité avec les modifications de 2018 du Code minier (loi N°18/001)"],
  ["COOMEAD compliant with 2018 Mining Code amendments in registration, oversight, traceability, and social obligations; partial implementation of environmental and technical requirements.", "COOMEAD est conforme aux modifications de 2018 du Code minier en matière d’enregistrement, de supervision, de traçabilité et d’obligations sociales ; la mise en œuvre des exigences environnementales et techniques demeure partielle."],
  ["Confirm classification as authorized artisanal cooperative", "Confirmer la classification en tant que coopérative artisanale autorisée"],
  ["Verify Cooperative decree N°21-235 of 8 August 1956 compliance", "Vérifier la conformité avec le décret coopératif N°21-235 du 8 août 1956"],
  ["COOMEAD is compliant with Cooperative Decree N°21‑235 (1956) in registration, statutes, and ministerial approval; partial evidence of member governance and reporting.", "COOMEAD est conforme au décret coopératif N°21‑235 (1956) en matière d’enregistrement, de statuts et d’agrément ministériel ; les preuves relatives à la gouvernance des membres et aux déclarations restent partielles."],
  ["Confirm compliance with Decree of 24 March 1956 relating to Indigenous Cooperatives", "Confirmer la conformité avec le décret du 24 mars 1956 relatif aux coopératives indigènes"],
  ["COOMEAD compliant with Decree of 24 March 1956; governance and reporting evidence partial.", "COOMEAD est conforme au décret du 24 mars 1956 ; les preuves de gouvernance et de déclaration sont partielles."],
  ["Verify current registration status with SAEMAPE", "Vérifier le statut actuel de l’enregistrement auprès de la SAEMAPE"],
  ["COOMEAD registered and operating under SAEMAPE oversight; status confirmed", "COOMEAD est enregistrée et opère sous la supervision de la SAEMAPE ; statut confirmé."],
  ["Obtain last 24 months of production reports submitted to SAEMAPE", "Obtenir les rapports de production des 24 derniers mois soumis à la SAEMAPE"],
  ["Confirm no outstanding compliance notices from SAEMAPE", "Confirmer l’absence d’avis de conformité en suspens de la SAEMAPE"],
  ["Review any SAEMAPE inspection reports", "Examiner tout rapport d’inspection de la SAEMAPE"],
  ["Confirm regular submission of environmental compliance reports", "Confirmer la soumission régulière des rapports de conformité environnementale"],
  ["Obtain letter of good standing from Ministry of Mines", "Obtenir une attestation de bonne situation du ministère des Mines"],
  ["Obtain letter of good standing from Ituri Provincial Mines Division", "Obtenir une attestation de bonne situation de la Division provinciale des Mines de l’Ituri"],
  ["Obtain letter from Governor of Ituri Province", "Obtenir une lettre du gouverneur de la province de l’Ituri"],
  ["Verify current relationship with Ituri Provincial Government", "Vérifier la relation actuelle avec le gouvernement provincial de l’Ituri"],
  ["Confirm North Kivu Province standing (given administrative office in Beni)", "Confirmer la situation dans la province du Nord-Kivu (compte tenu du bureau administratif à Beni)"],
  ["Verify current status with tax number A22 01031W", "Vérifier le statut actuel du numéro fiscal A22 01031W"],
  ["Confirm payment of all corporate income taxes", "Confirmer le paiement de tous les impôts sur les bénéfices des sociétés"],
  ["Verify payment of mining royalties (royalties minières)", "Vérifier le paiement des redevances minières"],
  ["Confirm payment of VAT (TVA) on applicable transactions", "Confirmer le paiement de la TVA sur les transactions applicables"],
  ["Confirm compliance with provincial and local taxes", "Confirmer la conformité avec les impôts provinciaux et locaux"],
  ["Verify no outstanding tax assessments or disputes", "Vérifier l’absence de redressements fiscaux ou de litiges en suspens"],
  ["Verify RCCM registration is current: CD/BN/RCCM/22-B-050", "Vérifier que l’immatriculation au RCCM est à jour : CD/BN/RCCM/22-B-050"],
  ["Confirm National ID: 19-B0500-N14051F is current", "Confirmer que le numéro national d’identification : 19-B0500-N14051F est à jour"],
  ["Verify tax registration: A22 01031W", "Vérifier l’immatriculation fiscale : A22 01031W"],
  ["Confirm import-export license status (noted 'démarche en cours' - in progress)", "Confirmer le statut de la licence d’import-export (mention « démarche en cours » — en cours)"],
  ["Verify all applicable business licenses are current", "Vérifier que toutes les licences commerciales applicables sont à jour"],
]);

const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const sheet = workbook.worksheets.getItem("Due Diligence Checklist");
const used = sheet.getUsedRange(false);
const values = used.values;
const missing = [];
let translatedCount = 0;

for (let row = 0; row < values.length; row++) {
  for (let col = 0; col < values[row].length; col++) {
    const value = values[row][col];
    if (typeof value !== "string" || !value.trim()) continue;
    const translated = translations.get(value);
    if (translated === undefined) {
      missing.push({ row: row + 1, col: col + 1, value });
      continue;
    }
    sheet.getCell(row, col).values = [[translated]];
    translatedCount++;
  }
}

if (missing.length) {
  throw new Error(`Untranslated text cells: ${JSON.stringify(missing)}`);
}

sheet.name = "Liste de contrôle DD";

// Keep the existing dropdown behavior while translating every displayed status.
for (const rangeAddress of ["C7:C25", "C28:C66", "C69:C95"]) {
  sheet.getRange(rangeAddress).dataValidation = {
    rule: { type: "list", values: ["☑ Vérifié", "◐ En cours", "☐ Non commencé"] },
  };
}

// French wording is often longer; preserve the source layout while preventing clipped descriptions.
for (const rangeAddress of ["B7:B25", "B28:B66", "B69:B95", "D7:D25", "D28:D66", "D69:D95"]) {
  sheet.getRange(rangeAddress).format.wrapText = true;
}
sheet.getRange("A1:D95").format.autofitRows();

await fs.mkdir(outputDir, { recursive: true });
const verification = await workbook.inspect({
  kind: "table",
  range: "'Liste de contrôle DD'!A1:D95",
  include: "values,formulas",
  tableMaxRows: 100,
  tableMaxCols: 4,
  tableMaxCellChars: 300,
});
await fs.writeFile(`${outputDir}/verification.ndjson`, verification.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
await fs.writeFile(`${outputDir}/formula-errors.ndjson`, errors.ndjson);
const preview = await workbook.render({ sheetName: "Liste de contrôle DD", autoCrop: "all", scale: 1, format: "png" });
await fs.writeFile(`${outputDir}/translated-preview.png`, new Uint8Array(await preview.arrayBuffer()));
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, translatedCount, formulaErrors: errors.ndjson }));
