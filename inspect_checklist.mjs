import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";
import fs from "node:fs/promises";

const inputPath = "C:/Users/vanmu/OneDrive/Documents/Checklist ver3.xlsx";
const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);
await fs.mkdir("outputs/checklist-fr", { recursive: true });
const preview = await workbook.render({ sheetName: "Due Diligence Checklist", autoCrop: "all", scale: 1, format: "png" });
await fs.writeFile("outputs/checklist-fr/original-preview.png", new Uint8Array(await preview.arrayBuffer()));

const summary = await workbook.inspect({
  kind: "workbook,sheet,table,formula",
  maxChars: 12000,
  tableMaxRows: 200,
  tableMaxCols: 30,
  tableMaxCellChars: 300,
  options: { maxResults: 1000 },
});
console.log(summary.ndjson);
for (const sheet of workbook.worksheets.items) {
  const used = sheet.getUsedRange(false);
  const values = used.values;
  const formulas = used.formulas;
  console.log(`SHEET:${sheet.name} USED:${used.address}`);
  for (let r = 0; r < values.length; r++) {
    for (let c = 0; c < values[r].length; c++) {
      const value = values[r][c];
      const formula = formulas?.[r]?.[c];
      if (typeof value === "string" && value.trim()) console.log(`TEXT ${sheet.name}!R${r+1}C${c+1}: ${JSON.stringify(value)}`);
      if (formula) console.log(`FORMULA ${sheet.name}!R${r+1}C${c+1}: ${formula}`);
    }
  }
}
