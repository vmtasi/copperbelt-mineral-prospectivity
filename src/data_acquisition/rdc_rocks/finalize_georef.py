import csv, json, math, pathlib
import numpy as np
ROOT=pathlib.Path(r"C:\Users\vanmu\copperbelt-mineral-prospectivity")
GEO=ROOT/"data/raw/rdc_rocks/derived/georef"
raw=json.loads((GEO/"georef_fit_raw.json").read_text(encoding="utf-8"))
fits={r["id"]:r for r in raw}

EXTRA={
 "4055":("insufficient","10 km metric grid is legible on the plate, but the edge labels are too small/faint for OCR; no reliable automated control points. Manual digitising required."),
 "2376":("insufficient","1-degree graticule is visible, but only one longitude label was recovered automatically; latitude labels were not recovered. Manual digitising required."),
 "1043":("insufficient","Only the 28 deg E and 12 deg S graticule labels are legible; too few control points. Manual digitising required."),
 "1317":("insufficient","Second scan of the same 1936 map; only the 28 deg E / 12 deg S labels are visible and the lower margin is cropped."),
 "379":("not_georeferenceable","The PDF contains the explanatory notice only. All 59 pages are portrait text/figures; the 1:100,000 map plate is absent."),
 "3129":("not_planimetric","Stratigraphic correlation section (2 mm = 1 m) with depth logs, not a plan map. No plan coordinates."),
}
REASON={
 "3854":"Only two easting and two northing grid labels survived OCR; a line through two points has no residual degrees of freedom and the northing spacing is inconsistent with the nominal scale. Not defensible.",
}

# isotropic fit helper
def iso(ev,ep,nv,npix):
    E=np.array(ev,float); x=np.array(ep,float); N=np.array(nv,float); y=np.array(npix,float)
    Ec=E-E.mean(); Nc=N-N.mean(); xc=x-x.mean(); yc=y-y.mean()
    s=(Ec@xc+Nc@yc)/(Ec@Ec+Nc@Nc)
    bx=x.mean()-s*E.mean(); by=y.mean()-s*N.mean()
    rx=x-(s*E+bx); ry=y-(s*N+by)
    return s,bx,by,rx,ry

gcp_rows=[]; comp_rows=[]; params={}
for mid,r in fits.items():
    if r.get("status")!="fitted" or mid not in ("3863","3856"):
        continue
    sx,sy=r["sx"],r["sy"]
    for i,(v,p,res) in enumerate(zip(r["ev"],r["ep"],r["rx"]),1):
        gcp_rows.append(dict(map_id=mid,point_id=f"{mid}-E{i:02d}",axis="E",grid_value=int(v),pixel_position=round(p,1),edge="top/bottom",method="OCR label anchor (axis fit)",residual_px=round(res,2),residual_m=round(res/sx,1),included="yes"))
    for i,(v,p,res) in enumerate(zip(r["nv"],r["np"],r["ry"]),1):
        gcp_rows.append(dict(map_id=mid,point_id=f"{mid}-N{i:02d}",axis="N",grid_value=int(v),pixel_position=round(p,1),edge="left/right",method="OCR label anchor (axis fit)",residual_px=round(res,2),residual_m=round(res/sy,1),included="yes"))
    s,bx,by,rx,ry=iso(r["ev"],r["ep"],r["nv"],r["np"])
    comp_rows.append(dict(map_id=mid,model="axis-aligned independent scale",n_e=len(r["ev"]),n_n=len(r["nv"]),params=4,rmse_m=round(r["rmse_m"],1),max_m=round(r["max_m"],1),scale_x=round(sx,5),scale_y=round(sy,5),scale_ratio=round(sx/sy,4),note="preferred"))
    iso_rmse=float(np.sqrt(np.mean(np.r_[(rx/s)**2,(ry/s)**2])))
    comp_rows.append(dict(map_id=mid,model="isotropic common scale",n_e=len(r["ev"]),n_n=len(r["nv"]),params=3,rmse_m=round(iso_rmse,1),max_m=round(float(max(np.abs(rx).max()/s,np.abs(ry).max()/s)),1),scale_x=round(s,5),scale_y=round(s,5),scale_ratio=1.0,note="rejected" if abs(sx-sy)/sx>0.02 else "comparable"))
    params[mid]=dict(title=r["title"],status="defensible in grid coordinates; datum UNRESOLVED",
        nominal_scale=r["scale_nominal"], raster_dpi=200,
        transform={"x_px":"a*E + b","y_px":"c*N + d","a":round(r["ax"],8),"b":round(r["bx"],2),"c":round(r["ay"],8),"d":round(r["by"],2)},
        grid_extent={"E_min":int(min(r["ev"])),"E_max":int(max(r["ev"])),"N_min":int(min(r["nv"])),"N_max":int(max(r["nv"]))},
        control={"eastings":len(r["ev"]),"northings":len(r["nv"]),"rmse_m":round(r["rmse_m"],1),"max_m":round(r["max_m"],1)},
        nominal_px_per_m=round(r["nominal_px_per_m"],6), fitted_px_per_m={"x":round(sx,6),"y":round(sy,6)},
        crs="UNRESOLVED: no datum/projection declaration. Metric grid treated as a local projected grid; do not label WGS84.",
        note="Scan is 180 deg rotated; upright image used." if mid=="3863" else "Only two northing grid lines exist; Y residual is not independently testable, but fitted scale matches nominal within 2%.")

for mid,(st,reason) in EXTRA.items():
    params[mid]=dict(title=mid,status=st,reason=reason,crs="UNRESOLVED")

with (GEO/"gcp_control_points_v1.csv").open("w",encoding="utf-8-sig",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(gcp_rows[0])); w.writeheader(); w.writerows(gcp_rows)
with (GEO/"transform_comparison_v1.csv").open("w",encoding="utf-8-sig",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(comp_rows[0])); w.writeheader(); w.writerows(comp_rows)
(GEO/"transform_params_v1.json").write_text(json.dumps(params,ensure_ascii=False,indent=1),encoding="utf-8")

md=["# Georeferencing control-point and residual report (v1)","",
"Pixel-to-grid fits use grid labels as control: easting labels constrain x(E), northing labels constrain y(N).",
"All coordinates are **map-grid metres**, not geographic coordinates. No datum or projection is declared on any sheet; WGS84 was not assumed.","",
"| Map | Scale | E controls | N controls | RMSE (m) | Max (m) | Fitted px/m x / y | Nominal px/m | Verdict |",
"|---|---|---|---|---|---|---|---|---|"]
for mid,r in fits.items():
    if mid in params and params[mid].get("status","").startswith("defensible"):
        md.append(f"| {mid} | 1:{r['scale_nominal']:,} | {len(r['ev'])} | {len(r['nv'])} | {r['rmse_m']:.1f} | {r['max_m']:.1f} | {r['sx']:.5f} / {r['sy']:.5f} | {r['nominal_px_per_m']:.5f} | defensible in grid coords |")
for mid,(st,reason) in EXTRA.items():
    md.append(f"| {mid} | - | - | - | - | - | - | - | {st} |")
if "3854" in fits:
    md.append("| 3854 | 1:500,000 | 2 | 2 | n/a | n/a | 0.01612 / 0.03899 | 0.01575 | insufficient |")
md += ["","## Interpretation", "",
"- **3863** is the only regional structural map with a defensible automated fit: 6 easting and 5 northing control lines, RMSE 135 m at 1:200,000, fitted scale within 1% of the nominal scale. It can support structural digitising in grid metres, but cannot be published as WGS84.",
"- **3856** has a consistent scale (fitted 0.1570 px/m vs nominal 0.1575), but only two northing grid lines (320 and 325 km). The Y fit is therefore a two-point calibration, not a tested residual surface.",
"- **3854, 4055, 2376, 1043 and 1317** are flagged insufficient: the grids are visible but the labels could not be recovered reliably enough for a defensible fit. They need manual control-point digitising on the page images.",
"- **379** does not contain the 1:100,000 map plate; only the notice. **3129** is a depth section, not a plan map.",
"","## Residual distribution", "",
"Control points are distributed along the four sheet margins (eastings on top/bottom rows, northings on left/right columns). No independent interior check points were available, so the residuals test grid consistency, not internal cartographic distortion. This limitation is why the fits are labelled 'defensible in grid coordinates' rather than 'survey-grade'."]
(GEO/"georeferencing_report_v1.md").write_text("\n".join(md)+"\n",encoding="utf-8")
print("gcps",len(gcp_rows),"comparisons",len(comp_rows),"params",len(params))
print(json.dumps({k:v["status"] for k,v in params.items()},indent=1))
