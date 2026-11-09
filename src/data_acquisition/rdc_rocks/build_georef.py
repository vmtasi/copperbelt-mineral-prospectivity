import csv, json, math, os, pathlib, re, subprocess, time
from collections import defaultdict
import numpy as np
from PIL import Image

ROOT = pathlib.Path(r"C:\Users\vanmu\copperbelt-mineral-prospectivity")
DER = ROOT / "data/raw/rdc_rocks/derived"
GEO = DER / "georef"
PS = pathlib.Path(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")
BOXES = ROOT / "src/data_acquisition/rdc_rocks/ocr_boxes.ps1"

MAPS = {
 "3863": dict(img=GEO/"render/RDC_3863_p001_upright.png", scale=200000, kind="metres", step=20000,
              erange=(190000,310000), nrange=(250000,350000), title="Arc cuprifere du Katanga - Zone Centrale, structures tectoniques principales"),
 "3854": dict(img=DER/"ocr/pages/RDC_3854_p001.png", scale=500000, kind="metres", step=20000,
              erange=(190000,270000), nrange=(210000,350000), title="Facies et epaisseurs du Groupe R.4"),
 "3856": dict(img=DER/"ocr/pages/RDC_3856_p001.png", scale=50000, kind="metres", step=5000,
              erange=(510000,540000), nrange=(315000,330000), title="Roan de Tenke-Fungurume"),
 "4055": dict(img=DER/"ocr/pages/RDC_4055_p001.png", scale=250000, kind="metres", step=10000,
              erange=(150000,215000), nrange=(120000,210000), title="Region de Kolwezi, structures tectoniques principales"),
 "2376": dict(img=DER/"ocr/pages/RDC_2376_p001.png", scale=1000000, kind="degrees", step=1,
              erange=(25,29), nrange=(10,13), title="Shaba Meridional, esquisse geologique"),
 "1043": dict(img=DER/"ocr/pages/RDC_1043_p001.png", scale=500000, kind="degrees", step=1,
              erange=(26,29), nrange=(10,13), title="Carte geologique du Shaba Meridional (1936)"),
}

def ocr(img):
    out = subprocess.run([str(PS), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(BOXES),
                          "-Path", str(img)], capture_output=True, text=True, encoding="utf-8", timeout=300)
    if out.returncode != 0: raise RuntimeError(out.stderr[-300:])
    return json.loads(out.stdout)

def parse_value(text, cfg, axis):
    t = re.sub(r"^[A-Za-z]+", "", text.replace(" ", ""))
    if cfg["kind"] == "metres":
        t = t.replace(",", ".")
        m = re.match(r"^(\d{1,3})\.(\d{3})$", t)
        v = int(m.group(1))*1000+int(m.group(2)) if m else (int(t) if re.match(r"^\d{5,6}$", t) else None)
        if v is None: return None
        lo, hi = cfg["erange"] if axis == "E" else cfg["nrange"]
        return v if lo <= v <= hi and v % cfg["step"] == 0 else None
    m = re.match(r"^(\d{1,2})(?:[°ºo]\s*0{0,2})?$", t)
    if not m: return None
    v = int(m.group(1)); lo, hi = cfg["erange"] if axis == "E" else cfg["nrange"]
    return v if lo <= v <= hi else None

def edge_strips(img, side):
    W, H = img.size
    if side in ("top","bottom"):
        for frac in (0.10, 0.16, 0.22):
            y0,y1 = (0,int(H*frac)) if side=="top" else (int(H*(1-frac)),H)
            for a,b in ((0,W//2+80),(W//2-80,W)):
                yield (a,y0,b,y1), False
    else:
        for frac in (0.10, 0.16, 0.22):
            x0,x1 = (0,int(W*frac)) if side=="left" else (int(W*(1-frac)),W)
            for a,b in ((0,H//2+80),(H//2-80,H)):
                yield (x0,a,x1,b), True  # rotation tried explicitly by caller

def edge_tokens(img, side, cfg):
    vals = []
    W,H = img.size
    n = 0
    for sub, split in edge_strips(img, side):
        if sub[2] - sub[0] < 20 or sub[3] - sub[1] < 20:
            continue
        for rotate in ((False,True) if side in ("left","right") else (False,)):
            crop = img.crop(sub)
            if rotate: crop = crop.rotate(-90, expand=True)
            scale = min(2.8, 9400/max(crop.size))
            if scale < 1.0: scale = 1.0
            crop = crop.resize((int(crop.width*scale), int(crop.height*scale)), Image.Resampling.LANCZOS)
            n += 1
            p = GEO/"render"/f"_tmp_strip_{os.getpid()}_{n}.png"
            for attempt in range(3):
                try:
                    crop.save(p)
                    words = ocr(p)
                    break
                except Exception:
                    if attempt == 2: words = []
                    time.sleep(0.4)
            found = 0
            for w in words:
                v = parse_value(w["text"], cfg, "E" if side in ("top","bottom") else "N")
                if v is None: continue
                found += 1
                xc = (w["x"]+w["w"]/2)/scale; yc = (w["y"]+w["h"]/2)/scale
                if rotate:
                    ox = sub[0] + yc
                    oy = sub[1] + (sub[3]-sub[1]) - xc
                else:
                    ox = sub[0] + xc; oy = sub[1] + yc
                vals.append(dict(v=v, px=ox, py=oy, edge=side, rot=rotate, frac=round((sub[3]-sub[1])/H,2) if side in ("top","bottom") else round((sub[2]-sub[0])/W,2)))
            if found and side in ("left","right"):
                break
            try: p.unlink()
            except OSError: pass
    return vals

def fit_axis(v, p):
    v = np.asarray(v,float); p = np.asarray(p,float)
    A = np.vstack([v, np.ones_like(v)]).T
    (a,b), *_ = np.linalg.lstsq(A,p,rcond=None)
    r = p-(a*v+b)
    return float(a), float(b), r

def robust(vals):
    v = [x["v"] for x in vals]; p = [x["px"] if x["edge"] in ("top","bottom") else x["py"] for x in vals]
    a,b,r = fit_axis(v,p)
    scale = abs(a)
    rm = np.abs(r)/scale if scale else np.zeros_like(r)
    med = np.median(rm); mad = np.median(np.abs(rm-med)) or 1e-9
    keep = rm <= max(med+4*1.4826*mad, 0.5/max(scale,1e-12))
    if keep.sum() >= 4:
        v2 = [x["v"] for x,k in zip(vals,keep) if k]; p2 = [x["px"] if x["edge"] in ("top","bottom") else x["py"] for x,k in zip(vals,keep) if k]
        a,b,r = fit_axis(v2,p2)
    return a,b,r,keep

def run(mid, cfg):
    img = Image.open(cfg["img"]).convert("RGB")
    tokens = {}
    for side in ("top","bottom","left","right"):
        tokens[side] = edge_tokens(img, side, cfg)
        vals = sorted({t["v"] for t in tokens[side]})
        print(f"{mid} {side}: {len(tokens[side])} hits, values={vals}")
    east = [t for s in ("top","bottom") for t in tokens[s]]
    north = [t for s in ("left","right") for t in tokens[s]]
    if not east or not north:
        return dict(id=mid, status="insufficient", reason="no grid labels recovered", east=0, north=0)
    def best_row(items, cluster_key, value_key, tol):
        pos = [t[cluster_key] for t in items]
        order = np.argsort(pos); groups = []; cur = [order[0]]
        for i in order[1:]:
            if abs(pos[i] - pos[cur[-1]]) <= tol: cur.append(i)
            else: groups.append(cur); cur = [i]
        groups.append(cur)
        def score(g):
            return (len({items[i]["v"] for i in g}), len(g))
        g = max(groups, key=score)
        d = defaultdict(list)
        for i in g:
            d[items[i]["v"]].append(items[i][value_key])
        return {v: float(np.median(ps)) for v, ps in d.items()}
    ea = best_row(east, "py", "px", 0.012 * img.size[1])
    na = best_row(north, "px", "py", 0.012 * img.size[0])
    if mid in ("3863","3854"):
        print("   E positions:", {v: round(x,1) for v,x in sorted(ea.items())})
        print("   N positions:", {v: round(y,1) for v,y in sorted(na.items())})
    ev = sorted(ea); ep = [ea[v] for v in ev]
    nv = sorted(na); np_ = [na[v] for v in nv]
    ax, bx, rx = fit_axis(ev, ep)
    ay, by, ry = fit_axis(nv, np_)
    sx, sy = abs(ax), abs(ay)
    rmse_x = float(np.sqrt(np.mean((rx/sx)**2))); rmse_y = float(np.sqrt(np.mean((ry/sy)**2)))
    max_x = float(np.max(np.abs(rx))/sx); max_y = float(np.max(np.abs(ry))/sy)
    return dict(id=mid, status="fitted", east=len(ev), north=len(nv), scale_nominal=cfg["scale"],
                sx=sx, sy=sy, ax=ax, bx=bx, ay=ay, by=by,
                ev=ev, ep=ep, nv=nv, np=np_, rx=rx, ry=ry,
                rmse_m=float(np.hypot(rmse_x, rmse_y)), max_m=float(max(max_x,max_y)),
                rmse_x=rmse_x, rmse_y=rmse_y, max_x=max_x, max_y=max_y,
                nominal_px_per_m=200/0.0254/cfg["scale"], title=cfg["title"])

if __name__ == "__main__":
    import sys
    out=[]
    for mid in (sys.argv[1:] or list(MAPS)):
        cfg = MAPS[mid]
        print("==",mid)
        try:
            r=run(mid,cfg); out.append(r)
            if r["status"]=="fitted":
                print(f"  E{r['east']} N{r['north']} rmse={r['rmse_m']:.1f} m max={r['max_m']:.1f} m scale=({r['sx']:.5f},{r['sy']:.5f}) nominal={r['nominal_px_per_m']:.5f}")
        except Exception as e:
            print("  ERROR",type(e).__name__,e); out.append(dict(id=mid,status="error",reason=str(e)))
    (GEO/"georef_fit_raw.json").write_text(json.dumps(out,ensure_ascii=False,indent=1,default=lambda o: o.tolist() if hasattr(o,"tolist") else str(o)),encoding="utf-8")

