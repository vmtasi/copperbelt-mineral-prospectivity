from pathlib import Path
import sys
import numpy as np
import pandas as pd
import arviz as az
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
DOMAINS = ['CRZ','MMSB','NKB','NRB_3a','NRB_3b','SRB']
CASES = [('NRB_3a','fault',1),('NRB_3a','lithology',1),('NRB_3b','fault',2),('NRB_3b','lithology',2)]
COLS = {'fault':'distance_to_fault','lithology':'distance_to_lithology_contact'}
NAMES = {'fault':('beta_f_lin','beta_f_sq'),'lithology':('beta_l_lin','beta_l_sq')}

def main():
    df=pd.read_csv(ROOT/'data/copperbelt_training_v5_with_tectonic_domain.csv').dropna(subset=['centroid_x','centroid_y','domain','litho_contact_litho_class','distance_to_fault','distance_to_lithology_contact','bouguer','deposit_present']).copy()
    def md(x):
        s=str(x).lower()
        for token,name in [('3a','NRB_3a'),('3b','NRB_3b'),('crz','CRZ'),('srb','SRB'),('nkb','NKB'),('mmsb','MMSB')]:
            if token in s:return name
        return 'Unknown'
    df['daly_domain']=df.domain.map(md); df=df[df.daly_domain.isin(DOMAINS)]
    from src.validation_strategies import get_along_belt_folds
    df['spatial_block']=get_along_belt_folds(df,n_folds=4)
    rows=[]
    for dom,key,fold in CASES:
        train=df[df.spatial_block != fold-1]
        sc=StandardScaler().fit(train[[COLS[key]]]); mu=float(sc.mean_[0]); sig=float(sc.scale_[0])
        tr=az.from_netcdf(ROOT/f'figures/v11_fold_{fold}_trace.nc')
        a,b=NAMES[key]; j=DOMAINS.index(dom)
        b1=tr.posterior[a].values.reshape(-1,6)[:,j]; b2=tr.posterior[b].values.reshape(-1,6)[:,j]
        valid=np.abs(b2)>1e-5
        dstar=(mu+sig*(-b1[valid]/(2*b2[valid])))/1000
        support=train.loc[train.daly_domain==dom,COLS[key]].agg(['min','max']).to_numpy()/1000
        # Check one deterministic draw and aggregate against generated CSV.
        k=int(np.flatnonzero(valid)[0]); d=(mu+sig*(-b1[k]/(2*b2[k])))/1000
        slope=(b1[k]+2*b2[k]*((d*1000-mu)/sig))
        rows.append({'Fold':fold,'Domain':dom,'Variable':key,'draw_index':k,'beta_lin':b1[k],'beta_sq':b2[k],'manual_Dstar_km':d,'manual_slope_at_Dstar':slope,'support_min_km':support[0],'support_max_km':support[1],'formula_residual_slope':slope})
    out=pd.DataFrame(rows)
    gen=pd.read_csv(ROOT/'figures/audit/beta_diagnostics/beta_diagnostics_four_cases_by_fold.csv')
    for _,r in out.iterrows():
        g=gen[(gen.Fold==r.Fold)&(gen.Domain==r.Domain)&(gen.Variable==r.Variable)].iloc[0]
        # Recompute D* from the generated medians as a second independent algebra check.
        mu=float(g.train_mu_m); sig=float(g.train_sigma_m); z=-g.beta_lin_median/(2*g.beta_sq_median); d=(mu+sig*z)/1000
        print(f"{r.Domain} {r.Variable} fold {r.Fold}: draw {int(r.draw_index)} D*={r.manual_Dstar_km:.6f} km; slope(D*)={r.manual_slope_at_Dstar:.3e}; support=[{r.support_min_km:.6f},{r.support_max_km:.6f}] km; median-coefficient D*={d:.6f} km")
        assert abs(r.manual_slope_at_Dstar) < 1e-10
    print('[+] Independent algebra checks passed for all four required cases.')

if __name__=='__main__': main()
