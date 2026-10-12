"""
Dendrite vs axon comparison + compartment areas / confluence check.

Needs smFISH_statistics_superplots.py (latest version) in the SAME folder - the statistics and
plots are shared, so everything looks and is computed like in the main statistics.

Input : counts_per_image.csv (from Fish_spots_batch_v4.py, folder _density_results)
Output: folder 'compartments_and_areas' next to the input file

PART 1 - DENDRITE vs AXON spot density, for every probe x timepoint x condition
  * t-test : paired t-test on log10 replicate means (dendrite and axon come from the same images)
  * robust linear model (colleague's method): images averaged per compartment and differentiation;
             per timepoint x condition ONE model across all probes:
                 mean ~ probe * compartment + differentiation (fixed effect)
             then the dendrite-vs-axon contrast within each probe
  Settings of the robust model (transformation, norm, df) are taken from the statistics script.

PART 2 - COMPARTMENT AREAS (confluence)
  a) per sample (probe x replicate x condition): median area per image of soma, dendrite and axon,
     and how far it is from the median of all samples at that timepoint -> flag if outside AREA_FLAG_FOLD
  b) which factors explain area differences between samples (ANOVA on log10 sample medians):
     differentiation, probe, condition
  c) does confluence influence the mRNA density?
     - robust linear model  log10(density) ~ pair + condition + log10(axon area),
       pair = probe x replicate (same differentiation). The area effect is therefore estimated from
       the confluence differences WITHIN each control/protease pair, so differences between
       differentiations cannot create a false association (simulated false-positive rate 3-5%;
       a plain between-sample model gave 16-19%).
       -> 'elasticity': 0 = no influence, 1 = density rises in proportion to the axon area
     - within samples: Spearman correlation of axon area and density over the images of each sample
"""
import os
import inspect
import numpy as np
import pandas as pd
from scipy import stats

import smFISH_statistics_superplots as SP

# =============================================================================
INPUT_FILE = None                         # None -> file dialog
COMPARE = ('Dendrite', 'Axon')            # density ratio reported as first / second
CONDITIONS = ['control', 'protease']      # conditions in which dendrite vs axon is tested
CONFLUENCE_COMPARTMENT = 'Axon'           # axon mask = all non-dendritic neurites in the field
AREA_FLAG_FOLD = 1.5                      # flag samples whose median area is > 1.5x or < 1/1.5x of the
                                          # median of all samples at the same timepoint
# =============================================================================

if 'plot_name' not in inspect.signature(SP.draw_panel).parameters or not hasattr(SP, 'rlm_pooled'):
    raise SystemExit("Please use the latest smFISH_statistics_superplots.py in the same folder as this script.")

import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# PART 1: dendrite vs axon
# -----------------------------------------------------------------------------
def dendrite_vs_axon(d):
    first, second = COMPARE
    dd = d[d.compartment.isin(COMPARE) & d.condition.isin(CONDITIONS)]
    rm = SP.replicate_means(dd, 'density')
    # t-test per probe x timepoint x condition
    rows = []
    for (probe, tp, cond), g in rm.groupby(['probe', 'timepoint', 'condition']):
        a = g[g.compartment == second].set_index('replicate')['density'].to_dict()
        b = g[g.compartment == first].set_index('replicate')['density'].to_dict()
        r = SP.paired_log_test(a, b)
        rows.append(dict(probe=probe, timepoint=tp, condition=cond,
                         **{f'mean_{first}': np.mean(list(b.values())) if b else np.nan,
                            f'mean_{second}': np.mean(list(a.values())) if a else np.nan},
                         test='paired t-test on log10 replicate means (same images)', n=r['n'],
                         **{f'ratio_{first}_vs_{second}': r['ratio']},
                         ci95_low=r['ci_low'], ci95_high=r['ci_high'], p=r['p']))
    t = pd.DataFrame(rows)
    # robust linear model across probes per timepoint x condition
    res = SP.rlm_pooled(SP.add_diff_ids(rm), 'density', 'compartment', second, first, ['timepoint', 'condition'])
    rows = []
    for _, r in res.iterrows():
        g = rm[(rm.probe == r.probe) & (rm.timepoint == r.timepoint) & (rm.condition == r.condition)]
        rows.append(dict(probe=r.probe, timepoint=r.timepoint, condition=r.condition,
                         **{f'mean_{first}': g[g.compartment == first].density.mean(),
                            f'mean_{second}': g[g.compartment == second].density.mean()},
                         test=SP.rlm_description().replace('group', 'compartment'), n=r.n,
                         **{f'ratio_{first}_vs_{second}': r.ratio},
                         ci95_low=r.ci_low, ci95_high=r.ci_high, p=r.p))
    m = pd.DataFrame(rows)
    return dd, rm, t, m


# -----------------------------------------------------------------------------
# PART 2: areas
# -----------------------------------------------------------------------------
SAMPLE = ['timepoint', 'probe', 'replicate', 'condition']


def area_per_sample(d):
    s = (d.groupby(SAMPLE + ['compartment'])
          .agg(n_images=('image_id', 'nunique'), median_area_um2=('area', 'median'),
               q25=('area', lambda x: x.quantile(0.25)), q75=('area', lambda x: x.quantile(0.75)),
               mean_density=('density', 'mean')).reset_index())
    tp_med = s.groupby(['timepoint', 'compartment'])['median_area_um2'].transform('median')
    s['fold_vs_timepoint_median'] = s['median_area_um2'] / tp_med
    s['flag'] = np.where((s.fold_vs_timepoint_median > AREA_FLAG_FOLD) |
                         (s.fold_vs_timepoint_median < 1 / AREA_FLAG_FOLD),
                         f'CHECK: area differs > {AREA_FLAG_FOLD}x from the other samples', '')
    return s


def area_factors(s):
    """Two-way ANOVA (type II) on log10 sample medians: which factor explains area differences?"""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    rows = []
    for (tp, comp), g in s.groupby(['timepoint', 'compartment']):
        g = g[g.median_area_um2 > 0].copy()
        g['y'] = np.log10(g.median_area_um2)
        g['diff'] = [SP.differentiation_id(p, r, t) for p, r, t in zip(g.probe, g.replicate, g.timepoint)]
        terms = [t for t, col in [('C(diff)', 'diff'), ('C(probe)', 'probe'), ('C(condition)', 'condition')]
                 if g[col].nunique() > 1]
        if len(g) < len(terms) + 3 or not terms:
            continue
        try:
            a = anova_lm(smf.ols('y ~ ' + ' + '.join(terms), g).fit(), typ=2)
        except Exception:
            continue
        cv = g.median_area_um2.std(ddof=1) / g.median_area_um2.mean()
        row = dict(timepoint=tp, compartment=comp, n_samples=len(g),
                   cv_between_samples=cv, fold_range=g.median_area_um2.max() / g.median_area_um2.min())
        for t in terms:
            name = {'C(diff)': 'differentiation', 'C(probe)': 'probe', 'C(condition)': 'condition'}[t]
            row[f'p_{name}'] = a.loc[t, 'PR(>F)']
        rows.append(row)
    return pd.DataFrame(rows)


def confluence_effect(d, s):
    """Does the axon area (confluence) influence the density of each compartment?"""
    import statsmodels.api as sm
    conf = s[s.compartment == CONFLUENCE_COMPARTMENT].set_index(SAMPLE)['median_area_um2']
    rows = []
    for (tp, comp), g in s.groupby(['timepoint', 'compartment']):
        g = g.set_index(SAMPLE).join(conf.rename('conf_area')).reset_index()
        g = g[(g.mean_density > 0) & (g.conf_area > 0)]
        row = dict(timepoint=tp, compartment=comp, confluence_measure=f'{CONFLUENCE_COMPARTMENT} area',
                   n_samples=len(g))
        # log density ~ pair + condition + log confluence (robust); pair = probe x replicate
        g = g.reset_index(drop=True)
        pair = g.probe.astype(str) + '_' + g.replicate.astype(str)
        X = pd.get_dummies(pair, prefix='pair', drop_first=True).astype(float)
        if g.condition.nunique() > 1:
            X['protease'] = (g.condition == 'protease').astype(float).to_numpy()
        X['log10_area'] = np.log10(g.conf_area.to_numpy())
        X = sm.add_constant(X, has_constant='add')
        df_res = len(g) - X.shape[1]
        if df_res >= 3 and np.linalg.matrix_rank(X.to_numpy()) == X.shape[1]:
            fit = sm.RLM(np.log10(g.mean_density.to_numpy()), X, M=getattr(sm.robust.norms, SP.RLM_NORM)()).fit()
            b, se = fit.params['log10_area'], fit.bse['log10_area']
            row.update(elasticity=b, ci95_low=b - stats.t.ppf(0.975, df_res) * se,
                       ci95_high=b + stats.t.ppf(0.975, df_res) * se,
                       p=2 * stats.t.sf(abs(b / se), df_res), n_pairs=pair.nunique())
        # within samples: Spearman over images
        rhos = []
        di = d[d.timepoint == tp]
        ca = di[di.compartment == CONFLUENCE_COMPARTMENT].set_index(['image_id'])['area']
        dc = di[di.compartment == comp].set_index('image_id')
        for _, gi in dc.groupby(SAMPLE):
            x = ca.reindex(gi.index); y = gi['density']
            ok = x.notna() & y.notna()
            if ok.sum() >= 6 and x[ok].nunique() > 2 and y[ok].nunique() > 2:
                rhos.append(stats.spearmanr(x[ok], y[ok]).statistic)
        if rhos:
            k = int(np.sum(np.array(rhos) > 0))
            row.update(median_spearman_within_samples=float(np.median(rhos)), n_samples_with_rho=len(rhos),
                       samples_with_positive_rho=k,
                       p_sign_test_within=stats.binomtest(k, len(rhos), 0.5).pvalue)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_areas(d, s, out):
    S = SP.STYLE
    for comp in [c for c in SP.COMPARTMENT_ORDER if c in set(d.compartment)]:
        tps = [t for t in SP.TIMEPOINT_ORDER if t in set(d.timepoint)]
        fig, axs = plt.subplots(len(tps), 1, figsize=(14, 3.4 * len(tps)), squeeze=False)
        for ax, tp in zip(axs[:, 0], tps):
            ss = s[(s.timepoint == tp) & (s.compartment == comp)].copy()
            ss['po'] = ss.probe.map({p: i for i, p in enumerate(SP.PROBE_ORDER)})
            ss = ss.sort_values(['po', 'replicate', 'condition']).reset_index(drop=True)
            rng = np.random.default_rng(0)
            for x, r in ss.iterrows():
                img = d[(d.timepoint == tp) & (d.compartment == comp) & (d.probe == r.probe) &
                        (d.replicate == r.replicate) & (d.condition == r.condition)]['area']
                col = S['replicate_colors'].get(r.replicate, 'grey')
                ax.scatter(x + rng.uniform(-0.2, 0.2, len(img)), img, s=6, alpha=0.4, color=col, linewidths=0)
                ax.plot([x - 0.3, x + 0.3], [r.median_area_um2] * 2, color='black', lw=1.5)
                if r.flag:
                    ax.text(x, ax.get_ylim()[1] if False else img.max() * 1.05, '!', ha='center', color='red',
                            fontweight='bold')
            med = ss.median_area_um2.median()
            ax.axhline(med, color='grey', ls='--', lw=1)
            ax.axhspan(med / AREA_FLAG_FOLD, med * AREA_FLAG_FOLD, color='grey', alpha=0.08)
            ax.set_xticks(range(len(ss)))
            ax.set_xticklabels([f"{r.probe}\n{r.replicate}\n{'C' if r.condition == 'control' else 'P'}"
                                for r in ss.itertuples()], fontsize=6)
            ax.set_ylabel(f'{comp} area per image (µm²)')
            ax.set_title(f'{comp} | {tp}: black bar = median per sample; dashed = median of all samples; '
                         f'band = {AREA_FLAG_FOLD}x range; red ! = outside the band', fontsize=9)
            ax.set_ylim(bottom=0)
        fig.tight_layout()
        for fmt in S['save_formats']:
            fig.savefig(os.path.join(out, f'AREA_per_sample_{comp}.{fmt}'), dpi=S['dpi'], bbox_inches='tight')
        plt.close(fig)


def plot_confluence(d, s, ce, out):
    S = SP.STYLE
    comps = [c for c in SP.COMPARTMENT_ORDER if c in set(s.compartment)]
    tps = [t for t in SP.TIMEPOINT_ORDER if t in set(s.timepoint)]
    conf = s[s.compartment == CONFLUENCE_COMPARTMENT].set_index(SAMPLE)['median_area_um2']
    cmap = plt.get_cmap('tab10')
    pcol = {p: cmap(i) for i, p in enumerate(SP.PROBE_ORDER)}
    fig, axs = plt.subplots(len(comps), len(tps), figsize=(4.6 * len(tps), 3.6 * len(comps)), squeeze=False)
    for i, comp in enumerate(comps):
        for j, tp in enumerate(tps):
            ax = axs[i, j]
            g = s[(s.timepoint == tp) & (s.compartment == comp)].set_index(SAMPLE).join(
                conf.rename('conf')).reset_index()
            for p, gp in g.groupby('probe'):
                ax.scatter(gp.conf, gp.mean_density, s=28, color=pcol.get(p, 'grey'),
                           marker='o', edgecolors='black', linewidths=0.4, label=p)
            ax.set_xscale('log'); ax.set_yscale('log')
            r = ce[(ce.timepoint == tp) & (ce.compartment == comp)]
            if len(r) and 'elasticity' in r and np.isfinite(r.iloc[0].get('elasticity', np.nan)):
                r = r.iloc[0]
                ax.text(0.02, 0.98, f"elasticity {r.elasticity:.2f} "
                                    f"(95% CI {r.ci95_low:.2f} to {r.ci95_high:.2f})\np = {r.p:.3f}",
                        transform=ax.transAxes, va='top', fontsize=7)
            ax.set_title(f'{comp} density | {tp}', fontsize=9)
            ax.set_xlabel(f'{CONFLUENCE_COMPARTMENT} area per image, sample median (µm²)', fontsize=8)
            ax.set_ylabel(SP.DENSITY_LABEL, fontsize=8)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper right', frameon=False, fontsize=8)
    fig.suptitle('Does confluence influence spot density? One dot = one sample (probe x replicate x condition). '
                 'Elasticity from log10(density) ~ pair + condition + log10(area), i.e. within control/protease '
                 'pairs: 0 = no influence.', fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.92, 0.95))
    for fmt in S['save_formats']:
        fig.savefig(os.path.join(out, f'CONFLUENCE_vs_density.{fmt}'), dpi=S['dpi'], bbox_inches='tight')
    plt.close(fig)


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    path = INPUT_FILE
    if path is None:
        from tkinter import Tk, filedialog
        tk = Tk(); tk.withdraw(); tk.attributes('-topmost', True)
        path = filedialog.askopenfilename(parent=tk, title='Select counts_per_image.csv', filetypes=[('CSV', '*.csv')])
        tk.destroy()
        if not path:
            raise SystemExit("No file selected.")
    out = os.path.join(os.path.dirname(os.path.abspath(path)), 'compartments_and_areas')
    os.makedirs(out, exist_ok=True)
    SP.setup_matplotlib()
    SP.STYLE['group_labels'].setdefault('Dendrite', 'Dend'); SP.STYLE['group_labels'].setdefault('Axon', 'Axon')
    d = SP.load(path)

    # ---- part 1
    dd, rm, t, m = dendrite_vs_axon(d)
    first, second = COMPARE
    ratio = f'ratio_{first}_vs_{second}'
    both = t[['probe', 'timepoint', 'condition', ratio, 'p']].merge(
        m[['probe', 'timepoint', 'condition', ratio, 'ci95_low', 'ci95_high', 'p']],
        on=['probe', 'timepoint', 'condition'], how='outer', suffixes=('_ttest', '_rlm'))
    both['significant_ttest'] = both.p_ttest < 0.05
    both['significant_rlm'] = both.p_rlm < 0.05
    conds = [c for c in SP.CONDITION_ORDER if c in CONDITIONS and c in set(d.condition)]
    tps = [x for x in SP.TIMEPOINT_ORDER if x in set(d.timepoint)]
    for tag, st, sub in [('ttest', t, 't_tests'), ('rlm', m, 'robust_linear_model')]:
        o = os.path.join(out, sub); os.makedirs(o, exist_ok=True)
        st.to_csv(os.path.join(o, f'stats_{first}_vs_{second}.csv'), index=False)
        SP.superplot(dd, rm, st, 'density', SP.DENSITY_LABEL, 'compartment', [first, second],
                     'condition', conds, 'timepoint', tps, True, o, f'SuperPlot_{first}_vs_{second}',
                     f'{first} vs {second} ({"t-tests" if tag == "ttest" else "robust linear model"})')

    # ---- part 2
    s = area_per_sample(d)
    fac = area_factors(s)
    ce = confluence_effect(d, s)
    plot_areas(d, s, out)
    plot_confluence(d, s, ce, out)

    with pd.ExcelWriter(os.path.join(out, 'compartments_and_areas.xlsx')) as xw:
        both.to_excel(xw, sheet_name=f'{first}_vs_{second}_both', index=False)
        t.to_excel(xw, sheet_name=f'{first}_vs_{second}_ttest'[:31], index=False)
        m.to_excel(xw, sheet_name=f'{first}_vs_{second}_rlm'[:31], index=False)
        s.to_excel(xw, sheet_name='area_per_sample', index=False)
        fac.to_excel(xw, sheet_name='area_which_factor', index=False)
        ce.to_excel(xw, sheet_name='confluence_vs_density', index=False)
        pd.DataFrame({'item': ['dendrite vs axon, t-test', 'dendrite vs axon, robust model', 'area per sample',
                               'area: which factor', 'confluence vs density (within pairs)',
                               'confluence vs density (within samples)'],
                      'method': ['paired t-test on log10 replicate means (dendrite and axon from the same images)',
                                 SP.rlm_description().replace('group', 'compartment'),
                                 f'median area per image; flagged if > {AREA_FLAG_FOLD}x from the timepoint median',
                                 'two-way ANOVA (type II) on log10 sample medians: differentiation + probe + condition',
                                 f'robust linear model log10(mean density) ~ pair (probe x replicate) + condition + '
                                 f'log10(median {CONFLUENCE_COMPARTMENT} area); slope = elasticity, estimated within '
                                 f'control/protease pairs',
                                 'Spearman correlation over the images of each sample; sign test across samples']}) \
            .to_excel(xw, sheet_name='methods', index=False)

    # ---- summary on screen
    print(f"=== {first} vs {second} ===")
    for tp in tps:
        for c in conds:
            b = both[(both.timepoint == tp) & (both.condition == c)]
            print(f"  {tp} {c:9s}: t-test {int(b.significant_ttest.sum())}/{len(b)} significant | "
                  f"robust model {int(b.significant_rlm.sum())}/{len(b)} significant | "
                  f"median {first}/{second} ratio {b[f'{ratio}_ttest'].median():.2f}")
    print("\n=== Areas ===")
    for _, r in fac.iterrows():
        ps = ', '.join(f"{k[2:]} p={r[k]:.3f}" for k in r.index if k.startswith('p_') and pd.notna(r[k]))
        print(f"  {r.compartment:8s} {r.timepoint}: between-sample CV {r.cv_between_samples:.0%}, "
              f"{r.fold_range:.1f}-fold range | {ps}")
    fl = s[s.flag != '']
    print(f"  {len(fl)} samples flagged (area > {AREA_FLAG_FOLD}x from the timepoint median) - see AREA_per_sample_*.pdf")
    print("\n=== Does confluence (axon area) influence the density? ===")
    for _, r in ce.iterrows():
        if pd.notna(r.get('elasticity', np.nan)):
            print(f"  {r.compartment:8s} {r.timepoint}: elasticity {r.elasticity:.2f} "
                  f"(95% CI {r.ci95_low:.2f} to {r.ci95_high:.2f}), p = {r.p:.3f}")
    print(f"\nResults in: {out}")
