"""
Statistics and SuperPlots for the smFISH compartment densities (version 2).

Input : counts_per_image.csv  (written by Fish_spots_batch_v4.py into _density_results)
Output: folder 'statistics_and_plots' next to the input file:
  smFISH_master.xlsx                  all data, replicate means, all statistics, Origin-style sheets
  stats_control_vs_protease.csv       densities: control vs protease
  stats_DIV9_vs_DIV27.csv             densities: DIV9 vs DIV27 (in TIMEPOINT_CONDITIONS)
  stats_area_control_vs_protease.csv  compartment AREAS: control vs protease (quality check)
  SuperPlot_*.pdf / .png              figures (+ one file per panel in 'panels/' if enabled)

Statistics (n = biological replicate = differentiation):
  * images are summarized per replicate (REPLICATE_SUMMARY)
  * control vs protease: paired t-test on log10 replicate means (tests ratio = 1)
  * DIV9 vs DIV27: paired t-test per probe where all replicates share differentiations,
    Welch t-test for probes listed in NOT_SAME_DIFFERENTIATION
  * area check: paired t-test on log10 replicate mean areas, control vs protease
  * multiple-testing correction optional (MULTIPLE_TESTING_CORRECTION, default off)
  Reported: ratio (geometric mean ratio) with 95% CI and p.

Density = number of spots in the compartment / compartment area, per image.
DENSITY_FACTOR only rescales the unit (100 -> spots per 100 um^2); statistics are unaffected.

THE LOOK OF ALL PLOTS IS SET IN THE 'STYLE' BLOCK BELOW.
"""
import os
import numpy as np
try:
    import statsmodels.api  # noqa: F401  (needed for the robust linear models)
except ImportError:
    raise SystemExit("Please install statsmodels first:  pip install statsmodels")
import pandas as pd
from scipy import stats

# =============================================================================
# DATA AND ANALYSIS SETTINGS
# =============================================================================
INPUT_FILE = None              # None -> file dialog; or r"...\_density_results\counts_per_image.csv"

PROBE_ORDER = ['PINK1', 'BNIP3L', 'BNIP3', 'FIS1', 'MFF', 'OPA1']
COMPARTMENT_ORDER = ['Soma', 'Dendrite', 'Axon']
TIMEPOINT_ORDER = ['DIV9', 'DIV27']
CONDITION_ORDER = ['control', 'protease']

REPLICATE_SUMMARY = 'mean'     # images -> replicate value: 'mean' or 'median'

# DIV9 vs DIV27: paired t-test for probes whose DIV9 and DIV27 samples come from the SAME
# differentiation in every replicate; unpaired (Welch) t-test for probes listed here, i.e. where
# at least one replicate's DIV9 sample was redone from a DIFFERENT differentiation.
# Format: (probe, replicate). CHECK THIS LIST AGAINST YOUR NOTES.
NOT_SAME_DIFFERENTIATION = [

]

# Two analyses are always run and saved in separate subfolders:
#  'robust_linear_model' : images averaged per condition x differentiation; per compartment
#                          (and timepoint/condition) ONE robust linear model across all probes:
#                              value ~ probe * group + differentiation   (differentiation = fixed effect)
#                          followed by emmeans-style contrasts of the group within each probe
#  't_tests'             : per probe, t-test on the replicate means (paired / Welch, log10 scale)
RLM_TRANSFORM = 'none'         # 'none' = untransformed means (as in the MASS::rlm description)
                               # 'log'  = log10(mean) -> multiplicative effects
RLM_NORM = 'HuberT'            # MASS::rlm default psi.huber (k = 1.345) = statsmodels HuberT
RLM_DF = 'residual'            # p-values of the contrasts: 'residual' = t-test with the model's residual df
                               # 'asymptotic' = z-test (df = Inf). If your colleague's emmeans output shows
                               # "df = Inf", use 'asymptotic' to match it exactly.
# Differentiation identity: by default the replicate number (n1, n2, n3). Samples redone from another
# differentiation (NOT_SAME_DIFFERENTIATION, DIV9 samples) get their own differentiation level.

MULTIPLE_TESTING_CORRECTION = False   # False = report uncorrected p-values (state this in the methods)
                                      # True  = Benjamini-Hochberg within each comparison type
TIMEPOINT_CONDITIONS = ['control', 'protease']   # conditions in which DIV9 vs DIV27 is tested,
                                      # e.g. ['protease'] or ['control', 'protease']
DENSITY_FACTOR = 1           # 100 -> spots per 100 um^2 ; 1 -> spots per um^2
DENSITY_LABEL = 'Spot density (per µm²)'
AREA_LABEL = 'Compartment area (µm²)'
AREA_FLAG_LIMITS = (0.8, 1.25)  # area check: flag protease/control area ratios outside this range
                                # (a 20% change in area changes the density by 20%)

# =============================================================================
# STYLE - edit freely
# =============================================================================
STYLE = dict(
    # --- figure ---
    panel_width=4.2,             # inches per panel
    panel_height=3.2,
    font='Arial',
    font_size=9,
    title_size=10,
    y_scale='linear',            # 'linear' or 'log' (log: images with 0 spots are not shown)
    y_min=0,                     # None = automatic (linear scale only)
    # --- zooming in on the data (y axis) ---
    y_top_percentile=None,       # e.g. 97: top of the axis at the 97th percentile of the images in each
                                 # panel; the few images above it are drawn as small triangles on the top
                                 # edge (not removed; statistics unchanged). None = show everything
    y_max=None,                  # fixed top of the y axis for every panel (in data units), e.g. 40;
                                 # overrides y_top_percentile. None = automatic
    show_images=True,            # False = plot only the replicate means (the axis then fits the means)
    # --- your own y axis per panel (overrides the automatic settings above) ---
    # key = panel title as shown on the plot, e.g. 'Soma | DIV27' (applies to all DENSITY figures with
    # that title), or 'figure name/panel title' for one figure only (needed for the AREA figure), e.g.
    #   'SuperPlot_density_control_vs_protease/Soma | DIV27'
    # value = (bottom, top) of the data range; brackets/stars are placed above it automatically
    y_limits={
        # 'Soma | DIV27': (0, 150),
        # 'Dendrite | DIV27': (0, 40),
    },
    # tick spacing: one number for all panels (e.g. 25), or per panel like y_limits; None = automatic
    y_tick_step=None,
    save_formats=['pdf', 'png'],
    dpi=300,
    save_individual_panels=True, # also save every panel as its own file (folder 'panels')

    # --- colours / symbols (SuperPlot: colour = replicate) ---
    replicate_colors={'n1': '#1b9e77', 'n2': '#d95f02', 'n3': '#7570b3'},
    replicate_markers={'n1': 'o', 'n2': 's', 'n3': '^'},

    # --- single images (small dots) ---
    image_dot_size=6,
    image_dot_alpha=0.35,
    image_jitter=0.12,

    # --- replicate means (large dots) ---
    mean_dot_size=55,
    mean_dot_edgecolor='black',
    mean_dot_edgewidth=0.8,
    connect_paired=True,         # lines between paired replicate means
    connect_line_width=1.0,
    connect_line_alpha=0.8,

    # --- summary (mean of replicate means +/- error) ---
    summary_error='sd',          # 'sd', 'sem' or 'none'
    summary_line_color='black',
    summary_line_width=1.5,
    summary_bar_halfwidth=0.18,
    error_cap_halfwidth=0.08,

    # --- layout ---
    group_spacing=1.0,           # distance between probes
    condition_offset=0.22,       # distance of the two groups from the probe centre
    group_labels={'control': 'Ctrl', 'protease': 'Prot', 'DIV9': 'early', 'DIV27': 'late'},
    tick_label_size=7,
    probe_label_size=9,

    # --- legend ---
    legend_location='upper right',   # e.g. 'upper right', 'lower center', or None

    # --- significance ---
    show_significance='stars',   # 'stars', 'p', 'both' (stars + p) or 'none'
    star_levels=[(0.001, '***'), (0.01, '**'), (0.05, '*')],
    show_ns=True,                # write 'ns' for non-significant comparisons
    p_decimals=3,
    bracket_line_width=0.8,
    bracket_height=0.04,         # size of the bracket ends, fraction of the y range
    significance_font_size=8,
)
# =============================================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory


def setup_matplotlib():
    plt.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': [STYLE['font'], 'Arial', 'Helvetica', 'DejaVu Sans'],
                         'font.size': STYLE['font_size'], 'axes.titlesize': STYLE['title_size'],
                         'pdf.fonttype': 42, 'ps.fonttype': 42,     # editable text in Illustrator
                         'axes.spines.top': False, 'axes.spines.right': False})


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def load(path):
    d = pd.read_csv(path)
    need = {'probe', 'timepoint', 'replicate', 'condition', 'compartment', 'image_id',
            'density_per_um2', 'area_um2'}
    missing = need - set(d.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns {missing} - is this counts_per_image.csv from v4?")
    d['density'] = d['density_per_um2'] * DENSITY_FACTOR
    d['area'] = d['area_um2']
    return d


def replicate_means(d, value):
    f = 'mean' if REPLICATE_SUMMARY == 'mean' else 'median'
    out = (d.groupby(['probe', 'timepoint', 'replicate', 'condition', 'compartment'])
            .agg(value=(value, f), n_images=('image_id', 'nunique')).reset_index())
    return out.rename(columns={'value': value})


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------
def bh(p):
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum():
        q = p[ok]; o = np.argsort(q); m = len(q)
        adj = np.minimum.accumulate((q[o] * m / np.arange(1, m + 1))[::-1])[::-1]
        res = np.empty(m); res[o] = np.minimum(adj, 1)
        out[ok] = res
    return out


def paired_log_test(a, b):
    """a, b: dict replicate -> value. Paired test of log10(b/a) = 0."""
    reps = sorted(r for r in set(a) & set(b) if a[r] > 0 and b[r] > 0)
    x = np.array([np.log10(b[r]) - np.log10(a[r]) for r in reps])
    n = len(x)
    if n < 2:
        return dict(n=n, ratio=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan)
    m, s = x.mean(), x.std(ddof=1)
    if s == 0:
        return dict(n=n, ratio=10 ** m, ci_low=np.nan, ci_high=np.nan, p=np.nan)
    tcrit = stats.t.ppf(0.975, n - 1)
    return dict(n=n, ratio=10 ** m, ci_low=10 ** (m - tcrit * s / np.sqrt(n)),
                ci_high=10 ** (m + tcrit * s / np.sqrt(n)), p=stats.ttest_1samp(x, 0).pvalue)


def welch_log_test(a, b):
    xa = np.log10([v for v in a.values() if v > 0]); xb = np.log10([v for v in b.values() if v > 0])
    if len(xa) < 2 or len(xb) < 2:
        return dict(n=f"{len(xa)}/{len(xb)}", ratio=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan)
    m = xb.mean() - xa.mean()
    va, vb = xa.var(ddof=1) / len(xa), xb.var(ddof=1) / len(xb)
    se = np.sqrt(va + vb)
    if se == 0:
        return dict(n=f"{len(xa)}/{len(xb)}", ratio=10 ** m, ci_low=np.nan, ci_high=np.nan, p=np.nan)
    df = (va + vb) ** 2 / (va ** 2 / (len(xa) - 1) + vb ** 2 / (len(xb) - 1))
    tcrit = stats.t.ppf(0.975, df)
    return dict(n=f"{len(xa)}/{len(xb)}", ratio=10 ** m, ci_low=10 ** (m - tcrit * se),
                ci_high=10 ** (m + tcrit * se), p=stats.ttest_ind(xb, xa, equal_var=False).pvalue)


def differentiation_id(probe, replicate, timepoint):
    redo = {(p.upper(), r) for p, r in NOT_SAME_DIFFERENTIATION}
    if timepoint == TIMEPOINT_ORDER[0] and (str(probe).upper(), replicate) in redo:
        return f"{replicate}_{probe}_redo"
    return replicate


def rlm_description():
    t = 'log10(mean)' if RLM_TRANSFORM == 'log' else 'mean'
    dft = 't-tests with residual df' if RLM_DF == 'residual' else 'asymptotic z-tests (df = Inf)'
    return (f"images averaged per condition and differentiation; robust linear model (statsmodels RLM, "
            f"{RLM_NORM}, equivalent to MASS::rlm default) per compartment across all probes: "
            f"{t} ~ probe * group + differentiation (fixed effect); contrasts of the group within each probe "
            f"(emmeans-style), {dft}")


def rlm_pooled(rm, value, group_col, ref, other, split_cols):
    """One robust linear model per split (e.g. compartment x timepoint) across all probes on the
    replicate means:  value ~ probe * group + differentiation.  Returns one row per probe with the
    contrast other - ref (and ratio), 95% CI and p."""
    import statsmodels.api as sm
    from scipy import stats as st_
    rows = []
    for key, g in rm.groupby(split_cols):
        key = key if isinstance(key, tuple) else (key,)
        g = g[g[group_col].isin([ref, other]) & g[value].notna()].reset_index(drop=True)
        if RLM_TRANSFORM == 'log':
            g = g[g[value] > 0].reset_index(drop=True)
        probes = [p for p in PROBE_ORDER if p in set(g.probe)]
        if len(g) < 6 or len(probes) == 0 or g[group_col].nunique() < 2:
            continue
        y = np.log10(g[value]) if RLM_TRANSFORM == 'log' else g[value].astype(float)
        grp = (g[group_col] == other).astype(float)
        X = pd.DataFrame({'const': 1.0, 'group': grp})
        for p in probes[1:]:
            ind = (g.probe == p).astype(float)
            X[f'probe_{p}'] = ind
            X[f'group_x_{p}'] = ind * grp
        X = pd.concat([X, pd.get_dummies(g.diff_id, prefix='diff', drop_first=True).astype(float)], axis=1)
        # drop columns that make the design singular (e.g. a differentiation seen in one cell only)
        keep = []
        for c in X.columns:
            if np.linalg.matrix_rank(X[keep + [c]].to_numpy()) == len(keep) + 1:
                keep.append(c)
        X = X[keep]
        fit = sm.RLM(y, X, M=getattr(sm.robust.norms, RLM_NORM)()).fit()
        V = fit.cov_params()
        df_res = len(y) - X.shape[1]
        for p in probes:
            L = pd.Series(0.0, index=X.columns)
            if 'group' not in L:
                continue
            L['group'] = 1.0
            if f'group_x_{p}' in L:
                L[f'group_x_{p}'] = 1.0
            est = float(L @ fit.params)
            se = float(np.sqrt(L @ V @ L))
            if RLM_DF == 'residual' and df_res > 0:
                pval = 2 * st_.t.sf(abs(est / se), df_res); crit = st_.t.ppf(0.975, df_res)
            else:
                pval = 2 * st_.norm.sf(abs(est / se)); crit = 1.96
            gp = g[g.probe == p]
            ref_mean = gp.loc[gp[group_col] == ref, value].mean()
            if RLM_TRANSFORM == 'log':
                ratio, lo, hi = 10 ** est, 10 ** (est - crit * se), 10 ** (est + crit * se)
            else:
                ratio, lo, hi = [(ref_mean + x) / ref_mean if ref_mean else np.nan
                                 for x in (est, est - crit * se, est + crit * se)]
            rows.append(dict(**dict(zip(split_cols, key)), probe=p,
                             estimate=est, se=se, df=df_res if RLM_DF == 'residual' else np.inf,
                             ratio=ratio, ci_low=lo, ci_high=hi, p=pval,
                             n=f"{len(gp)} means / {gp.diff_id.nunique()} differentiations"))
    return pd.DataFrame(rows)


def add_diff_ids(rm):
    rm = rm.copy()
    rm['diff_id'] = [differentiation_id(p, r, t) for p, r, t in zip(rm.probe, rm.replicate, rm.timepoint)]
    return rm


def test_conditions(rm, value, method):
    """control vs protease within every probe x timepoint x compartment."""
    rows = []
    if method == 'rlm':
        res = rlm_pooled(add_diff_ids(rm), value, 'condition', 'control', 'protease', ['compartment', 'timepoint'])
        idx = res.set_index(['probe', 'timepoint', 'compartment']) if len(res) else None
    for (probe, tp, comp), g in rm.groupby(['probe', 'timepoint', 'compartment']):
        a = g[g.condition == 'control'].set_index('replicate')[value].to_dict()
        b = g[g.condition == 'protease'].set_index('replicate')[value].to_dict()
        if method == 'rlm':
            if idx is None or (probe, tp, comp) not in idx.index:
                continue
            r = idx.loc[(probe, tp, comp)]
            test = rlm_description()
        else:
            r = paired_log_test(a, b)
            test = 'paired t-test on log10 replicate means'
        rows.append(dict(probe=probe, timepoint=tp, compartment=comp,
                         mean_control=np.mean(list(a.values())) if a else np.nan,
                         mean_protease=np.mean(list(b.values())) if b else np.nan,
                         test=test, n=r['n'], ratio_protease_vs_control=r['ratio'],
                         ci95_low=r['ci_low'], ci95_high=r['ci_high'], p=r['p']))
    s = pd.DataFrame(rows)
    if len(s) and MULTIPLE_TESTING_CORRECTION:
        s['p_adjusted_BH'] = bh(s['p'])
    return s


def flag_areas(s):
    """With n = 3 a real area change is often not significant, so the check is judged by the size
    of the change: flag ratios outside AREA_FLAG_LIMITS."""
    if not len(s):
        s['area_flag'] = []
        return s
    lo, hi = AREA_FLAG_LIMITS
    r = s['ratio_protease_vs_control']
    s['area_flag'] = np.where(r.lt(lo) | r.gt(hi), f'CHECK: area changed by more than {int(round((1 - lo) * 100))}%', '')
    return s


def test_timepoints(rm, value, method):
    rows = []
    t0, t1 = TIMEPOINT_ORDER[0], TIMEPOINT_ORDER[-1]
    if method == 'rlm':
        sub = rm[rm.condition.isin(TIMEPOINT_CONDITIONS)]
        res = rlm_pooled(add_diff_ids(sub), value, 'timepoint', t0, t1, ['compartment', 'condition'])
        idx = res.set_index(['probe', 'condition', 'compartment']) if len(res) else None
    for (probe, cond, comp), g in rm.groupby(['probe', 'condition', 'compartment']):
        if cond not in TIMEPOINT_CONDITIONS:
            continue
        a = g[g.timepoint == t0].set_index('replicate')[value].to_dict()
        b = g[g.timepoint == t1].set_index('replicate')[value].to_dict()
        if method == 'rlm':
            if idx is None or (probe, cond, comp) not in idx.index:
                continue
            r = idx.loc[(probe, cond, comp)]
            test = rlm_description()
        else:
            paired = timepoints_paired(probe)
            r = paired_log_test(a, b) if paired else welch_log_test(a, b)
            test = (('paired t-test (same differentiations)' if paired else
                     'Welch t-test (unpaired: not all replicates from the same differentiation)')
                    + ' on log10 replicate means')
        rows.append(dict(probe=probe, condition=cond, compartment=comp,
                         **{f'mean_{t0}': np.mean(list(a.values())) if a else np.nan,
                            f'mean_{t1}': np.mean(list(b.values())) if b else np.nan},
                         test=test, n=r['n'], **{f'ratio_{t1}_vs_{t0}': r['ratio']},
                         ci95_low=r['ci_low'], ci95_high=r['ci_high'], p=r['p']))
    s = pd.DataFrame(rows)
    if len(s) and MULTIPLE_TESTING_CORRECTION:
        s['p_adjusted_BH'] = bh(s['p'])
    return s


def timepoints_paired(probe):
    return not any(p.upper() == str(probe).upper() for p, _ in NOT_SAME_DIFFERENTIATION)


def p_column():
    return 'p_adjusted_BH' if MULTIPLE_TESTING_CORRECTION else 'p'


# -----------------------------------------------------------------------------
# SuperPlots
# -----------------------------------------------------------------------------
def p_text(p):
    """Label above a bracket according to STYLE['show_significance']."""
    S = STYLE
    if not np.isfinite(p) or S['show_significance'] == 'none':
        return ''
    stars = next((s for lvl, s in S['star_levels'] if p < lvl), 'ns')
    if stars == 'ns' and not S['show_ns']:
        return ''
    small = 10 ** (-S['p_decimals'])
    ptxt = f"p = {p:.{S['p_decimals']}f}" if p >= small else f"p < {small:.{S['p_decimals']}f}"
    if S['show_significance'] == 'stars':
        return stars
    if S['show_significance'] == 'p':
        return ptxt
    return stars if stars == 'ns' else f"{stars}\n{ptxt}"          # 'both'


_NOTES = set()


def lookup_panel(table, plot_name, title):
    """'figure/panel' keys apply to that figure; plain panel-title keys apply to the DENSITY figures only
    (the area figures have the same panel titles but other units)."""
    if not isinstance(table, dict):
        return None
    if f"{plot_name}/{title}" in table:
        return table[f"{plot_name}/{title}"]
    return None if 'AREA' in plot_name else table.get(title)


def draw_panel(ax, sub, subm, st_rows, value, group_col, groups, paired, probes, ylabel, title, plot_name=''):
    """One SuperPlot panel: probes on x, the two groups next to each other."""
    S = STYLE
    rng = np.random.default_rng(0)
    ticks, ticklabels = [], []
    brackets = []
    # top of the data region: your own limits, fixed, percentile of the images, or everything
    manual = lookup_panel(S['y_limits'], plot_name, title)
    cap = None
    if manual is not None:
        cap = float(manual[1])
    elif S['y_max'] is not None:
        cap = float(S['y_max'])
    elif S['y_top_percentile'] is not None and S['show_images'] and len(sub):
        cap = float(np.nanpercentile(sub[value], S['y_top_percentile']))
    mean_top = subm[value].max() if len(subm) else np.nan
    if cap is not None and np.isfinite(mean_top) and mean_top > cap:
        if manual is not None and (plot_name, title) not in _NOTES:
            _NOTES.add((plot_name, title))
            print(f"   note: {plot_name} '{title}': a replicate mean ({mean_top:.3g}) is above your y limit "
                  f"{cap:g} - axis extended to show it")
        cap = mean_top * 1.05                           # never hide a replicate mean
    for k, probe in enumerate(probes):
        centre = k * S['group_spacing']
        xs = {grp: centre + (2 * gi - (len(groups) - 1)) * S['condition_offset']
              for gi, grp in enumerate(groups)}
        for grp in groups:
            x0 = xs[grp]
            ticks.append(x0); ticklabels.append(S['group_labels'].get(grp, grp))
            img = sub[(sub.probe == probe) & (sub[group_col] == grp)]
            for rep, gi in (img.groupby('replicate') if S['show_images'] else []):
                jit = x0 + rng.uniform(-S['image_jitter'], S['image_jitter'], len(gi))
                v = gi[value].to_numpy()
                inside = v <= cap if cap is not None else np.ones(len(v), bool)
                col = S['replicate_colors'].get(rep, 'grey')
                ax.scatter(jit[inside], v[inside], s=S['image_dot_size'], alpha=S['image_dot_alpha'],
                           color=col, linewidths=0, zorder=1)
                if (~inside).any():                         # images above the axis: triangles at the top
                    ax.scatter(jit[~inside], np.full((~inside).sum(), cap), marker='^',
                               s=S['image_dot_size'] * 2.5, alpha=min(1, S['image_dot_alpha'] * 2),
                               color=col, linewidths=0, zorder=1, clip_on=False)
            mm = subm[(subm.probe == probe) & (subm[group_col] == grp)]
            for _, r in mm.iterrows():
                ax.scatter(x0, r[value], s=S['mean_dot_size'], marker=S['replicate_markers'].get(r.replicate, 'o'),
                           color=S['replicate_colors'].get(r.replicate, 'grey'),
                           edgecolors=S['mean_dot_edgecolor'], linewidths=S['mean_dot_edgewidth'], zorder=4)
            if len(mm):
                mu = mm[value].mean()
                err = 0
                if len(mm) > 1 and S['summary_error'] != 'none':
                    err = mm[value].std(ddof=1) / (np.sqrt(len(mm)) if S['summary_error'] == 'sem' else 1)
                hw, cw, lw = S['summary_bar_halfwidth'], S['error_cap_halfwidth'], S['summary_line_width']
                ax.plot([x0 - hw, x0 + hw], [mu, mu], color=S['summary_line_color'], lw=lw, zorder=3)
                if err and np.isfinite(err):
                    ax.plot([x0, x0], [mu - err, mu + err], color=S['summary_line_color'], lw=lw * 0.7, zorder=3)
                    for yv in (mu - err, mu + err):
                        ax.plot([x0 - cw, x0 + cw], [yv, yv], color=S['summary_line_color'], lw=lw * 0.7, zorder=3)
        if (paired if not callable(paired) else paired(probe)) and S['connect_paired']:
            mp = subm[subm.probe == probe].pivot_table(index='replicate', columns=group_col, values=value)
            for rep, r in mp.iterrows():
                if all(g in r.index and np.isfinite(r[g]) for g in groups):
                    ax.plot([xs[g] for g in groups], [r[g] for g in groups],
                            color=S['replicate_colors'].get(rep, 'grey'), lw=S['connect_line_width'],
                            alpha=S['connect_line_alpha'], zorder=2)
        row = st_rows[st_rows.probe == probe] if len(st_rows) else st_rows
        if len(row):
            p = row[p_column()].iloc[0]
            lab = p_text(p)
            vals = sub[sub.probe == probe][value] if S['show_images'] else subm[subm.probe == probe][value]
            if lab and len(vals):
                top = np.nanmax(vals) if cap is None else min(np.nanmax(vals), cap)
                if not S['show_images']:
                    mm_all = subm[subm.probe == probe]
                    err = mm_all.groupby(group_col)[value].std(ddof=1).max() if len(mm_all) > 2 else 0
                    top = top + (err if np.isfinite(err) else 0)
                brackets.append((xs[groups[0]], xs[groups[-1]], centre, top, lab))

    # axes limits first, then brackets placed in data units above the data
    if S['show_images']:
        data_top = np.nanmax(sub[value]) if len(sub) else 1
    else:
        data_top = np.nanmax(subm[value]) * 1.15 if len(subm) else 1
    if cap is not None:
        data_top = min(data_top, cap)
    top_data = max([b[3] for b in brackets] + [data_top])
    ax.set_yscale(S['y_scale'])
    if S['y_scale'] == 'linear':
        span = top_data - (S['y_min'] or 0)
        extra = 0.30 if S['show_significance'] == 'both' else 0.18
        ax.set_ylim(S['y_min'] if S['y_min'] is not None else None, top_data + span * (extra + 0.05))
        for x1, x2, centre, ytop, lab in brackets:
            y = ytop + span * 0.05
            h = span * S['bracket_height']
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color='black', lw=S['bracket_line_width'])
            ax.text(centre, y + h * 1.2, lab, ha='center', va='bottom', fontsize=S['significance_font_size'],
                    linespacing=1.0)
    else:
        for x1, x2, centre, ytop, lab in brackets:
            y = ytop * 1.25
            ax.plot([x1, x1, x2, x2], [y, y * 1.08, y * 1.08, y], color='black', lw=S['bracket_line_width'])
            ax.text(centre, y * 1.12, lab, ha='center', va='bottom', fontsize=S['significance_font_size'])
        ax.set_ylim(top=top_data * 3)

    if S['y_scale'] == 'linear':
        lo = float(manual[0]) if manual is not None else ax.get_ylim()[0]
        ax.set_ylim(bottom=lo)
        step = lookup_panel(S['y_tick_step'], plot_name, title) if isinstance(S['y_tick_step'], dict) \
            else S['y_tick_step']
        if step:
            top_tick = cap if cap is not None else data_top
            ax.set_yticks(np.arange(lo, top_tick + step * 0.001, step))
        if manual is not None or step:
            # axis line ends at the last tick (clean look; the bracket area above stays free)
            yt = [t for t in ax.get_yticks() if lo <= t <= (cap if cap is not None else ax.get_ylim()[1])]
            if yt:
                ax.spines['left'].set_bounds(lo, max(yt))

    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, fontsize=S['tick_label_size'])
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for k, probe in enumerate(probes):
        ax.text(k * S['group_spacing'], -0.16, probe, transform=tr, ha='center', va='top',
                fontsize=S['probe_label_size'], fontweight='bold')
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def legend_handles(reps):
    return [Line2D([], [], marker=STYLE['replicate_markers'].get(r, 'o'), ls='',
                   color=STYLE['replicate_colors'].get(r, 'grey'), markeredgecolor='black',
                   markersize=7, label=f"replicate {r}") for r in reps]


def save(fig, base):
    for fmt in STYLE['save_formats']:
        fig.savefig(f"{base}.{fmt}", dpi=STYLE['dpi'], bbox_inches='tight')
    plt.close(fig)


def superplot(d, rm, st, value, ylabel, group_col, groups, row_col, rows, col_col, cols, paired, out_dir, name,
              title):
    S = STYLE
    probes = [p for p in PROBE_ORDER if p in set(d.probe)]
    reps = sorted(set(d.replicate))
    if not rows or not cols or not groups:
        return
    fig, axs = plt.subplots(len(rows), len(cols), figsize=(S['panel_width'] * len(cols),
                                                           S['panel_height'] * len(rows)), squeeze=False)
    panel_dir = os.path.join(out_dir, 'panels')
    for i, rv in enumerate(rows):
        for j, cv in enumerate(cols):
            sub = d[(d[row_col] == rv) & (d[col_col] == cv)]
            subm = rm[(rm[row_col] == rv) & (rm[col_col] == cv)]
            st_rows = st[(st[row_col] == rv) & (st[col_col] == cv)] \
                if len(st) and row_col in st and col_col in st else pd.DataFrame()
            args = (sub, subm, st_rows, value, group_col, groups, paired, probes,
                    ylabel, f"{rv} | {cv}", name)
            draw_panel(axs[i, j], *args)
            if j > 0:
                axs[i, j].set_ylabel('')
            if S['save_individual_panels']:
                os.makedirs(panel_dir, exist_ok=True)
                f1, a1 = plt.subplots(figsize=(S['panel_width'], S['panel_height']))
                draw_panel(a1, *args)
                if S['legend_location']:                  # outside the plot: never covers data
                    a1.legend(handles=legend_handles(reps), loc='upper left', bbox_to_anchor=(1.01, 1.0),
                              frameon=False, fontsize=S['font_size'] - 1)
                save(f1, os.path.join(panel_dir, f"{name}_{rv}_{cv}"))
    if S['legend_location']:
        fig.legend(handles=legend_handles(reps), loc=S['legend_location'], frameon=False)
    fig.suptitle(title, fontsize=S['title_size'] + 1)
    fig.tight_layout(rect=(0, 0.02, 0.92, 0.95))
    save(fig, os.path.join(out_dir, name))


# -----------------------------------------------------------------------------
# Excel master file
# -----------------------------------------------------------------------------
def origin_wide(df, value, col_parts):
    cols = {}
    for key, g in df.groupby(col_parts):
        cols['_'.join(map(str, key))] = g[value].reset_index(drop=True)
    return pd.DataFrame(cols)


def methods_table(method):
    if method == 'rlm':
        cond = time = rlm_description() + ('; DIV9 samples redone from another differentiation are coded as '
                                           'their own differentiation level')
    else:
        cond = 'paired t-test on log10 replicate means, per probe'
        time = ('paired t-test on log10 replicate means for probes whose DIV9 and DIV27 samples come from the same '
                'differentiations (' + ', '.join(p for p in PROBE_ORDER if timepoints_paired(p)) +
                '); Welch t-test for ' + ', '.join(p for p in PROBE_ORDER if not timepoints_paired(p)))
    return pd.DataFrame({'setting': ['density', 'value per differentiation', 'control vs protease (density and area)',
                                     'DIV9 vs DIV27', 'DIV comparison tested in', 'multiple testing'],
                         'value': [f'spots / compartment area per image, x{DENSITY_FACTOR}',
                                   f'{REPLICATE_SUMMARY} of the images of each condition and differentiation',
                                   cond, time, ', '.join(TIMEPOINT_CONDITIONS),
                                   'Benjamini-Hochberg within each comparison type' if MULTIPLE_TESTING_CORRECTION
                                   else 'no correction for multiple comparisons']})


def write_excel(path, d, rm_dens, rm_area, s_cond, s_time, s_area, method):
    with pd.ExcelWriter(path) as xw:
        d.to_excel(xw, sheet_name='per_image', index=False)
        rm_dens.to_excel(xw, sheet_name='replicate_means_density', index=False)
        rm_area.to_excel(xw, sheet_name='replicate_means_area', index=False)
        s_cond.to_excel(xw, sheet_name='stats_density_ctrl_vs_prot', index=False)
        s_time.to_excel(xw, sheet_name='stats_density_DIV', index=False)
        s_area.to_excel(xw, sheet_name='stats_AREA_ctrl_vs_prot', index=False)
        for comp in COMPARTMENT_ORDER:
            origin_wide(d[d.compartment == comp], 'density', ['probe', 'timepoint', 'condition']) \
                .to_excel(xw, sheet_name=f'Origin_images_{comp}'[:31], index=False)
            origin_wide(rm_dens[rm_dens.compartment == comp], 'density', ['probe', 'timepoint', 'condition']) \
                .to_excel(xw, sheet_name=f'Origin_repmeans_{comp}'[:31], index=False)
        methods_table(method).to_excel(xw, sheet_name='methods', index=False)


def comparison_table(res):
    """Both analyses side by side, one row per comparison."""
    out = []
    for kind, keys, ratio_col in [('density: control vs protease', ['probe', 'timepoint', 'compartment'],
                                   'ratio_protease_vs_control'),
                                  (f'density: {TIMEPOINT_ORDER[0]} vs {TIMEPOINT_ORDER[-1]}',
                                   ['probe', 'condition', 'compartment'],
                                   f'ratio_{TIMEPOINT_ORDER[-1]}_vs_{TIMEPOINT_ORDER[0]}'),
                                  ('AREA: control vs protease', ['probe', 'timepoint', 'compartment'],
                                   'ratio_protease_vs_control')]:
        a = res['t_tests'][kind]; b = res['robust_linear_model'][kind]
        if not len(a) or not len(b):
            continue
        m = a[keys + [ratio_col, 'p']].merge(b[keys + [ratio_col, 'ci95_low', 'ci95_high', 'p']], on=keys,
                                            how='outer', suffixes=('_ttest', '_rlm'))
        m.insert(0, 'comparison', kind)
        m = m.rename(columns={f'{ratio_col}_ttest': 'ratio_ttest', f'{ratio_col}_rlm': 'ratio_rlm',
                              'ci95_low': 'ci95_low_rlm', 'ci95_high': 'ci95_high_rlm'})
        m['significant_ttest'] = m['p_ttest'] < 0.05
        m['significant_rlm'] = m['p_rlm'] < 0.05
        out.append(m)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def run_analysis(method, d, rm_d, rm_a, out):
    os.makedirs(out, exist_ok=True)
    s_cond = test_conditions(rm_d, 'density', method)
    s_time = test_timepoints(rm_d, 'density', method)
    s_area = flag_areas(test_conditions(rm_a, 'area', method))
    s_cond.to_csv(os.path.join(out, 'stats_control_vs_protease.csv'), index=False)
    s_time.to_csv(os.path.join(out, 'stats_DIV9_vs_DIV27.csv'), index=False)
    s_area.to_csv(os.path.join(out, 'stats_area_control_vs_protease.csv'), index=False)
    write_excel(os.path.join(out, 'smFISH_master.xlsx'), d, rm_d, rm_a, s_cond, s_time, s_area, method)

    comps = [c for c in COMPARTMENT_ORDER if c in set(d.compartment)]
    tps = [t for t in TIMEPOINT_ORDER if t in set(d.timepoint)]
    conds = [c for c in CONDITION_ORDER if c in set(d.condition)]
    tag = 'robust linear model' if method == 'rlm' else 't-tests'
    superplot(d, rm_d, s_cond, 'density', DENSITY_LABEL, 'condition', conds, 'compartment', comps,
              'timepoint', tps, True, out, 'SuperPlot_density_control_vs_protease',
              f'Spot density: control vs protease ({tag})')
    superplot(d, rm_d, s_time, 'density', DENSITY_LABEL, 'timepoint', tps, 'compartment', comps,
              'condition', [c for c in conds if c in TIMEPOINT_CONDITIONS], timepoints_paired, out,
              'SuperPlot_density_DIV9_vs_DIV27', (f'Spot density: {tps[0]} vs {tps[-1]} ({tag})') if tps else '')
    superplot(d, rm_a, s_area, 'area', AREA_LABEL, 'condition', conds, 'compartment', comps,
              'timepoint', tps, True, out, 'SuperPlot_AREA_control_vs_protease',
              f'Quality check - compartment area: control vs protease ({tag})')

    def n_sig(s):
        return int((s[p_column()] < 0.05).sum()) if len(s) else 0
    print(f"\n=== {tag.upper()} ===")
    print("Test:", rlm_description() if method == 'rlm' else 't-tests on replicate means, per probe')
    print(f"Density control vs protease : {len(s_cond)} comparisons, {n_sig(s_cond)} significant (p < 0.05)")
    print(f"Density {TIMEPOINT_ORDER[0]} vs {TIMEPOINT_ORDER[-1]} ({', '.join(TIMEPOINT_CONDITIONS)}): "
          f"{len(s_time)} comparisons, {n_sig(s_time)} significant")
    flagged = s_area[s_area.area_flag != ''] if len(s_area) else s_area
    print(f"AREA control vs protease    : {len(s_area)} comparisons, {n_sig(s_area)} significant, "
          f"{len(flagged)} with an area change outside {AREA_FLAG_LIMITS[0]}-{AREA_FLAG_LIMITS[1]}x")
    for _, r in flagged.iterrows():
        print(f"   CHECK {r.probe} {r.timepoint} {r.compartment}: protease/control area = "
              f"{r.ratio_protease_vs_control:.2f} (95% CI {r.ci95_low:.2f}-{r.ci95_high:.2f})")
    return {'density: control vs protease': s_cond,
            f'density: {TIMEPOINT_ORDER[0]} vs {TIMEPOINT_ORDER[-1]}': s_time,
            'AREA: control vs protease': s_area}


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    path = INPUT_FILE
    if path is None:
        from tkinter import Tk, filedialog
        tk = Tk(); tk.withdraw(); tk.attributes('-topmost', True)
        path = filedialog.askopenfilename(parent=tk, title='Select counts_per_image.csv',
                                          filetypes=[('CSV', '*.csv')])
        tk.destroy()
        if not path:
            raise SystemExit("No file selected.")
    out = os.path.join(os.path.dirname(os.path.abspath(path)), 'statistics_and_plots')
    os.makedirs(out, exist_ok=True)
    setup_matplotlib()

    d = load(path)
    rm_d = replicate_means(d, 'density')
    rm_a = replicate_means(d, 'area')
    print(f"Images: {d.image_id.nunique()} | replicates: {', '.join(sorted(set(d.replicate)))} | "
          f"p-values {'BH-adjusted' if MULTIPLE_TESTING_CORRECTION else 'uncorrected'}")
    res = {'t_tests': run_analysis('t', d, rm_d, rm_a, os.path.join(out, 't_tests')),
           'robust_linear_model': run_analysis('rlm', d, rm_d, rm_a, os.path.join(out, 'robust_linear_model'))}
    comp = comparison_table(res)
    comp.to_csv(os.path.join(out, 'COMPARISON_ttest_vs_robust_linear_model.csv'), index=False)
    with pd.ExcelWriter(os.path.join(out, 'COMPARISON_ttest_vs_robust_linear_model.xlsx')) as xw:
        comp.to_excel(xw, sheet_name='both_methods', index=False)
        pd.concat([methods_table('t').assign(analysis='t-tests'),
                   methods_table('rlm').assign(analysis='robust linear model')]) \
            .to_excel(xw, sheet_name='methods', index=False)
    print(f"\nResults in: {out}")
    print("  t_tests\\ and robust_linear_model\\ : tables, Excel master and SuperPlots of each analysis")
    print("  COMPARISON_ttest_vs_robust_linear_model.xlsx : both analyses side by side")
