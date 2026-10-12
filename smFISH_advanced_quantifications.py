"""
Advanced smFISH quantifications - uses the finished results of Fish_spots_batch_v4.py (no refitting).

Needs in the SAME folder:  Fish_spots_batch_v4.py        (masks: same settings incl. dilation)
                           smFISH_statistics_superplots.py (statistics + SuperPlots)
Select the top folder ("biological replicates"); results go to _density_results/advanced_quantifications/

1. DISTANCE FROM THE SOMA along the dendrites (path through the dendrite mask, not a straight line)
   - density profile: counted dendritic spots per um2 of dendrite in distance bins (BIN_UM)
   - 'distal index' = mean distance of the spots / mean distance of all dendrite pixels
                      (1 = spread like the dendrite itself, > 1 = more distal, < 1 = more proximal)
   - median distance of the dendritic spots from the soma (um)
2. LOCALIZATION INDICES (independent of how well a probe labels)
   - dendrite/soma and dendrite/axon spot-density ratio per image
3. mRNAs PER SPOT
   - single-mRNA intensity per control/protease pair = most frequent integrated intensity of the
     counted neurite spots (both conditions pooled, so control and protease use the same unit)
   - per compartment: mean mRNAs per spot, fraction of multi-mRNA spots (>= MULTI_COPY_THRESHOLD units),
     mRNA copies per 100 um2; per neuron: mRNA copies in soma + dendrites
6. MORPHOLOGY of the imaged neuron (dendritic tree connected to the soma)
   - soma area, total dendrite length (pruned skeleton), number of primary dendrites, maximal dendrite reach
   (branch points are only listed per image: on projected masks they are not reliable enough for statistics)

Statistics for every metric (t-tests AND robust linear model, as in the main statistics):
   DIV9 vs DIV27 (conditions in the statistics script's TIMEPOINT_CONDITIONS) and control vs protease.
"""
import os
import json
import time
import numpy as np
import pandas as pd

import Fish_spots_batch_v4 as V4
import smFISH_statistics_superplots as SP

# =============================================================================
ROOT_FOLDER = None            # None -> dialog; the 'biological replicates' folder
BIN_UM = 5.0                  # distance bin width for the profiles (um)
MAX_DISTANCE_UM = 150.0       # profiles are shown up to this distance
MIN_BIN_AREA_UM2 = 5.0        # bins with less dendrite area in an image are ignored (unreliable)
MULTI_COPY_THRESHOLD = 1.5    # spot counts as multi-mRNA if its intensity >= 1.5 single-mRNA units
SPUR_UM = 3.0                 # skeleton side branches shorter than this are segmentation noise, removed
HOLE_UM2 = 2.0                # holes in the dendrite mask smaller than this are filled before skeletonizing
JUNCTION_MERGE_UM = 1.0       # junction pixels closer than this count as one branch point
MIN_SPOTS_FOR_UNIT = 30       # fewer counted neurite spots in a pair -> single-mRNA unit from all compartments
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)
OUT_NAME = 'advanced_quantifications'
# =============================================================================


# -----------------------------------------------------------------------------
# per image: geodesic distance, profile, morphology (runs in parallel)
# -----------------------------------------------------------------------------
def geodesic_from_soma(dend, soma):
    """Path length (px) from the soma through the dendrite mask; inf where not connected."""
    from skimage.graph import MCP_Geometric
    from scipy import ndimage as ndi
    start = dend & ndi.binary_dilation(soma, iterations=1)
    if not start.any():
        return None
    costs = np.where(dend, 1.0, np.inf)
    dist, _ = MCP_Geometric(costs).find_costs(starts=list(zip(*np.nonzero(start))))
    dist[~dend] = np.inf
    return dist


def _neighbours(sk):
    from scipy import ndimage as ndi
    return ndi.convolve(sk.astype(np.int16), np.ones((3, 3), np.int16), mode='constant') - sk


def prune_spurs(sk, px_um, spur_um=SPUR_UM):
    """Standard skeleton pruning (Gonzalez & Woods): remove end pixels spur_um deep - this deletes every
    spur shorter than spur_um completely and shortens real branches - then grow the remaining branch ends
    back along the ORIGINAL skeleton, so real branches regain their full length."""
    n = max(1, int(round(spur_um / px_um)))
    ys, xs = np.nonzero(sk)
    if ys.size == 0:
        return sk
    y0, y1 = max(ys.min() - 2, 0), ys.max() + 3
    x0, x1 = max(xs.min() - 2, 0), xs.max() + 3
    orig = sk[y0:y1, x0:x1].copy()
    x = orig.copy()
    for _ in range(n):                                   # 1) thin away end points n times
        ends = x & (_neighbours(x) == 1)
        single = x & (_neighbours(x) == 0)
        if not ends.any():
            break
        x = x & ~ends & ~single
    grow = x & (_neighbours(x) == 1)                     # 2) end points of what is left
    from scipy import ndimage as ndi
    for _ in range(n):                                   # 3) regrow them along the original skeleton
        grow = ndi.binary_dilation(grow, structure=np.ones((3, 3), bool)) & orig
    out = sk.copy()
    out[y0:y1, x0:x1] = x | grow                         # 4) union
    return out


def skeleton_stats(mask, px_um):
    """Length (um), branch points and end points of the pruned skeleton of the mask."""
    from skimage.morphology import skeletonize, remove_small_holes
    from scipy import ndimage as ndi
    # clean the mask first: fill small holes (they create artificial loops) and smooth ragged edges
    m = remove_small_holes(mask, max(1, int(HOLE_UM2 / px_um ** 2)))
    r = max(1, int(round(0.1 / px_um)))
    ball = np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r
    m = ndi.binary_opening(ndi.binary_closing(m, structure=ball), structure=ball) & mask | (m & mask)
    sk = skeletonize(m)
    if not sk.any():
        return 0.0, 0, 0
    sk = prune_spurs(sk, px_um)
    h = np.sum(sk[:, 1:] & sk[:, :-1]); v = np.sum(sk[1:, :] & sk[:-1, :])
    d1 = np.sum(sk[1:, 1:] & sk[:-1, :-1]); d2 = np.sum(sk[1:, :-1] & sk[:-1, 1:])
    length = (h + v + np.sqrt(2) * (d1 + d2)) * px_um
    nb = _neighbours(sk)
    junc = sk & (nb >= 3)
    k = max(1, int(round(JUNCTION_MERGE_UM / px_um / 2)))
    branch = ndi.label(ndi.binary_dilation(junc, iterations=k))[1] if junc.any() else 0   # merged junctions
    ends = int(np.sum(sk & (nb == 1)))
    return float(length), int(branch), ends


def per_image(r):
    cache = os.path.join(os.path.dirname(r['fish_path']), '_pipeline_cache', f"{r['image_id']}_advanced.json")
    filt = os.path.join(os.path.dirname(r['fish_path']), '_pipeline_cache', f"{r['image_id']}_spots_filtered.csv.gz")
    if not os.path.exists(filt):
        return r['image_id'], None, 'no filtered spots (run Fish_spots_batch_v4.py first)'
    key = dict(masks=V4.mask_settings(), bin=BIN_UM, spots=os.path.getmtime(filt))
    if os.path.exists(cache):
        try:
            c = json.load(open(cache))
            if c.get('key') == json.loads(json.dumps(key)):
                return r['image_id'], c, 'cached'
        except Exception:
            pass
    try:
        masks = V4.build_masks(r)
        px = r['px_um']
        dend, soma = masks['Dendrite'], masks['Soma']
        spots = pd.read_csv(filt)
        kept = spots[spots['kept'].astype(bool)]
        ds = kept[kept.compartment == 'Dendrite']
        out = dict(key=key)
        # geodesic distance (defines the dendritic tree connected to the soma)
        dist = geodesic_from_soma(dend, soma)
        tree = np.isfinite(dist) if dist is not None else dend
        # morphology of the dendritic tree connected to the soma (loose fragments excluded)
        L, nbranch, nend = skeleton_stats(tree, px)
        out.update(soma_area_um2=float(soma.sum() * px ** 2), dendrite_area_um2=float(dend.sum() * px ** 2),
                   dendrite_tree_area_um2=float(tree.sum() * px ** 2),
                   dendrite_length_um=L, dendrite_branch_points=nbranch, dendrite_end_points=nend)
        if dist is not None:
            dd = dist * px
            # primary dendrites: separate dendrite trunks crossing a ring 2-5 um from the soma
            from scipy import ndimage as ndi
            ring = np.isfinite(dd) & (dd >= 2.0) & (dd <= 5.0)
            lab, nlab = ndi.label(ring, structure=np.ones((3, 3)))
            sizes = np.bincount(lab.ravel())[1:] * px ** 2
            out['primary_dendrites'] = int(np.sum(sizes >= 0.5))
            reach = np.isfinite(dd)
            pix_d = dd[reach]
            sd = dd[ds.spot_y.to_numpy(int), ds.spot_x.to_numpy(int)] if len(ds) else np.array([])
            sd_ok = sd[np.isfinite(sd)]
            nb = int(np.ceil(max(pix_d.max() if pix_d.size else 0, MAX_DISTANCE_UM) / BIN_UM))
            edges = np.arange(nb + 1) * BIN_UM
            out.update(max_dendrite_reach_um=float(pix_d.max()) if pix_d.size else np.nan,
                       fraction_dendrite_connected=float(reach.sum() / max(dend.sum(), 1)),
                       fraction_spots_connected=float(sd_ok.size / max(len(sd), 1)) if len(sd) else np.nan,
                       distal_index=float(sd_ok.mean() / pix_d.mean()) if sd_ok.size and pix_d.size else np.nan,
                       median_spot_distance_um=float(np.median(sd_ok)) if sd_ok.size else np.nan,
                       profile_counts=np.histogram(sd_ok, edges)[0].tolist(),
                       profile_area_um2=(np.histogram(pix_d, edges)[0] * px ** 2).tolist())
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, 'w') as fh:
            json.dump(out, fh)
        return r['image_id'], out, 'computed'
    except Exception as e:
        return r['image_id'], None, f"ERROR {type(e).__name__}: {e}"


# -----------------------------------------------------------------------------
# mRNAs per spot
# -----------------------------------------------------------------------------
def single_mrna_unit(values):
    """Most frequent integrated intensity (mode of the log-intensity distribution)."""
    from scipy.stats import gaussian_kde
    v = np.log10(values[values > 0])
    if v.size < 30:
        return np.nan
    if v.size > 20000:
        v = np.random.default_rng(0).choice(v, 20000, replace=False)
    grid = np.linspace(np.percentile(v, 1), np.percentile(v, 99), 400)
    return float(10 ** grid[np.argmax(gaussian_kde(v)(grid))])


def copies_metrics(inv, root):
    rows, units = [], []
    for pair, g in inv.groupby('pair'):
        tabs = {}
        for rr in g.itertuples():
            f = os.path.join(root, rr.folder, '_pipeline_cache', f"{rr.image_id}_spots_filtered.csv.gz")
            if os.path.exists(f):
                t = pd.read_csv(f, usecols=['compartment', 'integrated_intensity', 'kept'])
                tabs[rr.image_id] = t[t['kept'].astype(bool) & (t.compartment != 'Background')]
        if not tabs:
            continue
        allsp = pd.concat(tabs.values())
        neur = allsp[allsp.compartment.isin(['Dendrite', 'Axon'])]
        src = 'neurite spots' if len(neur) >= MIN_SPOTS_FOR_UNIT else 'all spots (too few neurite spots)'
        use = neur if len(neur) >= MIN_SPOTS_FOR_UNIT else allsp
        unit = single_mrna_unit(use.integrated_intensity.to_numpy())
        units.append(dict(pair=pair, single_mrna_integrated_intensity=unit, estimated_from=src,
                          n_spots_used=len(use)))
        if not np.isfinite(unit):
            continue
        for rr in g.itertuples():
            if rr.image_id not in tabs:
                continue
            t = tabs[rr.image_id]
            for comp in V4.COMPARTMENTS:
                c = t[t.compartment == comp].integrated_intensity.to_numpy() / unit
                rows.append(dict(image_id=rr.image_id, compartment=comp, n_spots=len(c),
                                 mean_mrnas_per_spot=float(c.mean()) if len(c) else np.nan,
                                 fraction_multi_mrna_spots=float(np.mean(c >= MULTI_COPY_THRESHOLD)) if len(c) else np.nan,
                                 mrna_copies=float(c.sum())))
    return pd.DataFrame(rows), pd.DataFrame(units)


# -----------------------------------------------------------------------------
# statistics and figures
# -----------------------------------------------------------------------------
def long_table(meta, df, cols):
    """image-level wide table -> long format (one row per image and metric) for the statistics."""
    m = df.merge(meta, on='image_id')
    out = []
    for col, label in cols.items():
        x = m[['probe', 'timepoint', 'replicate', 'condition', 'image_id', col]].rename(columns={col: 'value'})
        x = x[np.isfinite(x['value'].astype(float))]
        x['compartment'] = label                                  # the statistics code groups by this
        out.append(x)
    return pd.concat(out, ignore_index=True)


def run_stats(long, out):
    rm = SP.replicate_means(long, 'value')
    res = []
    for method, tag in [('t', 't-test'), ('rlm', 'robust linear model')]:
        c = SP.test_conditions(rm, 'value', method).assign(comparison='control vs protease', method=tag)
        t = SP.test_timepoints(rm, 'value', method).assign(comparison='DIV9 vs DIV27', method=tag)
        res += [c, t]
    s = pd.concat(res, ignore_index=True).rename(columns={'compartment': 'metric'})
    return rm, s


def plot_metrics(long, rm, s, out):
    tps = [t for t in SP.TIMEPOINT_ORDER if t in set(long.timepoint)]
    conds = [c for c in SP.CONDITION_ORDER if c in SP.TIMEPOINT_CONDITIONS and c in set(long.condition)]
    for method, sub in [('robust linear model', 'robust_linear_model'), ('t-test', 't_tests')]:
        od = os.path.join(out, sub)
        os.makedirs(od, exist_ok=True)
        st = s[(s.method == method) & (s.comparison == 'DIV9 vs DIV27')].rename(columns={'metric': 'compartment'})
        for metric in long.compartment.unique():
            assert not any(ch in metric for ch in '\\/:*?"<>|'), metric   # must be usable in file names
            d = long[long.compartment == metric]
            safe = ''.join(ch if ch.isalnum() else '_' for ch in metric)
            SP.superplot(d.assign(density=d.value), rm[rm.compartment == metric].assign(density=lambda x: x.value),
                         st, 'value', metric, 'timepoint', tps, 'compartment', [metric], 'condition', conds,
                         SP.timepoints_paired, od, f'SuperPlot_{safe}', f'{metric}: DIV9 vs DIV27 ({method})')


def plot_profiles(prof, out):
    import matplotlib.pyplot as plt
    probes = [p for p in SP.PROBE_ORDER if p in set(prof.probe)]
    tps = [t for t in SP.TIMEPOINT_ORDER if t in set(prof.timepoint)]
    cols = {tps[0]: '#1f77b4', tps[-1]: '#d62728'} if tps else {}
    for cond in [c for c in SP.CONDITION_ORDER if c in set(prof.condition)]:
        for norm in (False, True):
            fig, axs = plt.subplots(1, len(probes), figsize=(3.2 * len(probes), 3.0), squeeze=False, sharey=norm)
            for ax, probe in zip(axs[0], probes):
                for tp in tps:
                    g = prof[(prof.probe == probe) & (prof.timepoint == tp) & (prof.condition == cond)]
                    if not len(g):
                        continue
                    col = 'relative_density' if norm else 'density_per_100um2'
                    rep = g.groupby(['replicate', 'bin_center_um'])[col].mean().unstack(0)
                    rep = rep[rep.index <= MAX_DISTANCE_UM]
                    m, sd = rep.mean(axis=1), rep.std(axis=1)
                    ax.plot(rep.index, m, color=cols.get(tp), lw=1.5, label=tp)
                    ax.fill_between(rep.index, m - sd, m + sd, color=cols.get(tp), alpha=0.2, lw=0)
                ax.set_title(probe, fontsize=9)
                ax.set_xlabel('distance from soma along dendrite (µm)', fontsize=7)
                ax.tick_params(labelsize=7)
                if norm:
                    ax.axhline(1, color='grey', lw=0.8, ls=':')
            axs[0, 0].set_ylabel('relative density (1 = dendrite mean)' if norm else SP.DENSITY_LABEL, fontsize=8)
            axs[0, 0].legend(fontsize=7, frameon=False)
            fig.suptitle(f"Dendritic spot density vs distance from the soma - {cond} "
                         f"(line = mean of replicates, band = SD)", fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            name = f"PROFILE_distance_from_soma_{cond}{'_relative' if norm else ''}"
            for fmt in SP.STYLE['save_formats']:
                fig.savefig(os.path.join(out, f"{name}.{fmt}"), dpi=SP.STYLE['dpi'], bbox_inches='tight')
            plt.close(fig)


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    root = ROOT_FOLDER
    if root is None:
        from tkinter import Tk, filedialog
        tk = Tk(); tk.withdraw(); tk.attributes('-topmost', True)
        root = filedialog.askdirectory(parent=tk, title='Select the top folder (e.g. "biological replicates")')
        tk.destroy()
        if not root:
            raise SystemExit("No folder selected.")
    root = os.path.abspath(root)
    res_dir = os.path.join(root, V4.RESULTS_FOLDER_NAME)
    inv = pd.read_csv(os.path.join(res_dir, 'inventory.csv'))
    counts = pd.read_csv(os.path.join(res_dir, 'counts_per_image.csv'))
    out = os.path.join(res_dir, OUT_NAME)
    os.makedirs(out, exist_ok=True)
    SP.setup_matplotlib()
    meta = inv[['image_id', 'probe', 'timepoint', 'replicate', 'condition', 'pair', 'setup']]

    # ---- 1 + 6: per image (parallel, cached)
    print(f"Distances and morphology for {len(inv)} images ({N_WORKERS} processes) ...")
    recs, t0 = {}, time.time()
    todo = inv.to_dict('records')
    if N_WORKERS > 1:
        from multiprocessing import Pool
        with Pool(N_WORKERS) as pool:
            it = pool.imap_unordered(per_image, todo)
            for k, (iid, o, msg) in enumerate(it, 1):
                if o is not None:
                    recs[iid] = o
                if msg not in ('cached', 'computed'):
                    print(f"  {iid}: {msg}")
                if k % 50 == 0:
                    print(f"  {k}/{len(todo)}  ~{(time.time() - t0) / k * (len(todo) - k) / 60:.0f} min left")
    else:
        for r in todo:
            iid, o, msg = per_image(r)
            if o is not None:
                recs[iid] = o
    img = pd.DataFrame([{k: v for k, v in o.items() if not k.startswith('profile') and k != 'key'} | {'image_id': i}
                        for i, o in recs.items()])
    prof_rows = []
    for i, o in recs.items():
        if 'profile_counts' not in o:
            continue
        for b, (n, a) in enumerate(zip(o['profile_counts'], o['profile_area_um2'])):
            if a >= MIN_BIN_AREA_UM2:
                prof_rows.append(dict(image_id=i, bin_center_um=(b + 0.5) * BIN_UM, spots=n, area_um2=a,
                                      density_per_100um2=100 * n / a))
    prof = pd.DataFrame(prof_rows).merge(meta, on='image_id')
    dmean = prof.groupby('image_id').apply(lambda g: 100 * g.spots.sum() / g.area_um2.sum(), include_groups=False)
    prof['relative_density'] = prof.density_per_100um2 / prof.image_id.map(dmean)

    # ---- 2: localization indices
    wide = counts.pivot_table(index='image_id', columns='compartment', values='density_per_um2')
    loc = pd.DataFrame({'image_id': wide.index})
    with np.errstate(divide='ignore', invalid='ignore'):
        loc['dendrite_soma_ratio'] = (wide['Dendrite'] / wide['Soma']).to_numpy()
        loc['dendrite_axon_ratio'] = (wide['Dendrite'] / wide['Axon']).to_numpy()
    loc = loc.replace([np.inf, -np.inf], np.nan)

    # ---- 3: mRNAs per spot
    print("mRNAs per spot ...")
    cop, units = copies_metrics(inv, root)
    ca = counts[['image_id', 'compartment', 'area_um2']]
    cop = cop.merge(ca, on=['image_id', 'compartment'], how='left')
    cop['mrna_copies_per_100um2'] = 100 * cop.mrna_copies / cop.area_um2
    cw = cop.pivot_table(index='image_id', columns='compartment',
                         values=['mean_mrnas_per_spot', 'fraction_multi_mrna_spots', 'mrna_copies'])
    cw.columns = [f"{a}_{b}" for a, b in cw.columns]
    cw = cw.reset_index()
    cw['mrna_copies_neuron_soma_dendrite'] = cw.get('mrna_copies_Soma', 0) + cw.get('mrna_copies_Dendrite', 0)

    # ---- all metrics per image
    per_img = meta.merge(img, on='image_id', how='left').merge(loc, on='image_id', how='left') \
        .merge(cw, on='image_id', how='left')
    metrics = {
        'distal_index': 'Dendrite distal index',
        'median_spot_distance_um': 'Median dendritic spot distance from soma (µm)',
        'dendrite_soma_ratio': 'Dendrite to soma density ratio',
        'dendrite_axon_ratio': 'Dendrite to axon density ratio',
        'mean_mrnas_per_spot_Soma': 'mRNAs per spot - soma',
        'mean_mrnas_per_spot_Dendrite': 'mRNAs per spot - dendrite',
        'mean_mrnas_per_spot_Axon': 'mRNAs per spot - axon',
        'fraction_multi_mrna_spots_Dendrite': 'Fraction multi-mRNA spots - dendrite',
        'fraction_multi_mrna_spots_Axon': 'Fraction multi-mRNA spots - axon',
        'mrna_copies_neuron_soma_dendrite': 'mRNA copies per neuron (soma + dendrites)',
        'soma_area_um2': 'Soma area (µm²)',
        'dendrite_length_um': 'Total dendrite length (µm)',
        'primary_dendrites': 'Primary dendrites (count)',
        'max_dendrite_reach_um': 'Maximal dendrite reach from soma (µm)',
    }
    metrics = {k: v for k, v in metrics.items() if k in per_img.columns}
    long = long_table(meta, per_img[['image_id'] + list(metrics)], metrics)
    rm, stats_all = run_stats(long, out)

    # ---- outputs
    stats_all.to_csv(os.path.join(out, 'stats_all_metrics.csv'), index=False)
    with pd.ExcelWriter(os.path.join(out, 'advanced_quantifications.xlsx')) as xw:
        per_img.to_excel(xw, sheet_name='per_image', index=False)
        rm.rename(columns={'compartment': 'metric'}).to_excel(xw, sheet_name='replicate_means', index=False)
        for comp in ['DIV9 vs DIV27', 'control vs protease']:
            for meth in ['t-test', 'robust linear model']:
                ss = stats_all[(stats_all.comparison == comp) & (stats_all.method == meth)].dropna(axis=1, how='all')
                ss.to_excel(xw, sheet_name=f"{'DIV' if 'DIV' in comp else 'ctrl_prot'}_{'ttest' if meth == 't-test' else 'rlm'}",
                            index=False)
        prof.to_excel(xw, sheet_name='distance_profiles_per_image', index=False)
        units.to_excel(xw, sheet_name='single_mRNA_unit_per_pair', index=False)
        pd.DataFrame({'metric': list(metrics.values()), 'column': list(metrics)}).to_excel(xw, sheet_name='metrics', index=False)
        pd.DataFrame({'item': ['masks', 'distance', 'distal index', 'localization ratios', 'mRNAs per spot',
                               'morphology', 'statistics'],
                      'method': [f"same as Fish_spots_batch_v4.py: {V4.mask_settings()}",
                                 f"geodesic path length from the soma through the dendrite mask (skimage MCP_Geometric); "
                                 f"profiles in {BIN_UM} um bins, bins with < {MIN_BIN_AREA_UM2} um2 dendrite ignored",
                                 'mean distance of counted dendritic spots / mean distance of all connected dendrite pixels',
                                 'spot density dendrite / soma and dendrite / axon, per image; replicate mean of the images',
                                 f'single-mRNA intensity per control/protease pair = mode of the integrated intensity of the '
                                 f'counted neurite spots (KDE on log10); multi-mRNA = >= {MULTI_COPY_THRESHOLD} units',
                                 f'soma mask area; dendritic tree = dendrite mask connected to the soma (holes < {HOLE_UM2} '
                                 f'um2 filled); length of its skeleton (8-connected, diagonal steps sqrt2) after pruning side '
                                 f'branches < {SPUR_UM} um (Gonzalez & Woods); primary dendrites = separate dendrite trunks '
                                 f'crossing the 2-5 um distance ring around the soma; reach = maximal geodesic distance. '
                                 f'Branch points are reported per image for information only (unreliable on projected masks)',
                                 'as smFISH_statistics_superplots.py: t-tests on replicate means and robust linear model '
                                 'with differentiation as fixed effect']}).to_excel(xw, sheet_name='methods', index=False)
    plot_profiles(prof, out)
    plot_metrics(long, rm, stats_all, out)

    # ---- summary on screen
    print(f"\nImages with results: {len(recs)} of {len(inv)}")
    if 'fraction_spots_connected' in per_img:
        print(f"Dendritic spots reachable from the soma through the mask: "
              f"median {per_img.fraction_spots_connected.median():.0%} per image")
    print("\nDIV9 vs DIV27 (robust linear model / t-test), number of probes with p < 0.05:")
    for metric in metrics.values():
        s = stats_all[(stats_all.metric == metric) & (stats_all.comparison == 'DIV9 vs DIV27')]
        n_r = int((s[s.method == 'robust linear model'].p < 0.05).sum())
        n_t = int((s[s.method == 't-test'].p < 0.05).sum())
        print(f"  {metric:50s} {n_r:2d} / {n_t:2d}")
    print(f"\nResults in: {out}")
