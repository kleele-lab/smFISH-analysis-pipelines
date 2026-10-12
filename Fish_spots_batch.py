"""
smFISH compartment quantification - batch version (v4).

Processes a whole project in one run: select the top folder (e.g. "biological replicates");
every image folder below it is found and analysed automatically.

Expected layout (names are parsed, capitalization does not matter):
    <top folder>/<date>_..._div9_..._n=1/<Probe> <Ctr|ctrl|(nothing)|Prot|Protease>/  image files
  timepoint  = 'div<number>' in the replicate folder name
  replicate  = 'n=<number>'   in the replicate folder name
  probe      = one of PROBES in the image folder name
  condition  = 'protease' if the image folder name contains 'prot', otherwise 'control'
A control/protease PAIR = same timepoint + replicate + probe (same differentiation).

Per image (all found automatically, matched by sample ID):
  FISH MIP  '...Cy5_MIP.tif'  | soma mask '... [1].tif' | spots 'Spots (...).csv'
  dendrite map '..._Probabilities_0.tif(f)' (no 'Neurite') | filler map '...Neurite..._Probabilities_0.tif(f)'
Images without a soma mask are ignored.

Pixel size is read from each image. Files calibrated in 'cm' (VisiView/EMCCD) get VISIVIEW_PX_UM.

TWO PASSES
  1. every spot is fitted (2D Gaussian + offset, as in v3) and assigned to a compartment.
     Results are cached per image, so a second run skips this (set REFIT = True to redo).
  2. filters, identical for all compartments and for control and protease:
       size (MIN/MAX_DIAMETER_NM), saturation, MIN_SNR (SNR = fitted amplitude above the local
       background / local noise). All intensities are background-subtracted (fitted offset).
     Optional intensity cutoff per control/protease pair: OFF by default. It is never applied
     automatically, because the dim and bright populations are not reliably separable by an
     algorithm in all datasets. The script PROPOSES a value and plots each pair's histogram;
     a cutoff is applied only if you enter it in _density_results/intensity_cutoffs.csv
     (column cutoff_manual) and set INTENSITY_CUTOFF = 'manual'.
     Then counts, densities, QC overlays and project-level tables.

Outputs: <top folder>/_density_results/ (project tables + intensity-cutoff report)
         and in every image folder: SpotCounts_v4.csv and QC overlays for a few images.
"""
import os
import re
import sys
import glob
import time
import numpy as np
import pandas as pd
import tifffile
from skimage.io import imread
from scipy.optimize import curve_fit
from skimage.segmentation import find_boundaries
from skimage.draw import disk, circle_perimeter

# =============================================================================
# USER PARAMETERS
# =============================================================================
ROOT_FOLDER = None            # None -> folder dialog; or e.g. r"D:\smFISH\biological replicates"

PROBES = ['BNIP3L', 'BNIP3', 'FIS1', 'MFF', 'OPA1', 'PINK1']   # longer names first
VISIVIEW_PX_UM = 0.13         # pixel size for files calibrated in 'cm' (VisiView/EMCCD)
PROB_THRESHOLD = 0.35          # Ilastik probability threshold
# Widen the thin Ilastik masks by this distance (in um, so every camera gets the same physical
# widening; 0 = off). The soma mask is never changed. The dendrite is widened first; the axon mask is
# then everything in the widened filler map outside the (widened) dendrite and the soma.
# Example: 0.2 um = 3 px on the 2304/2048 cameras (0.065 um/px), 1-2 px on the 1024 ones (0.13 um/px).
DENDRITE_DILATION_UM = 0.1
AXON_DILATION_UM = 0.1
# Changing these (or PROB_THRESHOLD) is detected automatically: the spot fits are kept, only the masks,
# compartment assignment and areas are recalculated (fast).

# --- Spot filters (identical for all compartments and conditions) ---
MIN_DIAMETER_NM = 140
MAX_DIAMETER_NM = 500
SATURATION_LEVEL = 65535
# SNR cutoff per imaging setup (camera / microscope). Spots with SNR below the value are rejected.
# Changing these values does NOT need a refit - rerunning takes minutes.
MIN_SNR_BY_SETUP = {
    '2304px_0.065um':          4.0,   # most of n=2 and n=3 (Nikon, 2304 x 2304 camera)
    '2048px_0.065um':          4.0,   # DIV9 n2 MFF, DIV9 n3 OPA1, DIV9 n3 PINK1 (2048 x 2048 camera)
    '1024px_0.130um':          8.0,   # Nikon 1024 x 1024: DIV9 n1 BNIP3/BNIP3L/OPA1/PINK1, DIV9 n2 OPA1
    '1024px_0.130um_VisiView': 12.0,   # VisiView EMCCD: all of DIV27 n1, DIV9 n1 FIS1/MFF
}
MIN_SNR = 5.0                  # used only for a setup that is not listed above (a warning is printed)

# Stricter (or looser) SNR cutoffs for particular imaging batches. These override MIN_SNR_BY_SETUP.
# Possible keys - the most specific matching key wins:
#   'DIV27_n3_PINK1'   one control/protease pair (timepoint_replicate_PROBE, as in the output tables)
#   ('n3', 'DIV9')     one replicate at one timepoint (= one imaging batch)
#   'n3'               a whole replicate (both timepoints)
#   'DIV9'             a whole timepoint (all replicates)
# There is deliberately no key for control/protease: both conditions of a pair always get the same
# cutoff. Changing these values does NOT need a refit - rerunning takes minutes.
MIN_SNR_OVERRIDES = {
    ('n3', 'DIV9'):  8.0,
    ('n3', 'DIV27'): 8.0,
}

# --- Intensity cutoff (dim population vs single mRNAs) ---
INTENSITY_CUTOFF = 'off'      # 'off' or 'manual' (uses cutoff_manual in intensity_cutoffs.csv)
MIN_SPOTS_FOR_CUTOFF = 200    # minimum pooled spots for a proposed cutoff

FIT_WINDOW = 7
REFIT = False                 # True = ignore cached fits and fit all spots again
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)   # parallel processes for fitting

QC_OVERLAYS_PER_FOLDER = 'all'    # QC overlays for this many random images per folder; 'all' or 0
QC_SPOT_RADIUS = 4
# Colours of the QC overlay channels (red, green, blue; 0-255). Fiji opens them as a composite.
QC_COLORS = {
    'raw':      (255, 255, 255),   # C1 smFISH image (grey)
    'soma':     (255, 255, 0),     # C2 soma outline (yellow)
    'dendrite': (0, 255, 255),     # C3 dendrite outline (cyan)
    'axon':     (0, 255, 0),       # C4 axon outline (green)
    'counted':  (0, 0, 200),       # C5 counted spots (dark blue)
    'rejected': (255, 0, 0),       # C6 rejected spots, circles (red)
}
QC_COUNTED_AS_RING = True      # True = counted spots drawn as rings (the spot stays visible underneath)
                               # False = filled discs
RANDOM_SEED = 1
RESULTS_FOLDER_NAME = '_density_results'
# =============================================================================

FISH_RX = re.compile(r'cy5_mip\.tiff?$', re.I)
SOMA_RX = re.compile(r'\[\d+\]\.tiff?$')
ID_RX = re.compile(r'^(.*)(?:_C\d+_|_w\d+conf|_conf)')
SKIP_DIRS = ('_', 'QC_', 'Threshold')
COMPARTMENTS = ['Soma', 'Dendrite', 'Axon']


# -----------------------------------------------------------------------------
# Discovery
# -----------------------------------------------------------------------------
def image_id(name):
    m = ID_RX.match(os.path.basename(name))
    return m.group(1) if m else None


def id_regex(iid):
    return re.compile(r'(?<![A-Za-z0-9])' + re.escape(iid) + r'(?![A-Za-z0-9])')


def parse_folder(rel_path):
    parts = rel_path.replace('\\', '/').split('/')
    top, leaf = parts[0], parts[-1]
    tp = re.search(r'div\s*(\d+)', top, re.I)
    rep = re.search(r'n\s*=\s*(\d+)', top, re.I)
    probe = next((p for p in PROBES if re.search(p, leaf, re.I)), None)
    cond = 'protease' if re.search(r'prot', leaf, re.I) else 'control'
    return (f"DIV{tp.group(1)}" if tp else None, f"n{rep.group(1)}" if rep else None, probe, cond)


def pixel_size(path):
    """Returns (px_um, setup label) from the TIFF calibration."""
    with tifffile.TiffFile(path) as t:
        shp = t.series[0].shape
        md = t.imagej_metadata or {}
        xr = t.pages[0].tags.get('XResolution')
        unit = str(md.get('unit', '')).lower()
        px = xr.value[1] / xr.value[0] if (xr is not None and xr.value[0]) else None
    if unit in ('micron', 'um', 'µm', '\\u00b5m') and px:
        return round(px, 4), f"{shp[-1]}px_{px:.3f}um"
    if unit == 'cm':
        return VISIVIEW_PX_UM, f"{shp[-1]}px_{VISIVIEW_PX_UM:.3f}um_VisiView"
    return None, None


def find_one(files, iid, kind, problems, folder):
    rx = id_regex(iid)
    hits = [f for f in files if rx.search(os.path.basename(f))]
    if len(hits) == 1:
        return hits[0]
    problems.append((folder, iid, f"{len(hits)} {kind} files"))
    return None


def discover(root):
    rows, problems = [], []
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(SKIP_DIRS))
        fish = sorted(f for f in files if FISH_RX.search(f) and '[' not in f)
        if not fish:
            continue
        rel = os.path.relpath(dirpath, root)
        tp, rep, probe, cond = parse_folder(rel)
        if None in (tp, rep, probe):
            problems.append((rel, '-', 'cannot parse timepoint/replicate/probe from folder names'))
            continue
        full = lambda L: [os.path.join(dirpath, f) for f in L]
        soma = full([f for f in files if SOMA_RX.search(f)])
        spots = full([f for f in files if f.lower().startswith('spots') and f.lower().endswith('.csv')])
        probs = full([f for f in files if re.search(r'_Probabilities_0\.tiff?$', f)])
        filler = [f for f in probs if 'neurite' in os.path.basename(f).lower()]
        dend = [f for f in probs if 'neurite' not in os.path.basename(f).lower()]
        soma_ids = {image_id(f) for f in soma}
        for f in fish:
            iid = image_id(f)
            if iid is None or iid not in soma_ids:
                continue                                          # no soma mask -> ignored
            paths = dict(fish=os.path.join(dirpath, f),
                         soma=find_one(soma, iid, 'soma mask', problems, rel),
                         spots=find_one(spots, iid, 'spots CSV', problems, rel),
                         dendrite=find_one(dend, iid, 'dendrite map', problems, rel),
                         filler=find_one(filler, iid, 'cell-filler map', problems, rel))
            if not all(paths.values()):
                continue
            px, setup = pixel_size(paths['fish'])
            if px is None:
                problems.append((rel, iid, 'no pixel calibration in FISH image'))
                continue
            rows.append(dict(folder=rel, image_id=iid, timepoint=tp, replicate=rep, probe=probe,
                             condition=cond, pair=f"{tp}_{rep}_{probe}", setup=setup, px_um=px,
                             **{k + '_path': v for k, v in paths.items()}))
    return pd.DataFrame(rows), problems


# -----------------------------------------------------------------------------
# Pass 1: fit all spots of one image (runs in parallel workers)
# -----------------------------------------------------------------------------
def _gauss(coords, A, x0, y0, sx, sy, off):
    x, y = coords
    return (off + A * np.exp(-((x - float(x0)) ** 2 / (2 * sx ** 2) +
                               (y - float(y0)) ** 2 / (2 * sy ** 2)))).ravel()


def fit_spot(image, y, x, w=FIT_WINDOW):
    h = w // 2
    y0, y1 = max(y - h, 0), min(y + h + 1, image.shape[0])
    x0, x1 = max(x - h, 0), min(x + h + 1, image.shape[1])
    s = image[y0:y1, x0:x1].astype(np.float64)
    yy, xx = np.mgrid[0:s.shape[0], 0:s.shape[1]]
    out = dict(amplitude=np.nan, sigma_x=np.nan, sigma_y=np.nan, bg_offset=np.nan, noise_sd=np.nan,
               snr=np.nan, integrated_intensity=np.nan, raw_max_pixel=float(s.max()) if s.size else np.nan)
    try:
        p, _ = curve_fit(_gauss, (xx, yy), s.ravel(),
                         p0=(s.max() - s.min(), (x1 - x0) / 2, (y1 - y0) / 2, 1.0, 1.0, s.min()),
                         bounds=([0, 0, 0, 0.5, 0.5, 0], [np.inf, s.shape[1], s.shape[0], 5, 5, np.inf]))
    except Exception:
        return out
    res = s.ravel() - _gauss((xx, yy), *p)
    noise = np.sqrt(np.sum(res ** 2) / max(res.size - 6, 1))
    out.update(amplitude=p[0], sigma_x=p[3], sigma_y=p[4], bg_offset=p[5], noise_sd=noise,
               snr=p[0] / noise if noise > 0 else np.nan,
               integrated_intensity=2 * np.pi * p[0] * p[3] * p[4])
    return out


def threshold_prob(img):
    p = img.astype(np.float32)
    if p.max() > 1.0:
        p = p / (float(np.iinfo(img.dtype).max) if np.issubdtype(img.dtype, np.integer) else p.max())
    return p > PROB_THRESHOLD


def dilate(mask, um, px_um):
    rad = int(round(um / px_um)) if um and px_um else 0
    if rad <= 0:
        return mask
    from scipy import ndimage as ndi
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    return ndi.binary_dilation(mask, structure=(yy ** 2 + xx ** 2) <= rad ** 2)


def mask_settings():
    return dict(prob=PROB_THRESHOLD, dend_um=DENDRITE_DILATION_UM, axon_um=AXON_DILATION_UM)


def build_masks(r):
    soma = imread(r['soma_path']) > 0
    dend = dilate(threshold_prob(imread(r['dendrite_path'])), DENDRITE_DILATION_UM, r['px_um']) & ~soma
    axon = dilate(threshold_prob(imread(r['filler_path'])), AXON_DILATION_UM, r['px_um']) & ~dend & ~soma
    return {'Soma': soma, 'Dendrite': dend, 'Axon': axon}


def assign_and_save(r, df, masks, cp):
    """(Re)assign compartments from the masks and save spots, areas and the mask settings used."""
    comp = np.full(len(df), 'Background', dtype=object)
    if len(df):
        yy, xx = df.spot_y.to_numpy(int), df.spot_x.to_numpy(int)
        for c in COMPARTMENTS:
            comp[masks[c][yy, xx]] = c
    df['compartment'] = comp
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    df.to_csv(cp, index=False)
    pd.Series({c: int(masks[c].sum()) for c in COMPARTMENTS}).to_csv(cp.replace('_spots.csv.gz', '_areas.csv'))
    import json
    with open(cp.replace('_spots.csv.gz', '_masks.json'), 'w') as fh:
        json.dump(mask_settings(), fh)


def load_spots(path):
    df = pd.read_csv(path)
    if {'axis-0', 'axis-1'}.issubset(df.columns):
        df = df.rename(columns={'axis-0': 'y', 'axis-1': 'x'})
    elif not {'y', 'x'}.issubset(df.columns):
        if df.shape[1] == 2:
            df.columns = ['y', 'x']
        else:
            raise ValueError(f"no y/x columns in {path}")
    return df[['y', 'x']].round().astype(int).to_numpy()


def cache_path(r):
    return os.path.join(os.path.dirname(r['fish_path']), '_pipeline_cache', r['image_id'] + '_spots.csv.gz')


def process_image(r):
    """Fit every spot of one image; returns (image_id, n spots, message)."""
    import json
    cp = cache_path(r)
    if os.path.exists(cp) and not REFIT and os.path.getmtime(cp) >= os.path.getmtime(r['spots_path']):
        # spot fits are up to date; only redo the masks if their settings changed
        mp = cp.replace('_spots.csv.gz', '_masks.json')
        used = json.load(open(mp)) if os.path.exists(mp) else None
        if used == mask_settings():
            return r['image_id'], None, 'cached'
        try:
            df = pd.read_csv(cp)
            assign_and_save(r, df, build_masks(r), cp)
            os.utime(cp, None)
            return r['image_id'], len(df), 'masks updated (fits kept)'
        except Exception as e:
            return r['image_id'], 0, f"ERROR {type(e).__name__}: {e}"
    try:
        fish = imread(r['fish_path'])
        masks = build_masks(r)
        shapes = {fish.shape} | {m.shape for m in masks.values()}
        if len(shapes) != 1:
            return r['image_id'], 0, f"image shapes differ {shapes} - skipped"
        sp = load_spots(r['spots_path'])
        sp = sp[(sp[:, 0] >= 0) & (sp[:, 0] < fish.shape[0]) & (sp[:, 1] >= 0) & (sp[:, 1] < fish.shape[1])]
        recs = []
        for y, x in sp:
            d = fit_spot(fish, int(y), int(x))
            d.update(spot_y=int(y), spot_x=int(x))
            recs.append(d)
        df = pd.DataFrame(recs, columns=['amplitude', 'sigma_x', 'sigma_y', 'bg_offset', 'noise_sd', 'snr',
                                         'integrated_intensity', 'raw_max_pixel', 'spot_y', 'spot_x'])
        assign_and_save(r, df, masks, cp)
        return r['image_id'], len(df), 'fitted'
    except Exception as e:
        return r['image_id'], 0, f"ERROR {type(e).__name__}: {e}"


# -----------------------------------------------------------------------------
# Pass 2: filters, intensity cutoff, outputs
# -----------------------------------------------------------------------------
def bright_cutoff(values):
    """Two-component Gaussian mixture on log10(values): (cutoff, Ashman's D)."""
    v = np.log10(values[values > 0])
    lo, hi = np.percentile(v, [25, 90])
    w, mu, sd = np.array([0.8, 0.2]), np.array([lo, hi]), np.array([v.std() / 2] * 2)
    for _ in range(300):
        pdf = w / (sd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((v[:, None] - mu) / sd) ** 2)
        resp = pdf / np.maximum(pdf.sum(axis=1, keepdims=True), 1e-300)
        nk = resp.sum(axis=0)
        w = nk / len(v)
        mu_new = (resp * v[:, None]).sum(axis=0) / nk
        sd = np.sqrt((resp * (v[:, None] - mu_new) ** 2).sum(axis=0) / nk) + 1e-6
        done = np.allclose(mu_new, mu, atol=1e-7)
        mu = mu_new
        if done:
            break
    o = np.argsort(mu); w, mu, sd = w[o], mu[o], sd[o]
    D = np.sqrt(2) * (mu[1] - mu[0]) / np.sqrt(sd[0] ** 2 + sd[1] ** 2)
    grid = np.linspace(mu[0], mu[1], 2000)
    p = w / sd * np.exp(-0.5 * ((grid[:, None] - mu) / sd) ** 2)
    k = np.flatnonzero(p[:, 1] >= p[:, 0])
    return 10 ** (grid[k[0]] if len(k) else mu[1]), D


def valley_ratio(values, cutoff):
    """Is there a real dip at the cutoff? Smoothed histogram (log10) density at the cutoff region
    divided by the density at the bright peak above it. ~1 or more = no dip (one population with a tail)."""
    from scipy.stats import gaussian_kde
    v = np.log10(values[values > 0]); c = np.log10(cutoff)
    kde = gaussian_kde(v, bw_method=0.15)
    grid = np.linspace(v.min(), v.max(), 800)
    d = kde(grid)
    upper = grid >= c
    if not upper.any(): return np.inf
    peak_hi = d[upper].max(); x_hi = grid[upper][d[upper].argmax()]
    between = (grid >= grid[grid < c][d[grid < c].argmax()] if (grid < c).any() else grid < c) & (grid <= x_hi)
    return float(d[between].min() / peak_hi)


_UNLISTED = set()


def min_snr_for(setup, replicate=None, timepoint=None, pair=None):
    """SNR cutoff for one image: MIN_SNR_OVERRIDES (most specific key first), then MIN_SNR_BY_SETUP
    (exact setup label), then MIN_SNR."""
    for key in (pair, (replicate, timepoint), replicate, timepoint):
        if key is not None and key in MIN_SNR_OVERRIDES:
            return float(MIN_SNR_OVERRIDES[key])
    if setup in MIN_SNR_BY_SETUP:
        return float(MIN_SNR_BY_SETUP[setup])
    if setup not in _UNLISTED:
        _UNLISTED.add(setup)
        print(f"  WARNING: setup '{setup}' is not in MIN_SNR_BY_SETUP - using MIN_SNR = {MIN_SNR}")
    return MIN_SNR


def base_filter(df, px_um, setup='', replicate=None, timepoint=None, pair=None):
    """All filters except the intensity cutoff. Returns rejection reason ('' = passes)."""
    dia = (df['sigma_x'] + df['sigma_y']) * px_um * 1000          # 2 * mean(sigma)
    reason = pd.Series('', index=df.index, dtype=object)
    for cond, txt in [(df['snr'].isna(), 'fit_failed'),
                      (df['raw_max_pixel'] >= SATURATION_LEVEL, 'saturated'),
                      (dia < MIN_DIAMETER_NM, 'too_small'),
                      (dia > MAX_DIAMETER_NM, 'too_large'),
                      (df['snr'] < min_snr_for(setup, replicate, timepoint, pair), 'snr_too_low')]:
        reason[(reason == '') & cond.fillna(False)] = txt
    return reason, dia


def lut(rgb):
    """ImageJ LUT running from black to the given (r, g, b) colour."""
    ramp = np.arange(256, dtype=np.float64) / 255.0
    return np.stack([np.round(ramp * c).astype(np.uint8) for c in rgb])


def save_qc(r, masks, df, px_um, path):
    """6-channel ImageJ composite: raw | soma, dendrite, axon outlines | counted | rejected (QC_COLORS)."""
    fish = imread(r['fish_path']).astype(np.uint16)
    chans = [fish]
    for c in COMPARTMENTS:
        o = np.zeros_like(fish); o[find_boundaries(masks[c], mode='outer')] = 65535; chans.append(o)
    kept = np.zeros_like(fish); rej = np.zeros_like(fish)
    inside = df[df.compartment != 'Background']
    for _, s in inside.iterrows():
        if s.kept:
            if QC_COUNTED_AS_RING:
                for rad in (QC_SPOT_RADIUS, QC_SPOT_RADIUS + 1):          # 2 px thick ring
                    rr, cc = circle_perimeter(int(s.spot_y), int(s.spot_x), rad, shape=fish.shape)
                    kept[rr, cc] = 65535
            else:
                rr, cc = disk((s.spot_y, s.spot_x), QC_SPOT_RADIUS, shape=fish.shape); kept[rr, cc] = 65535
        else:
            rr, cc = circle_perimeter(int(s.spot_y), int(s.spot_x), QC_SPOT_RADIUS, shape=fish.shape)
            rej[rr, cc] = 65535
    chans += [kept, rej]
    lo, hi = np.percentile(fish, [1, 99.8])
    tifffile.imwrite(path, np.stack(chans), imagej=True, compression='zlib',
                     resolution=(1 / px_um, 1 / px_um),
                     metadata={'axes': 'CYX', 'mode': 'composite', 'unit': 'um',
                               'Ranges': (float(lo), float(hi)) + (0.0, 65535.0) * 5,
                               'LUTs': [lut(QC_COLORS[k]) for k in
                                        ('raw', 'soma', 'dendrite', 'axon', 'counted', 'rejected')]})


def analyse(inv, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = os.path.join(root, RESULTS_FOLDER_NAME)
    os.makedirs(out, exist_ok=True)
    cut_file = os.path.join(out, 'intensity_cutoffs.csv')
    manual = {}
    if os.path.exists(cut_file):
        old = pd.read_csv(cut_file)
        if 'cutoff_manual' in old:
            manual = old.dropna(subset=['cutoff_manual']).set_index('pair')['cutoff_manual'].to_dict()

    rng = np.random.default_rng(RANDOM_SEED)
    count_rows, kept_tables, cut_rows, hist_data = [], [], [], []
    for pair, g in inv.groupby('pair'):
        tables = {}
        for _, r in g.iterrows():
            cp = cache_path(r)
            if not os.path.exists(cp):
                continue
            df = pd.read_csv(cp)
            df['reason'], df['diameter_nm'] = (base_filter(df, r.px_um, r.setup, r.replicate, r.timepoint, pair)
                                               if len(df) else ('', np.nan))
            tables[r.image_id] = df
        if not tables:
            continue
        # --- intensity cutoff for this pair (control + protease pooled, cell compartments) ---
        pool = pd.concat([t[(t.reason == '') & (t.compartment != 'Background')] for t in tables.values()])
        auto, D = (np.nan, np.nan)
        if len(pool) >= MIN_SPOTS_FOR_CUTOFF:
            auto, D = bright_cutoff(pool['integrated_intensity'].to_numpy())
        dip = valley_ratio(pool['integrated_intensity'].to_numpy(), auto) if np.isfinite(auto) else np.nan
        if INTENSITY_CUTOFF == 'manual' and pair in manual:
            cutoff, how = float(manual[pair]), 'manual'
        else:
            cutoff, how = None, 'none'
        cut_rows.append(dict(pair=pair, setup='/'.join(sorted(set(g.setup))), n_spots_pooled=len(pool),
                             cutoff_proposed=round(auto, 1) if np.isfinite(auto) else np.nan,
                             bimodality_D=round(D, 2) if np.isfinite(D) else np.nan,
                             dip_ratio=round(dip, 2) if np.isfinite(dip) else np.nan,
                             pct_below_proposed=round(100 * np.mean(pool['integrated_intensity'] < auto), 1)
                             if np.isfinite(auto) else np.nan,
                             cutoff_manual=manual.get(pair, np.nan), cutoff_used=cutoff))
        hist_data.append((pair, pool['integrated_intensity'].to_numpy(), cutoff, auto, D, dip))

        # --- apply, count, outputs per image ---
        qc_pick = {}
        for folder, gf in g.groupby('folder'):
            ids = list(gf.image_id)
            n = len(ids) if QC_OVERLAYS_PER_FOLDER == 'all' else min(int(QC_OVERLAYS_PER_FOLDER), len(ids))
            qc_pick.update({i: True for i in rng.permutation(ids)[:n]})
        for _, r in g.iterrows():
            if r.image_id not in tables:
                continue
            df = tables[r.image_id]
            if cutoff is not None:
                low = (df.reason == '') & (df.integrated_intensity < cutoff)
                df.loc[low, 'reason'] = 'below_intensity_cutoff'
            df['kept'] = df.reason == ''
            areas = pd.read_csv(cache_path(r).replace('_spots.csv.gz', '_areas.csv'), index_col=0).iloc[:, 0]
            for c in COMPARTMENTS:
                dc = df[df.compartment == c]
                a_um2 = areas[c] * r.px_um ** 2
                k = dc[dc.kept]
                count_rows.append(dict(timepoint=r.timepoint, replicate=r.replicate, probe=r.probe,
                                       condition=r.condition, pair=pair, setup=r.setup, folder=r.folder,
                                       image_id=r.image_id, compartment=c, n_spots=len(k),
                                       n_spots_before_filter=len(dc), area_um2=round(a_um2, 2),
                                       density_per_um2=len(k) / a_um2 if a_um2 > 0 else np.nan,
                                       median_integrated_intensity=k.integrated_intensity.median()))
            kt = df[df.kept & (df.compartment != 'Background')].copy()
            for col in ['timepoint', 'replicate', 'probe', 'condition', 'pair', 'setup', 'image_id']:
                kt[col] = r[col]
            kept_tables.append(kt.drop(columns=['reason', 'kept']))
            df.to_csv(cache_path(r).replace('_spots.csv.gz', '_spots_filtered.csv.gz'), index=False)
            if qc_pick.get(r.image_id):
                qd = os.path.join(root, r.folder, 'QC_Overlays_v4')
                os.makedirs(qd, exist_ok=True)
                save_qc(r, build_masks(r), df, r.px_um, os.path.join(qd, f"{r.image_id}_QC_v4.tif"))
        print(f"  {pair:22s} {len(tables):3d} images, intensity cutoff: {how}"
              + (f" ({cutoff:.0f})" if cutoff is not None else ""))

    counts = pd.DataFrame(count_rows)
    counts.to_csv(os.path.join(out, 'counts_per_image.csv'), index=False)
    if kept_tables:
        pd.concat(kept_tables, ignore_index=True).to_csv(os.path.join(out, 'kept_spots_all_images.csv.gz'),
                                                         index=False)
    pd.DataFrame(cut_rows).to_csv(cut_file, index=False)
    for folder, gf in counts.groupby('folder'):
        gf.to_csv(os.path.join(root, folder, 'SpotCounts_v4.csv'), index=False)

    # --- pair summary: protease vs control within each pair ---
    agg = counts.groupby(['pair', 'timepoint', 'replicate', 'probe', 'compartment', 'condition']).agg(
        n_images=('image_id', 'nunique'), mean_density=('density_per_um2', 'mean'),
        median_single_spot_intensity=('median_integrated_intensity', 'median')).reset_index()
    wide = agg.pivot_table(index=['pair', 'timepoint', 'replicate', 'probe', 'compartment'],
                           columns='condition', values=['n_images', 'mean_density',
                                                        'median_single_spot_intensity']).reset_index()
    wide.columns = ['_'.join(c).strip('_') for c in wide.columns]
    if {'mean_density_protease', 'mean_density_control'}.issubset(wide.columns):
        wide['density_ratio_protease_vs_control'] = wide.mean_density_protease / wide.mean_density_control
    if {'median_single_spot_intensity_protease', 'median_single_spot_intensity_control'}.issubset(wide.columns):
        wide['spot_brightness_ratio_protease_vs_control'] = (wide.median_single_spot_intensity_protease /
                                                             wide.median_single_spot_intensity_control)
    wide.to_csv(os.path.join(out, 'pair_summary.csv'), index=False)

    # --- intensity cutoff report ---
    n = len(hist_data)
    if n:
        cols = 4
        fig, axs = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(4.2 * cols, 2.8 * np.ceil(n / cols)),
                                squeeze=False)
        for a, (pair, vals, cutoff, auto, D, dip) in zip(axs.flat, hist_data):
            v = np.log10(vals[vals > 0])
            if len(v):
                a.hist(v, bins=60, color='0.5')
                a.set_yscale('log')
            if np.isfinite(auto):
                a.axvline(np.log10(auto), color='tab:blue', ls=':', lw=1.2)
            if cutoff is not None:
                a.axvline(np.log10(cutoff), color='r', ls='--')
            a.set_title(f"{pair}  D={D:.1f} dip={dip:.2f}" if np.isfinite(D) else pair, fontsize=8)
            a.tick_params(labelsize=7)
        for a in list(axs.flat)[n:]:
            a.axis('off')
        fig.suptitle('Integrated intensity (background-subtracted) of all spots that passed size/SNR filters, per '
                     'control/protease pair (x: log10, y: log counts). Blue dotted = proposed cutoff (NOT applied); '
                     'red dashed = cutoff applied (cutoff_manual). dip < ~0.8 = visible gap between populations.',
                     fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(os.path.join(out, 'intensity_cutoffs_per_pair.png'), dpi=110)
        plt.close(fig)

    with open(os.path.join(out, 'Analysis_parameters_v4.txt'), 'w') as fh:
        for k in ['VISIVIEW_PX_UM', 'PROB_THRESHOLD', 'DENDRITE_DILATION_UM', 'AXON_DILATION_UM', 'MIN_DIAMETER_NM', 'MAX_DIAMETER_NM', 'SATURATION_LEVEL',
                  'MIN_SNR_BY_SETUP', 'MIN_SNR_OVERRIDES', 'MIN_SNR', 'INTENSITY_CUTOFF', 'MIN_SPOTS_FOR_CUTOFF', 'FIT_WINDOW']:
            fh.write(f"{k} = {globals()[k]!r}\n")
        for r in cut_rows:
            if r['cutoff_used'] is not None:
                fh.write(f"intensity cutoff {r['pair']} = {r['cutoff_used']}\n")
        for k in []:
            fh.write(f"{k} = {globals()[k]!r}\n")
    print(f"\nResults: {out}")


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    root = ROOT_FOLDER
    if root is None:
        from tkinter import Tk, filedialog
        Tk().withdraw()
        root = filedialog.askdirectory(title='Select the top folder (e.g. "biological replicates")')
    root = os.path.abspath(root)

    print("Scanning project ...")
    inv, problems = discover(root)
    for folder, iid, msg in problems:
        print(f"  WARNING {folder} | {iid}: {msg}")
    if not len(inv):
        raise SystemExit("No complete image sets found.")
    print(f"{len(inv)} images in {inv.folder.nunique()} folders, {inv.pair.nunique()} control/protease pairs, "
          f"setups: {', '.join(sorted(inv.setup.unique()))}")
    os.makedirs(os.path.join(root, RESULTS_FOLDER_NAME), exist_ok=True)
    inv.to_csv(os.path.join(root, RESULTS_FOLDER_NAME, 'inventory.csv'), index=False)

    print(f"\nPass 1: fitting spots ({N_WORKERS} parallel processes) ...")
    todo = inv.to_dict('records')
    t0 = time.time()
    if N_WORKERS > 1:
        from multiprocessing import Pool
        with Pool(N_WORKERS) as pool:
            for k, (iid, n, msg) in enumerate(pool.imap_unordered(process_image, todo), 1):
                if msg != 'cached':
                    print(f"  [{k}/{len(todo)}] {iid}: {msg}" + (f" ({n} spots)" if n else "")
                          + f"  ~{(time.time() - t0) / k * (len(todo) - k) / 60:.0f} min left")
    else:
        for k, r in enumerate(todo, 1):
            iid, n, msg = process_image(r)
            if msg != 'cached':
                print(f"  [{k}/{len(todo)}] {iid}: {msg}" + (f" ({n} spots)" if n else ""))

    print("\nSNR cutoff per imaging batch (replicate x timepoint x setup):")
    snr_tab = inv.assign(min_snr=[min_snr_for(a, b, c, d) for a, b, c, d in
                                  zip(inv.setup, inv.replicate, inv.timepoint, inv.pair)])
    for (rep, tp, st_), g in snr_tab.groupby(['replicate', 'timepoint', 'setup']):
        vals = sorted(set(g.min_snr))
        print(f"  {rep} {tp:6s} {st_:26s} {len(g):5d} images -> MIN_SNR "
              + ', '.join(f'{v:g}' for v in vals) + ('  (differs between pairs)' if len(vals) > 1 else ''))
    snr_tab.groupby(['replicate', 'timepoint', 'setup', 'pair']).min_snr.first().reset_index() \
        .to_csv(os.path.join(root, RESULTS_FOLDER_NAME, 'SNR_cutoff_per_pair.csv'), index=False)
    print("\nPass 2: filters, intensity cutoffs, outputs ...")
    analyse(inv, root)
