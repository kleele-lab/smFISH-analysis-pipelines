"""
smFISH SuperPlot editor - change the look and text of all plots interactively, without rerunning anything.

Needs in the SAME folder: smFISH_statistics_superplots.py (latest version) - and, to show these
analyses too: smFISH_compartments_and_areas.py (dendrite vs axon).
Run:  python smFISH_plot_editor.py  -> choose the folder '_density_results' (or 'statistics_and_plots')

Figures available (when the corresponding results exist):
  Density: control vs protease | Density: DIV9 vs DIV27 | Area check: control vs protease
  Dendrite vs axon                       (recomputed from the per-image densities)
  Advanced: <metric> DIV9 vs DIV27 / control vs protease   (from advanced_quantifications.xlsx)
Statistics are always recomputed in the editor from the per-image data, with the selected replicates
(and optionally only the shown probes for the robust model), with the same functions as the scripts.
"""
import os
import json
import inspect
import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, messagebox

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.lines import Line2D

import smFISH_statistics_superplots as SP

if 'plot_name' not in inspect.signature(SP.draw_panel).parameters or not hasattr(SP, 'rlm_pooled'):
    raise SystemExit("Please put the latest smFISH_statistics_superplots.py in the same folder as the editor.")
try:
    import smFISH_compartments_and_areas as CA
except SystemExit:
    CA = None
except ImportError:
    CA = None
for _k, _v in dict(y_limits={}, y_tick_step=None, y_top_percentile=None, y_max=None, show_images=True).items():
    SP.STYLE.setdefault(_k, _v)

ANALYSES = ['Robust linear model', 't-tests']
MARKERS = ['o', 's', '^', 'v', 'D', 'P', 'X', '*', 'h']
FONTS = ['Arial', 'Helvetica', 'Calibri', 'Times New Roman', 'DejaVu Sans']

NUMERIC = [
    ('axis_line_width', 'Axis line width', float, 1.0, 'Axes'),
    ('tick_width', 'Tick width', float, 1.0, 'Axes'),
    ('tick_length', 'Tick length', float, 4.0, 'Axes'),
    ('font_size', 'Axis label font size', float, SP.STYLE['font_size'], 'Axes'),
    ('y_tick_font_size', 'Y tick font size', float, SP.STYLE['font_size'], 'Axes'),
    ('tick_label_size', 'Group label size', float, SP.STYLE['tick_label_size'], 'Axes'),
    ('probe_label_size', 'Probe name size', float, SP.STYLE['probe_label_size'], 'Axes'),
    ('title_size', 'Title size', float, SP.STYLE['title_size'], 'Axes'),
    ('panel_width', 'Panel width (inch)', float, SP.STYLE['panel_width'], 'Axes'),
    ('panel_height', 'Panel height (inch)', float, SP.STYLE['panel_height'], 'Axes'),
    ('group_spacing', 'Distance between probes', float, SP.STYLE['group_spacing'], 'Axes'),
    ('condition_offset', 'Distance between groups', float, SP.STYLE['condition_offset'], 'Axes'),
    ('image_dot_size', 'Image dot size', float, SP.STYLE['image_dot_size'], 'Points'),
    ('image_dot_alpha', 'Image dot transparency (0-1)', float, SP.STYLE['image_dot_alpha'], 'Points'),
    ('image_jitter', 'Image dot spread', float, SP.STYLE['image_jitter'], 'Points'),
    ('mean_dot_size', 'Mean symbol size', float, SP.STYLE['mean_dot_size'], 'Points'),
    ('mean_dot_edgewidth', 'Mean symbol edge width', float, SP.STYLE['mean_dot_edgewidth'], 'Points'),
    ('connect_line_width', 'Connecting line width', float, SP.STYLE['connect_line_width'], 'Points'),
    ('connect_line_alpha', 'Connecting line transparency', float, SP.STYLE['connect_line_alpha'], 'Points'),
    ('summary_line_width', 'Mean bar line width', float, SP.STYLE['summary_line_width'], 'Summary'),
    ('summary_bar_halfwidth', 'Mean bar width', float, SP.STYLE['summary_bar_halfwidth'], 'Summary'),
    ('error_cap_halfwidth', 'Error cap width', float, SP.STYLE['error_cap_halfwidth'], 'Summary'),
    ('bracket_line_width', 'Bracket line width', float, SP.STYLE['bracket_line_width'], 'Summary'),
    ('bracket_height', 'Bracket end height', float, SP.STYLE['bracket_height'], 'Summary'),
    ('significance_font_size', 'Star / p font size', float, SP.STYLE['significance_font_size'], 'Summary'),
    ('p_decimals', 'p-value decimals', int, SP.STYLE['p_decimals'], 'Summary'),
]


# =============================================================================
# Data sources and figure definitions
# =============================================================================
class Sources:
    """Loads the per-image data of all available analyses once."""

    def __init__(self, folder):
        folder = os.path.abspath(folder)
        if os.path.basename(folder) == 'statistics_and_plots':
            folder = os.path.dirname(folder)
        self.res = folder
        self.stats_img = None
        self.adv_img = None
        self.adv_metrics = {}
        xl = os.path.join(folder, 'statistics_and_plots', 't_tests', 'smFISH_master.xlsx')
        if os.path.exists(xl):
            self.stats_img = pd.read_excel(xl, sheet_name='per_image')
        ax = os.path.join(folder, 'advanced_quantifications', 'advanced_quantifications.xlsx')
        if os.path.exists(ax):
            wide = pd.read_excel(ax, sheet_name='per_image')
            mt = pd.read_excel(ax, sheet_name='metrics')
            self.adv_metrics = dict(zip(mt['column'], mt['metric']))
            rows = []
            for col, label in self.adv_metrics.items():
                if col in wide:
                    x = wide[['probe', 'timepoint', 'replicate', 'condition', 'image_id', col]].rename(columns={col: 'value'})
                    x = x[pd.to_numeric(x['value'], errors='coerce').notna()].copy()
                    x['value'] = x['value'].astype(float)
                    x = x[np.isfinite(x['value'])]
                    x['compartment'] = label
                    rows.append(x)
            if rows:
                self.adv_img = pd.concat(rows, ignore_index=True)

    def figures(self):
        f = {}
        if self.stats_img is not None:
            f['Density: control vs protease'] = dict(src='stats', value='density', group='condition',
                                                     row='compartment', col='timepoint',
                                                     ylabel=SP.DENSITY_LABEL, name='SuperPlot_density_control_vs_protease')
            f['Density: DIV9 vs DIV27'] = dict(src='stats', value='density', group='timepoint', row='compartment',
                                              col='condition', ylabel=SP.DENSITY_LABEL,
                                              name='SuperPlot_density_DIV9_vs_DIV27')
            f['Area check: control vs protease'] = dict(src='stats', value='area', group='condition', row='compartment',
                                                       col='timepoint', ylabel=SP.AREA_LABEL,
                                                       name='SuperPlot_AREA_control_vs_protease')
            if CA is not None:
                f['Dendrite vs axon'] = dict(src='dva', value='density', group='compartment', row='condition',
                                             col='timepoint', ylabel=SP.DENSITY_LABEL, name='SuperPlot_Dendrite_vs_Axon')
        if self.adv_img is not None:
            f['Advanced metrics: DIV9 vs DIV27'] = dict(src='adv', value='value', group='timepoint', row='compartment',
                                                       col='condition', ylabel='{row}', name='SuperPlot_advanced_DIV')
            f['Advanced metrics: control vs protease'] = dict(src='adv', value='value', group='condition',
                                                             row='compartment', col='timepoint', ylabel='{row}',
                                                             name='SuperPlot_advanced_ctrl_prot')
        return f

    def data(self, fig, reps, stat_probes, method):
        """Returns image table, replicate means and statistics for a figure and selection."""
        if fig['src'] == 'adv':
            d = self.adv_img
        else:
            d = self.stats_img
        d = d[d.replicate.isin(reps)].copy()
        if fig['src'] == 'dva':
            d = d[d.compartment.isin(CA.COMPARE) & d.condition.isin(CA.CONDITIONS)]
            d_st = d[d.probe.isin(stat_probes)] if stat_probes is not None else d
            _, rm, t, m = CA.dendrite_vs_axon(d_st)
            st = m if method == 'rlm' else t
            rm = SP.replicate_means(d, 'density')
            return d, rm, st
        value = fig['value']
        rm = SP.replicate_means(d, value)
        d_st = d[d.probe.isin(stat_probes)] if stat_probes is not None else d
        rm_st = SP.replicate_means(d_st, value)
        st = SP.test_timepoints(rm_st, value, method) if fig['group'] == 'timepoint' \
            else SP.test_conditions(rm_st, value, method)
        return d, rm, st


def group_values(fig, d):
    if fig['group'] == 'condition':
        return [c for c in SP.CONDITION_ORDER if c in set(d.condition)]
    if fig['group'] == 'timepoint':
        return [t for t in SP.TIMEPOINT_ORDER if t in set(d.timepoint)]
    return list(CA.COMPARE) if CA is not None else ['Dendrite', 'Axon']


def paired_for(fig):
    return SP.timepoints_paired if fig['group'] == 'timepoint' else True


# =============================================================================
class Editor(tk.Tk):
    def __init__(self, folder=None):
        super().__init__()
        self.title('smFISH SuperPlot editor')
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(1450, sw - 40)}x{min(900, sh - 90)}+10+10")   # fit the screen
        self._after = None
        self._loading = False
        self.vars = {}
        self.panel_y = {}          # panel key -> (ymin, ymax, step)
        self.panel_text = {}       # panel key -> (title, ylabel)
        self.group_labels = dict(SP.STYLE['group_labels'])
        self.probe_names = {}
        self.rep_names = {}
        self.v_probe, self.v_rep = {}, {}
        self._cache = {}
        self._build_controls()
        self._build_plot()
        if folder is None:
            folder = filedialog.askdirectory(title="Select the '_density_results' folder")
        if not folder:
            self.destroy(); return
        self.src = Sources(folder)
        self.FIG = self.src.figures()
        if not self.FIG:
            messagebox.showerror('No results', 'No statistics results found in this folder.\n'
                                               'Select the _density_results folder after running the statistics script.')
            self.destroy(); return
        base = self.src.stats_img if self.src.stats_img is not None else self.src.adv_img
        self.cb_fig['values'] = list(self.FIG)
        self.v_figure.set(list(self.FIG)[0])
        reps = sorted(set(base.replicate))
        probes = [p for p in SP.PROBE_ORDER if p in set(base.probe)]
        self._build_colour_tab(reps)
        self._build_data_tab(probes, reps)
        self._build_text_tab()
        self.on_figure_change()

    # ------------------------------------------------------------------ layout
    def _build_controls(self):
        left = ttk.Frame(self, padding=6)
        left.pack(side='left', fill='y')
        sel = ttk.LabelFrame(left, text='Which plot', padding=6)
        sel.pack(fill='x')
        self.v_analysis = tk.StringVar(value=ANALYSES[0])
        self.v_figure = tk.StringVar()
        self.v_row = tk.StringVar(); self.v_col = tk.StringVar()
        ttk.Label(sel, text='Statistics').pack(anchor='w')
        cb = ttk.Combobox(sel, textvariable=self.v_analysis, values=ANALYSES, state='readonly', width=38)
        cb.pack(fill='x'); cb.bind('<<ComboboxSelected>>', lambda e: self.on_figure_change())
        ttk.Label(sel, text='Figure').pack(anchor='w')
        self.cb_fig = ttk.Combobox(sel, textvariable=self.v_figure, state='readonly', width=38)
        self.cb_fig.pack(fill='x'); self.cb_fig.bind('<<ComboboxSelected>>', lambda e: self.on_figure_change())
        ttk.Label(sel, text='Panel').pack(anchor='w')
        self.cb_row = ttk.Combobox(sel, textvariable=self.v_row, state='readonly', width=38)
        self.cb_row.pack(fill='x')
        self.cb_col = ttk.Combobox(sel, textvariable=self.v_col, state='readonly', width=38)
        self.cb_col.pack(fill='x', pady=(2, 0))
        for c in (self.cb_row, self.cb_col):
            c.bind('<<ComboboxSelected>>', lambda e: self.on_panel_change())

        nb = ttk.Notebook(left)          # packed after the Save box, so the Save buttons always stay visible
        names = ['Data', 'Text', 'Axes', 'Y axis', 'Points', 'Colours', 'Summary']
        tabs = {n: ttk.Frame(nb, padding=6) for n in names}
        for n in names:
            nb.add(tabs[n], text=n)
        self.data_tab, self.col_tab = tabs['Data'], tabs['Colours']
        # the Text tab can get long: put it in a scrollable area
        cv = tk.Canvas(tabs['Text'], highlightthickness=0, width=300)
        sb = ttk.Scrollbar(tabs['Text'], orient='vertical', command=cv.yview)
        self.text_tab = ttk.Frame(cv)
        self.text_tab.bind('<Configure>', lambda e: cv.configure(scrollregion=cv.bbox('all')))
        cv.create_window((0, 0), window=self.text_tab, anchor='nw')
        cv.configure(yscrollcommand=sb.set)
        cv.pack(side='left', fill='both', expand=True); sb.pack(side='right', fill='y')
        cv.bind_all('<MouseWheel>', lambda e: cv.yview_scroll(int(-e.delta / 120), 'units')
                    if str(e.widget).startswith(str(cv)) else None)
        rows = {k: 0 for k in tabs}
        for key, label, typ, default, tab in NUMERIC:
            v = tk.StringVar(value=str(default))
            self.vars[key] = (v, typ)
            ttk.Label(tabs[tab], text=label).grid(row=rows[tab], column=0, sticky='w', pady=1)
            ttk.Spinbox(tabs[tab], textvariable=v, width=8, from_=0, to=1000,
                        increment=0.1 if typ is float else 1).grid(row=rows[tab], column=1, sticky='w')
            v.trace_add('write', lambda *a: self.schedule())
            rows[tab] += 1

        t = tabs['Y axis']
        ttk.Label(t, text='For the CURRENT panel (remembered per panel):', foreground='#555').grid(
            row=0, column=0, columnspan=2, sticky='w', pady=(0, 6))
        self.v_ymin, self.v_ymax, self.v_ystep = tk.StringVar(), tk.StringVar(), tk.StringVar()
        for i, (lab, var) in enumerate([('Y minimum (empty = auto)', self.v_ymin),
                                        ('Y maximum (empty = auto)', self.v_ymax),
                                        ('Tick step (empty = auto)', self.v_ystep)], start=1):
            ttk.Label(t, text=lab).grid(row=i, column=0, sticky='w')
            ttk.Entry(t, textvariable=var, width=10).grid(row=i, column=1, sticky='w')
            var.trace_add('write', lambda *a: self.on_y_change())
        ttk.Separator(t).grid(row=4, column=0, columnspan=2, sticky='ew', pady=6)
        self.v_pct = tk.StringVar(value='')
        ttk.Label(t, text='All panels - zoom: top at percentile').grid(row=5, column=0, sticky='w')
        ttk.Entry(t, textvariable=self.v_pct, width=10).grid(row=5, column=1, sticky='w')
        self.v_pct.trace_add('write', lambda *a: self.schedule())
        self.v_log = tk.BooleanVar(value=False)
        ttk.Checkbutton(t, text='Log scale (all panels)', variable=self.v_log, command=self.schedule).grid(
            row=6, column=0, sticky='w')

        t = tabs['Points']; r = rows['Points']
        self.v_show_images = tk.BooleanVar(value=True)
        self.v_connect = tk.BooleanVar(value=True)
        self.v_legend = tk.BooleanVar(value=True)
        for lab, var in [('Show single images', self.v_show_images), ('Connect paired means', self.v_connect),
                         ('Show legend', self.v_legend)]:
            ttk.Checkbutton(t, text=lab, variable=var, command=self.schedule).grid(row=r, column=0, sticky='w'); r += 1

        self.color = dict(SP.STYLE['replicate_colors'])
        self.marker = dict(SP.STYLE['replicate_markers'])
        self.v_edgecolor = SP.STYLE['mean_dot_edgecolor']
        self.v_summary_color = SP.STYLE['summary_line_color']

        t = tabs['Summary']; r = rows['Summary']
        self.v_error = tk.StringVar(value=SP.STYLE['summary_error'])
        self.v_sig = tk.StringVar(value=SP.STYLE['show_significance'])
        self.v_ns = tk.BooleanVar(value=SP.STYLE['show_ns'])
        for lab, var, vals in [('Error bars', self.v_error, ['sd', 'sem', 'none']),
                               ('Significance', self.v_sig, ['stars', 'p', 'both', 'none'])]:
            ttk.Label(t, text=lab).grid(row=r, column=0, sticky='w')
            c = ttk.Combobox(t, textvariable=var, values=vals, state='readonly', width=8)
            c.grid(row=r, column=1, sticky='w'); c.bind('<<ComboboxSelected>>', lambda e: self.schedule()); r += 1
        ttk.Checkbutton(t, text="Show 'ns'", variable=self.v_ns, command=self.schedule).grid(row=r, column=0, sticky='w')

        b = ttk.LabelFrame(left, text='Save', padding=6)
        b.pack(fill='x', side='bottom')
        for row_buttons in [[('Export this panel…', self.export_panel), ('Export all panels…', self.export_all)],
                            [('Export whole figure (all panels in one)…', self.export_combined)],
                            [('Save style…', self.save_style), ('Load style…', self.load_style)]]:
            fr = ttk.Frame(b); fr.pack(fill='x', pady=1)
            for lab, cmd in row_buttons:
                ttk.Button(fr, text=lab, command=cmd).pack(side='left', fill='x', expand=True, padx=1)
        nb.pack(fill='both', expand=True, pady=6)

    def _build_data_tab(self, probes, reps):
        t = self.data_tab
        self.v_stats_shown = tk.BooleanVar(value=False)
        self.v_note = tk.BooleanVar(value=True)
        ttk.Label(t, text='Probes shown in the graph').grid(row=0, column=0, sticky='w')
        r = 1
        for p in probes:
            self.v_probe[p] = tk.BooleanVar(value=True)
            ttk.Checkbutton(t, text=p, variable=self.v_probe[p], command=self.on_selection_change).grid(row=r, column=0, sticky='w')
            r += 1
        ttk.Separator(t).grid(row=r, column=0, sticky='ew', pady=6); r += 1
        ttk.Label(t, text='Biological replicates included\n(unticked = removed from the graph\nAND from the statistics)').grid(
            row=r, column=0, sticky='w'); r += 1
        for rep in reps:
            self.v_rep[rep] = tk.BooleanVar(value=True)
            ttk.Checkbutton(t, text=rep, variable=self.v_rep[rep], command=self.on_selection_change).grid(row=r, column=0, sticky='w')
            r += 1
        ttk.Separator(t).grid(row=r, column=0, sticky='ew', pady=6); r += 1
        ttk.Checkbutton(t, text='Robust model: fit only the shown probes', variable=self.v_stats_shown,
                        command=self.on_selection_change).grid(row=r, column=0, sticky='w'); r += 1
        ttk.Label(t, text='(t-tests are per probe and do not change)', foreground='#555').grid(row=r, column=0, sticky='w'); r += 1
        ttk.Checkbutton(t, text='Note excluded replicates on the graph', variable=self.v_note,
                        command=self.schedule).grid(row=r, column=0, sticky='w', pady=(6, 0)); r += 1
        ttk.Button(t, text='Export statistics of this figure…', command=self.export_stats).grid(
            row=r, column=0, sticky='ew', pady=(8, 0))

    def _build_text_tab(self):
        t = self.text_tab
        for w in t.winfo_children():
            w.destroy()
        r = 0
        ttk.Label(t, text='CURRENT panel (empty = default):', foreground='#555').grid(row=r, column=0, columnspan=2, sticky='w'); r += 1
        self.v_title_txt, self.v_ylabel_txt = tk.StringVar(), tk.StringVar()
        for lab, var in [('Panel title', self.v_title_txt), ('Y axis label', self.v_ylabel_txt)]:
            ttk.Label(t, text=lab).grid(row=r, column=0, sticky='w')
            ttk.Entry(t, textvariable=var, width=26).grid(row=r, column=1, sticky='w'); r += 1
            var.trace_add('write', lambda *a: self.on_text_change())
        ttk.Button(t, text='Use this y label for all panels of the figure',
                   command=self.ylabel_to_all).grid(row=r, column=0, columnspan=2, sticky='ew', pady=(2, 4)); r += 1
        self.v_show_title = tk.BooleanVar(value=True)
        ttk.Checkbutton(t, text='Show panel titles', variable=self.v_show_title, command=self.schedule).grid(
            row=r, column=0, sticky='w'); r += 1
        ttk.Label(t, text='Font').grid(row=r, column=0, sticky='w')
        self.v_font = tk.StringVar(value=SP.STYLE['font'])
        c = ttk.Combobox(t, textvariable=self.v_font, values=FONTS, width=16)
        c.grid(row=r, column=1, sticky='w'); c.bind('<<ComboboxSelected>>', lambda e: self.schedule())
        self.v_font.trace_add('write', lambda *a: self.schedule()); r += 1
        ttk.Separator(t).grid(row=r, column=0, columnspan=2, sticky='ew', pady=6); r += 1
        ttk.Label(t, text='Labels under the groups (all figures):', foreground='#555').grid(row=r, column=0, columnspan=2, sticky='w'); r += 1
        self.v_group = {}
        for g in SP.CONDITION_ORDER + SP.TIMEPOINT_ORDER + ['Dendrite', 'Axon']:
            self.v_group[g] = tk.StringVar(value=self.group_labels.get(g, g))
            ttk.Label(t, text=g).grid(row=r, column=0, sticky='w')
            ttk.Entry(t, textvariable=self.v_group[g], width=14).grid(row=r, column=1, sticky='w')
            self.v_group[g].trace_add('write', lambda *a: self.schedule()); r += 1
        ttk.Label(t, text='Probe names:', foreground='#555').grid(row=r, column=0, columnspan=2, sticky='w', pady=(6, 0)); r += 1
        self.v_pname = {}
        for p in self.v_probe:
            self.v_pname[p] = tk.StringVar(value=self.probe_names.get(p, p))
            ttk.Label(t, text=p).grid(row=r, column=0, sticky='w')
            ttk.Entry(t, textvariable=self.v_pname[p], width=14).grid(row=r, column=1, sticky='w')
            self.v_pname[p].trace_add('write', lambda *a: self.schedule()); r += 1
        ttk.Label(t, text='Legend names of replicates:', foreground='#555').grid(row=r, column=0, columnspan=2, sticky='w', pady=(6, 0)); r += 1
        self.v_rname = {}
        for rep in self.v_rep:
            self.v_rname[rep] = tk.StringVar(value=self.rep_names.get(rep, f'replicate {rep}'))
            ttk.Label(t, text=rep).grid(row=r, column=0, sticky='w')
            ttk.Entry(t, textvariable=self.v_rname[rep], width=14).grid(row=r, column=1, sticky='w')
            self.v_rname[rep].trace_add('write', lambda *a: self.schedule()); r += 1

    def _build_colour_tab(self, reps):
        for w in self.col_tab.winfo_children():
            w.destroy()
        ttk.Label(self.col_tab, text='Replicate colours and symbols').grid(row=0, column=0, columnspan=3, sticky='w')
        for i, rep in enumerate(reps, start=1):
            self.color.setdefault(rep, '#888888'); self.marker.setdefault(rep, 'o')
            ttk.Label(self.col_tab, text=rep).grid(row=i, column=0, sticky='w')
            tk.Button(self.col_tab, width=6, bg=self.color[rep], command=lambda r=rep: self.pick_colour(r)).grid(
                row=i, column=1, padx=4, pady=2)
            mv = tk.StringVar(value=self.marker[rep])
            c = ttk.Combobox(self.col_tab, textvariable=mv, values=MARKERS, width=4, state='readonly')
            c.grid(row=i, column=2)
            c.bind('<<ComboboxSelected>>', lambda e, r=rep, v=mv: (self.marker.__setitem__(r, v.get()), self.schedule()))
        n = len(reps) + 1
        for j, (lab, attr) in enumerate([('Mean symbol edge', 'v_edgecolor'), ('Mean bar / error', 'v_summary_color')]):
            ttk.Label(self.col_tab, text=lab).grid(row=n + j, column=0, sticky='w', pady=(8 if j == 0 else 2, 2))
            tk.Button(self.col_tab, width=6, bg=getattr(self, attr), command=lambda a=attr: self.pick_other(a)).grid(
                row=n + j, column=1, padx=4)
        self._reps = reps

    def pick_colour(self, rep):
        c = colorchooser.askcolor(color=self.color[rep], title=f'Colour for {rep}')[1]
        if c:
            self.color[rep] = c; self._build_colour_tab(self._reps); self.schedule()

    def pick_other(self, attr):
        c = colorchooser.askcolor(color=getattr(self, attr))[1]
        if c:
            setattr(self, attr, c); self._build_colour_tab(self._reps); self.schedule()

    def _build_plot(self):
        right = ttk.Frame(self)
        right.pack(side='right', fill='both', expand=True)
        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        NavigationToolbar2Tk(self.canvas, right)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.canvas.get_tk_widget().bind('<Configure>', lambda e: self.schedule(), add='+')
        self.status = ttk.Label(right, text='', foreground='#555')
        self.status.pack(anchor='w')

    # ------------------------------------------------------------------ selection
    def shown_probes(self):
        return [p for p in SP.PROBE_ORDER if p in self.v_probe and self.v_probe[p].get()]

    def included_reps(self):
        return [r for r, v in sorted(self.v_rep.items()) if v.get()]

    def excluded_reps(self):
        return [r for r, v in sorted(self.v_rep.items()) if not v.get()]

    def on_selection_change(self):
        if not self.included_reps():
            messagebox.showwarning('No replicate', 'At least one replicate must stay included.')
            next(iter(self.v_rep.values())).set(True)
        self.on_figure_change()

    def method(self):
        return 'rlm' if self.v_analysis.get() == 'Robust linear model' else 't'

    def current(self):
        fig = self.FIG[self.v_figure.get()]
        reps = tuple(self.included_reps())
        stat_probes = tuple(self.shown_probes()) if (self.v_stats_shown.get() and self.method() == 'rlm') else None
        key = (self.v_figure.get(), self.method(), reps, stat_probes)
        if key not in self._cache:
            self._cache[key] = self.src.data(fig, reps, stat_probes, self.method())
        d, rm, st = self._cache[key]
        return fig, d, rm, st

    def on_figure_change(self):
        fig, d, rm, st = self.current()
        rows = list(dict.fromkeys(d[fig['row']]))
        order = SP.COMPARTMENT_ORDER + SP.CONDITION_ORDER + SP.TIMEPOINT_ORDER
        rows = sorted(rows, key=lambda c: order.index(c) if c in order else 99)
        if fig['src'] == 'adv':
            rows = [m for m in self.src.adv_metrics.values() if m in set(d[fig['row']])]
        cols = list(dict.fromkeys(d[fig['col']]))
        if fig['col'] == 'condition' and fig['group'] == 'timepoint':
            cols = [c for c in cols if c in SP.TIMEPOINT_CONDITIONS]
        cols = sorted(cols, key=lambda c: order.index(c) if c in order else 99)
        self.cb_row['values'] = rows; self.cb_col['values'] = cols
        if self.v_row.get() not in rows:
            self.v_row.set(rows[0] if rows else '')
        if self.v_col.get() not in cols:
            self.v_col.set(cols[0] if cols else '')
        self.on_panel_change()

    def panel_key(self, row=None, col=None):
        return f"{self.FIG[self.v_figure.get()]['name']}/{row or self.v_row.get()} | {col or self.v_col.get()}"

    def on_panel_change(self):
        self._loading = True
        lo, hi, st = self.panel_y.get(self.panel_key(), ('', '', ''))
        self.v_ymin.set(lo); self.v_ymax.set(hi); self.v_ystep.set(st)
        tt, yl = self.panel_text.get(self.panel_key(), ('', ''))
        self.v_title_txt.set(tt); self.v_ylabel_txt.set(yl)
        self._loading = False
        self.redraw()

    def on_y_change(self):
        if not self._loading:
            self.panel_y[self.panel_key()] = (self.v_ymin.get(), self.v_ymax.get(), self.v_ystep.get())
            self.schedule()

    def on_text_change(self):
        if not self._loading:
            self.panel_text[self.panel_key()] = (self.v_title_txt.get(), self.v_ylabel_txt.get())
            self.schedule()

    def ylabel_to_all(self):
        yl = self.v_ylabel_txt.get()
        for r in self.cb_row['values']:
            for c in self.cb_col['values']:
                k = self.panel_key(r, c)
                self.panel_text[k] = (self.panel_text.get(k, ('', ''))[0], yl)
        self.schedule()

    # ------------------------------------------------------------------ style
    def num(self, key):
        v, typ = self.vars[key]
        try:
            return typ(float(v.get()))
        except ValueError:
            return None

    def apply_style(self):
        S = SP.STYLE
        for key, *_ in NUMERIC:
            val = self.num(key)
            if val is not None and key in S:
                S[key] = val
        S['font'] = self.v_font.get() or 'Arial'
        S['replicate_colors'] = dict(self.color)
        S['replicate_markers'] = dict(self.marker)
        S['mean_dot_edgecolor'] = self.v_edgecolor
        S['summary_line_color'] = self.v_summary_color
        S['summary_error'] = self.v_error.get()
        S['show_significance'] = self.v_sig.get()
        S['show_ns'] = self.v_ns.get()
        S['show_images'] = self.v_show_images.get()
        S['connect_paired'] = self.v_connect.get()
        S['y_scale'] = 'log' if self.v_log.get() else 'linear'
        S['group_labels'] = {g: (v.get() or g) for g, v in self.v_group.items()}
        try:
            S['y_top_percentile'] = float(self.v_pct.get()) if self.v_pct.get().strip() else None
        except ValueError:
            S['y_top_percentile'] = None
        S['y_limits'], S['y_tick_step'], S['y_max'], S['y_min'] = {}, {}, None, 0
        for key, (lo, hi, step) in self.panel_y.items():
            try:
                if hi.strip():
                    S['y_limits'][key] = (float(lo) if lo.strip() else 0.0, float(hi))
                elif lo.strip():
                    S['y_limits'][key] = (float(lo), np.inf)
                if step.strip():
                    S['y_tick_step'][key] = float(step)
            except ValueError:
                pass
        SP.setup_matplotlib()

    def schedule(self):
        if self._after:
            self.after_cancel(self._after)
        self._after = self.after(250, self.redraw)

    # ------------------------------------------------------------------ drawing
    def legend_handles(self, reps):
        S = SP.STYLE
        return [Line2D([], [], marker=S['replicate_markers'].get(r, 'o'), ls='', color=S['replicate_colors'].get(r, 'grey'),
                       markeredgecolor='black', markersize=7, label=self.v_rname.get(r, tk.StringVar(value=r)).get() or r)
                for r in reps]

    def draw_on(self, ax, row, col, legend=True):
        fig, d, rm, st = self.current()
        S = SP.STYLE
        rc, cc = fig['row'], fig['col']
        sub = d[(d[rc] == row) & (d[cc] == col)]
        subm = rm[(rm[rc] == row) & (rm[cc] == col)]
        st_rows = st[(st[rc] == row) & (st[cc] == col)] if len(st) and rc in st and cc in st else pd.DataFrame()
        groups = group_values(fig, d)
        probes = [p for p in self.shown_probes() if p in set(d.probe)] or [p for p in SP.PROBE_ORDER if p in set(d.probe)]
        key = f"{fig['name']}/{row} | {col}"
        title_txt, ylab_txt = self.panel_text.get(key, ('', ''))
        ylabel = ylab_txt or fig['ylabel'].replace('{row}', str(row))
        lim = S['y_limits'].get(key)
        if lim is not None and not np.isfinite(lim[1]):
            S['y_limits'].pop(key)
        SP.draw_panel(ax, sub, subm, st_rows, fig['value'], fig['group'], groups, paired_for(fig), probes,
                      ylabel, f"{row} | {col}", fig['name'])
        if lim is not None and not np.isfinite(lim[1]):
            ax.set_ylim(bottom=lim[0]); S['y_limits'][key] = lim
        ax.set_title((title_txt or f"{row} | {col}") if self.v_show_title.get() else '',
                     fontsize=self.num('title_size') or S['title_size'])
        # probe names (drawn as text under the axis by draw_panel)
        for t in ax.texts:
            if t.get_text() in self.v_pname:
                t.set_text(self.v_pname[t.get_text()].get() or t.get_text())
        lw = self.num('axis_line_width') or 1.0
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_linewidth(lw)
        ax.tick_params(axis='y', width=self.num('tick_width') or 1, length=self.num('tick_length') or 4,
                       labelsize=self.num('y_tick_font_size') or S['font_size'])
        ax.tick_params(axis='x', width=self.num('tick_width') or 1, length=self.num('tick_length') or 4)
        ax.yaxis.label.set_size(self.num('font_size') or S['font_size'])
        notes = []
        if self.excluded_reps():
            notes.append('excluded: ' + ', '.join(self.excluded_reps()))
        if self.v_stats_shown.get() and self.method() == 'rlm' and len(self.shown_probes()) < len(self.v_probe):
            notes.append('model fitted on the shown probes')
        if notes and self.v_note.get():
            ax.text(1.0, -0.30, '; '.join(notes), transform=ax.transAxes, ha='right', va='top', fontsize=6, color='0.35')
        if legend and self.v_legend.get():
            ax.legend(handles=self.legend_handles(sorted(set(d.replicate))), loc='upper left', bbox_to_anchor=(1.01, 1.0),
                      frameon=False, fontsize=(self.num('font_size') or 9) - 1)

    def redraw(self):
        self._after = None
        if not hasattr(self, 'FIG') or not self.v_row.get():
            return
        self.apply_style()
        self.fig.clear()
        W, H = self.fig.get_size_inches()
        pw, ph = SP.STYLE['panel_width'], SP.STYLE['panel_height']
        ml, mr, mb, mt = 0.9, (1.6 if self.v_legend.get() else 0.3), 0.9, 0.5
        scale = min(1.0, (W - 0.1) / (pw + ml + mr), (H - 0.1) / (ph + mb + mt))
        aw, ah = pw * scale, ph * scale
        left = (W - (aw + (ml + mr) * scale)) / 2 + ml * scale
        bottom = (H - (ah + (mb + mt) * scale)) / 2 + mb * scale
        ax = self.fig.add_axes([left / W, bottom / H, aw / W, ah / H])
        self.draw_on(ax, self.v_row.get(), self.v_col.get())
        self.canvas.draw()
        exc = self.excluded_reps()
        info = f" | EXCLUDED: {', '.join(exc)} (statistics recomputed)" if exc else ''
        note = '' if scale >= 0.999 else f'   (preview at {scale:.0%}; exports use the exact size)'
        self.status.config(text=f"{self.v_analysis.get()} | {self.v_figure.get()} | {self.v_row.get()} | "
                                f"{self.v_col.get()}{info}{note}")

    # ------------------------------------------------------------------ export
    def _figure_for(self, row, col):
        fig = Figure(figsize=(SP.STYLE['panel_width'] + (1.6 if self.v_legend.get() else 0.2),
                              SP.STYLE['panel_height'] + 0.8))
        ax = fig.add_subplot(111)
        self.draw_on(ax, row, col)
        fig.tight_layout()
        return fig

    def _safe(self, s):
        return ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in str(s))

    def export_panel(self):
        name = f"{self.FIG[self.v_figure.get()]['name']}_{self._safe(self.v_row.get())}_{self._safe(self.v_col.get())}"
        path = filedialog.asksaveasfilename(defaultextension='.pdf', initialfile=name,
                                            filetypes=[('PDF', '*.pdf'), ('SVG', '*.svg'), ('PNG', '*.png'), ('TIFF', '*.tif')])
        if path:
            self.apply_style()
            self._figure_for(self.v_row.get(), self.v_col.get()).savefig(path, dpi=300, bbox_inches='tight')
            self.status.config(text=f'Saved {path}')

    def export_all(self):
        folder = filedialog.askdirectory(title='Folder for the exported panels')
        if not folder:
            return
        self.apply_style()
        n = 0
        for row in self.cb_row['values']:
            for col in self.cb_col['values']:
                f = self._figure_for(row, col)
                base = os.path.join(folder, f"{self.FIG[self.v_figure.get()]['name']}_{self._safe(row)}_{self._safe(col)}")
                for ext in ('pdf', 'png'):
                    f.savefig(f'{base}.{ext}', dpi=300, bbox_inches='tight')
                n += 1
        self.status.config(text=f'Saved {n} panels (PDF + PNG) to {folder}')

    def export_combined(self):
        path = filedialog.asksaveasfilename(defaultextension='.pdf', initialfile=self.FIG[self.v_figure.get()]['name'],
                                            filetypes=[('PDF', '*.pdf'), ('SVG', '*.svg'), ('PNG', '*.png')])
        if not path:
            return
        self.apply_style()
        rows, cols = list(self.cb_row['values']), list(self.cb_col['values'])
        w, h = SP.STYLE['panel_width'], SP.STYLE['panel_height']
        legend = self.v_legend.get()
        fig = Figure(figsize=(w * len(cols) + (1.6 if legend else 0), (h + 0.6) * len(rows)))
        axs = fig.subplots(len(rows), len(cols), squeeze=False)
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                self.draw_on(axs[i, j], r, c, legend=False)
        if legend:
            d = self.current()[1]
            fig.legend(handles=self.legend_handles(sorted(set(d.replicate))), loc='upper right', frameon=False)
        fig.tight_layout(rect=(0, 0, 0.88 if legend else 1, 1))
        fig.savefig(path, dpi=300, bbox_inches='tight')
        self.status.config(text=f'Saved {path}')

    def export_stats(self):
        fig, d, rm, st = self.current()
        exc = self.excluded_reps()
        name = f"stats_{fig['name'].replace('SuperPlot_', '')}_{'rlm' if self.method() == 'rlm' else 'ttest'}"
        if exc:
            name += '_without_' + '_'.join(exc)
        path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV', '*.csv')], initialfile=name)
        if not path:
            return
        out = st.copy()
        out.insert(0, 'replicates_excluded', ', '.join(exc) if exc else 'none')
        out.insert(1, 'statistics', self.v_analysis.get())
        if self.v_stats_shown.get() and self.method() == 'rlm':
            out.insert(2, 'model_fitted_on_probes', ', '.join(self.shown_probes()))
        out.to_csv(path, index=False)
        self.status.config(text=f'Statistics saved: {path}')

    # ------------------------------------------------------------------ style files
    def save_style(self):
        path = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('Style', '*.json')],
                                            initialfile='my_plot_style.json')
        if not path:
            return
        st = {k: self.vars[k][0].get() for k, *_ in NUMERIC}
        st.update(colors=self.color, markers=self.marker, edgecolor=self.v_edgecolor,
                  summary_color=self.v_summary_color, error=self.v_error.get(), sig=self.v_sig.get(),
                  ns=self.v_ns.get(), show_images=self.v_show_images.get(), connect=self.v_connect.get(),
                  legend=self.v_legend.get(), log=self.v_log.get(), pct=self.v_pct.get(), font=self.v_font.get(),
                  show_title=self.v_show_title.get(), panel_y=self.panel_y, panel_text=self.panel_text,
                  group_labels={g: v.get() for g, v in self.v_group.items()},
                  probe_names={p: v.get() for p, v in self.v_pname.items()},
                  replicate_names={r: v.get() for r, v in self.v_rname.items()},
                  probes_shown=self.shown_probes(), replicates_excluded=self.excluded_reps(),
                  model_on_shown_probes=self.v_stats_shown.get(), note_exclusions=self.v_note.get())
        with open(path, 'w') as fh:
            json.dump(st, fh, indent=2)
        self.status.config(text=f'Style saved: {path}')

    def load_style(self):
        path = filedialog.askopenfilename(filetypes=[('Style', '*.json')])
        if not path:
            return
        with open(path) as fh:
            st = json.load(fh)
        self._loading = True
        for k, *_ in NUMERIC:
            if k in st:
                self.vars[k][0].set(st[k])
        self.color.update(st.get('colors', {})); self.marker.update(st.get('markers', {}))
        self.v_edgecolor = st.get('edgecolor', self.v_edgecolor)
        self.v_summary_color = st.get('summary_color', self.v_summary_color)
        for var, k in [(self.v_error, 'error'), (self.v_sig, 'sig'), (self.v_pct, 'pct'), (self.v_font, 'font')]:
            if k in st:
                var.set(st[k])
        for var, k in [(self.v_ns, 'ns'), (self.v_show_images, 'show_images'), (self.v_connect, 'connect'),
                       (self.v_legend, 'legend'), (self.v_log, 'log'), (self.v_show_title, 'show_title'),
                       (self.v_stats_shown, 'model_on_shown_probes'), (self.v_note, 'note_exclusions')]:
            if k in st:
                var.set(bool(st[k]))
        self.panel_y = {k: tuple(v) for k, v in st.get('panel_y', {}).items()}
        self.panel_text = {k: tuple(v) for k, v in st.get('panel_text', {}).items()}
        for src, dst in [('group_labels', self.v_group), ('probe_names', self.v_pname), ('replicate_names', self.v_rname)]:
            for k, v in st.get(src, {}).items():
                if k in dst:
                    dst[k].set(v)
        for p, v in self.v_probe.items():
            v.set(p in st.get('probes_shown', list(self.v_probe)))
        for r, v in self.v_rep.items():
            v.set(r not in st.get('replicates_excluded', []))
        self._loading = False
        self._build_colour_tab(self._reps)
        self.on_figure_change()
        self.status.config(text=f'Style loaded: {path}')


if __name__ == '__main__':
    import sys
    Editor(sys.argv[1] if len(sys.argv) > 1 else None).mainloop()
