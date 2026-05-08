import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Confirmation of [3+3] macrocycle formation in the aerosol reactor
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme('talk', 'ticks', font='Arial', font_scale=0.8, rc={'svg.fonttype': 'none'})
    return mo, np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    from pathlib import Path

    _default = Path('macrocycle')
    if not _default.is_dir():
        _default = Path.cwd()

    data_root_text = mo.ui.text(value=str(_default.resolve()), full_width=True, label='Data root')
    data_root_browser = mo.ui.file_browser(
        initial_path=_default,
        selection_mode='directory',
        multiple=False,
        label='(optional) browse for a folder, tick its checkbox to apply',
    )
    mo.vstack([data_root_text, data_root_browser])
    return Path, data_root_browser, data_root_text


@app.cell(hide_code=True)
def _(Path, data_root_browser, data_root_text):
    # Prefer a directory ticked in the file browser; otherwise fall back to the text field.
    if data_root_browser.value:
        data_root = Path(data_root_browser.path(0))
    else:
        data_root = Path(data_root_text.value).expanduser()

    out_dir = (data_root / 'out')
    out_dir.mkdir(exist_ok=True)
    data_root
    return data_root, out_dir


@app.cell
def _(data_root, pd):
    _cols = ['mz', 'intensity']
    high_frag = pd.read_csv(data_root / '28 scans high_frag.csv', header=None, names=_cols, encoding='utf-8-sig')
    pure_macro = pd.read_csv(data_root / '42 scans pure_macrocycle.csv', header=None, names=_cols, encoding='utf-8-sig')

    def _norm(df):
        out = df.copy()
        out['intensity_norm'] = out['intensity'] / out['intensity'].max()
        return out

    high_frag = _norm(high_frag)
    pure_macro = _norm(pure_macro)
    return high_frag, pure_macro


@app.cell(hide_code=True)
def _(mo):
    window = mo.ui.slider(1, 20, value=1, label='Smoothing window (points)', show_value=True)
    window
    return (window,)


@app.cell(hide_code=True)
def _(boundaries, high_frag, out_dir, plt, pure_macro, smooth, sns, window):
    _fig, _ax = plt.subplots(figsize=(10, 4))

    _ax.plot(pure_macro['mz'], smooth(pure_macro['intensity_norm'], window.value),
             color='C0', label='Pre-formed macrocycle')
    _ax.plot(high_frag['mz'], smooth(high_frag['intensity_norm'], window.value),
             color='C3', alpha=0.8, label='Aerosol reaction mixture')

    for _b in boundaries:
        _ax.axvline(_b, color='0.4', ls=':', lw=1, ymax=0.7)

    _ax.set_xlabel('m/z')
    _ax.set_ylabel('Normalized intensity')
    _ax.set_ylim(0, 1.05)
    _ax.legend(frameon=False)
    sns.despine(ax=_ax)

    _fig.tight_layout()
    _fig.savefig(out_dir / 'Comparison.svg', transparent=True, bbox_inches='tight')
    _fig.savefig(out_dir / 'Comparison.png', transparent=True, bbox_inches='tight')
    _fig
    return


@app.cell
def _():
    boundaries = [636.8, 638.05, 639.0, 640.0, 641.0]
    return (boundaries,)


@app.cell(hide_code=True)
def _(boundaries, high_frag, np, pd, pure_macro):
    def _integrate_windows(df, edges):
        mz = df['mz'].to_numpy()
        y = df['intensity'].to_numpy()
        areas = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (mz >= lo) & (mz <= hi)
            areas.append(np.trapezoid(y[mask], mz[mask]))
        return np.array(areas)

    _labels = [f'[{lo:.2f}, {hi:.2f}]' for lo, hi in zip(boundaries[:-1], boundaries[1:])]
    _window_kind = ['base peak'] + [f'isotope +{i}' for i in range(1, len(_labels))]

    _rows = []
    for _name, _df in [('Pre-formed macrocycle', pure_macro),
                       ('Aerosol reaction mixture', high_frag)]:
        _areas = _integrate_windows(_df, boundaries)
        _rel = 100.0 * _areas / _areas[0]
        for _lbl, _kind, _a, _r in zip(_labels, _window_kind, _areas, _rel):
            _rows.append({
                'sample': _name,
                'window': _lbl,
                'kind': _kind,
                'area': _a,
                '% of base peak': _r,
            })

    isotope_table = pd.DataFrame(_rows)
    isotope_table
    return


@app.cell(hide_code=True)
def _(np, pd):
    def read_spec(path):
        return pd.read_csv(path, header=None, names=['mz', 'intensity'], encoding='utf-8-sig')

    def smooth(y, w):
        """Centered moving average; w=1 is a no-op. Edges use available points."""
        if w is None or w <= 1:
            return np.asarray(y)
        return pd.Series(np.asarray(y, dtype=float)).rolling(int(w), center=True, min_periods=1).mean().to_numpy()

    return read_spec, smooth


@app.cell(hide_code=True)
def _(mo):
    picker = mo.ui.dropdown({'Low source voltage': 'low_frag', 'Typical source voltage': 'standard', 'High source voltage': 'high_frag'}, value='Typical source voltage')
    show_macrocycle = mo.ui.checkbox(value=True, label='Pre-formed macrocycle')
    show_reaction  = mo.ui.checkbox(value=True, label='Aerosol reaction mixture')
    layout = mo.ui.radio(['Superimposed', 'Stacked'], value='Superimposed', label='Layout', inline=True)
    legend_loc = mo.ui.radio(
        {'Top left': 'upper left', 'Top middle': 'upper center', 'Top right': 'upper right'},
        value='Top middle', label='Legend', inline=True,
    )
    mo.hstack([picker, show_macrocycle, show_reaction, layout, legend_loc], justify='start')
    return layout, legend_loc, picker, show_macrocycle, show_reaction


@app.cell(hide_code=True)
def _(
    data_root,
    layout,
    legend_loc,
    mo,
    np,
    out_dir,
    picker,
    plt,
    read_spec,
    show_macrocycle,
    show_reaction,
    smooth,
    window,
):
    from scipy.signal import find_peaks

    full_macrocycle = read_spec(data_root / f'full_spectrum_macrocycle_{picker.value}.csv')
    full_reaction   = read_spec(data_root / f'full_spectrum_reaction_{picker.value}.csv')

    # Zero out the MS calibrant peak (~620-625 m/z) in both spectra so it does not
    # get annotated or skew the per-trace normalization.
    for _df in (full_macrocycle, full_reaction):
        _df.loc[_df['mz'].between(620, 625), 'intensity'] = 0.0

    _traces = [
        (show_macrocycle.value, 'Pre-formed macrocycle',    full_macrocycle, 'C0', 1.0),
        (show_reaction.value,  'Aerosol reaction mixture', full_reaction,   'C3', 0.8),
    ]
    _active = [t for t in _traces if t[0]]

    _stacked = layout.value == 'Stacked' and len(_active) > 1
    if _stacked:
        _fig, _axes = plt.subplots(len(_active), 1, figsize=(12, 3.2 * len(_active)),
                                   sharex=True)
    else:
        _fig, _ax = plt.subplots(figsize=(12, 4))
        _axes = [_ax] * len(_active)

    _min_sep_mz = 10.0  # minimum peak separation in m/z

    for _ax_i, (_visible, _label, _df, _color, _alpha) in zip(_axes, _active):
        _mz = _df['mz'].to_numpy()
        _y = smooth(_df['intensity'] / _df['intensity'].max(), window.value)
        _ax_i.plot(_mz, _y, color=_color, alpha=_alpha, label=_label)
        if _stacked:
            _ax_i.set_title(_label, loc='center')

        _step = float(np.median(np.diff(_mz)))
        _distance = max(1, int(round(_min_sep_mz / _step)))
        _peaks, _ = find_peaks(_y, height=0.09, distance=_distance)

        for _i in _peaks:
            _ax_i.annotate(
                f'{_mz[_i]:.1f} ({round(100 * _y[_i])}%)',
                xy=(_mz[_i], _y[_i]),
                xytext=(0, 3),
                textcoords='offset points',
                rotation=90, rotation_mode='anchor',
                ha='left', va='center', multialignment='center',
                fontsize=10, color='black',
                annotation_clip=False,
            )

    # Apply identical cosmetics to every axes used (deduplicated).
    _axes_seen = []
    for a in _axes:
        if not any(a is s for s in _axes_seen):
            _axes_seen.append(a)

    for _a in _axes_seen:
        _a.set_ylabel('Normalized intensity')
        _a.set_ylim(0, 1.0)
        _a.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        _a.spines['top'].set_visible(False)
        _a.spines['right'].set_visible(False)
        if not _stacked:
            # Anchor the legend just above the axes (y > 1) so it lives in the
            # extra space we reserved with subplots_adjust(top=...).
            _anchors = {
                'upper left':   ((0.0, 1.02), 'lower left'),
                'upper center': ((0.5, 1.02), 'lower center'),
                'upper right':  ((1.0, 1.02), 'lower right'),
            }
            _bbox, _legloc = _anchors[legend_loc.value]
            _a.legend(frameon=False, loc=_legloc,
                      bbox_to_anchor=_bbox, bbox_transform=_a.transAxes)
    _axes_seen[-1].set_xlabel('m/z')

    # Reserve extra space above each axes for the vertical annotations.
    if _stacked:
        _fig.subplots_adjust(top=0.88, hspace=0.55)
    else:
        _fig.subplots_adjust(top=0.72)

    _fig.savefig(out_dir / f'Full spectrum_{picker.value}.svg', transparent=True, bbox_inches='tight')
    _fig.savefig(out_dir / f'Full spectrum_{picker.value}.png', transparent=True, bbox_inches='tight', dpi=300)
    mo.mpl.interactive(_fig)
    return


if __name__ == "__main__":
    app.run()
