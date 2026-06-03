# /// script
# dependencies = ["marimo"]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(
    width="full",
    app_title="[3+3] macrocycle",
    css_file="",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MS evidence of [3+3] macrocycle formation in the aerosol reactor
    """)
    return


@app.cell
def _():
    from urllib import request
    import zipfile
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    data_root = mo.notebook_dir() / "data"
    out_dir = mo.notebook_dir() / "out"
    sns.set_theme(
        "talk", "ticks", font="Arial", font_scale=0.8, rc={"svg.fonttype": "none"}
    )
    return data_root, mo, np, out_dir, pd, plt, request, sns, zipfile


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Download dataset if it does not exist
    """)
    return


@app.cell
def _(data_root, request, zipfile):
    if not data_root.exists():
        data_root.mkdir()
        request.urlretrieve('https://zenodo.org/records/20524239/files/macrocycle_data.zip?download=1', 'macrocycle_data.zip')
        with zipfile.ZipFile("macrocycle_data.zip") as z:
            [data_root.joinpath(f).write_bytes(z.read(m)) for m in z.infolist() if (f := m.filename.split("/")[-1]).lower().endswith(".csv") and not f.startswith('.')]
    return


@app.cell
def _(data_root, out_dir, pd):
    out_dir.mkdir(exist_ok=True)

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
    picker = mo.ui.dropdown({'Low source voltage (20 V)': 'low_frag', 'Typical source voltage (20–50 V)': 'standard', 'High source voltage (40–60 V)': 'high_frag'}, value='Typical source voltage (20–50 V)')
    show_macrocycle = mo.ui.checkbox(value=True, label='Pre-formed macrocycle')
    show_reaction  = mo.ui.checkbox(value=True, label='Aerosol reaction mixture')
    layout = mo.ui.radio(['Superimposed', 'Stacked'], value='Superimposed', label='Layout', inline=True)
    header_loc = mo.ui.radio(
        {'Left': 'left', 'Center': 'center', 'Right': 'right'},
        value='Center', label='Header position', inline=True,
    )
    intensity_mode = mo.ui.radio(['Relative', 'Absolute'], value='Relative', label='Intensity', inline=True)

    mo.vstack([
        mo.hstack([picker, show_macrocycle, show_reaction], justify='start'),
        mo.hstack([layout, intensity_mode, header_loc], justify='start'),
    ])
    return (
        header_loc,
        intensity_mode,
        layout,
        picker,
        show_macrocycle,
        show_reaction,
    )


@app.cell(hide_code=True)
def _(
    data_root,
    header_loc,
    intensity_mode,
    layout,
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
    from matplotlib.ticker import ScalarFormatter

    full_macrocycle = read_spec(data_root / f'full_spectrum_macrocycle_{picker.value}.csv')
    full_reaction   = read_spec(data_root / f'full_spectrum_reaction_{picker.value}.csv')

    # Zero out the MS calibrant peak (~620-625 m/z) in both spectra so it does not
    # get annotated or skew the per-trace normalization.
    for _df in (full_macrocycle, full_reaction):
        _df.loc[_df['mz'].between(620, 625), 'intensity'] = 0.0

    _relative = intensity_mode.value == 'Relative'

    _traces = [
        (show_macrocycle.value, f'Pre-formed macrocycle — {picker.selected_key}',    full_macrocycle, 'C0', 1.0),
        (show_reaction.value,   f'Aerosol reaction mixture — {picker.selected_key}', full_reaction,   'C3', 0.8),
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
    # (peak labels in absolute mode use scientific notation, formatted inline)

    # Track ymax in absolute mode so we can give the labels headroom afterwards.
    _abs_ymax = {id(a): 0.0 for a in _axes}

    for _ax_i, (_visible, _label, _df, _color, _alpha) in zip(_axes, _active):
        _mz = _df['mz'].to_numpy()
        _raw = _df['intensity'].to_numpy(dtype=float)
        _norm = _raw / _raw.max() if _raw.max() else _raw
        # Plotted signal depends on mode; peak detection always uses the
        # max-normalized signal so the 0.09 threshold is interpretable in both.
        _y_plot = smooth(_norm if _relative else _raw, window.value)
        _y_pick = smooth(_norm, window.value)
        _ax_i.plot(_mz, _y_plot, color=_color, alpha=_alpha, label=_label)
        if _stacked:
            _ax_i.set_title(_label, loc=header_loc.value)

        _step = float(np.median(np.diff(_mz)))
        _distance = max(1, int(round(_min_sep_mz / _step)))
        _peaks, _ = find_peaks(_y_pick, height=0.09, distance=_distance)

        for _i in _peaks:
            if _relative:
                _text = f'{_mz[_i]:.1f} ({round(100 * _y_pick[_i])}%)'
            else:
                _text = f'{_mz[_i]:.1f} ({_y_plot[_i]:.1e})'
            _ax_i.annotate(
                _text,
                xy=(_mz[_i], _y_plot[_i]),
                xytext=(0, 3),
                textcoords='offset points',
                rotation=90, rotation_mode='anchor',
                ha='left', va='center', multialignment='center',
                fontsize=10, color='black',
                annotation_clip=False,
            )

        _abs_ymax[id(_ax_i)] = max(_abs_ymax[id(_ax_i)], float(_y_plot.max()))

    # Apply identical cosmetics to every axes used (deduplicated).
    _axes_seen = []
    for a in _axes:
        if not any(a is s for s in _axes_seen):
            _axes_seen.append(a)

    for _a in _axes_seen:
        if _relative:
            _a.set_ylabel('Normalized intensity')
            _a.set_ylim(0, 1.0)
            _a.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        else:
            _a.set_ylabel('Intensity')
            _a.set_ylim(0, _abs_ymax[id(_a)] * 1.02)
            _sf = ScalarFormatter(useMathText=True)
            _sf.set_scientific(True)
            _sf.set_powerlimits((0, 0))  # always use scientific notation
            _a.yaxis.set_major_formatter(_sf)
        _a.spines['top'].set_visible(False)
        _a.spines['right'].set_visible(False)
        if not _stacked:
            # Anchor the legend just above the axes (y > 1) so it lives in the
            # extra space we reserved with subplots_adjust(top=...).
            _anchors = {
                'left':   ((0.0, 1.02), 'lower left'),
                'center': ((0.5, 1.02), 'lower center'),
                'right':  ((1.0, 1.02), 'lower right'),
            }
            _bbox, _legloc = _anchors[header_loc.value]
            _a.legend(frameon=False, loc=_legloc,
                      bbox_to_anchor=_bbox, bbox_transform=_a.transAxes)
    _axes_seen[-1].set_xlabel('m/z')

    # Reserve extra space above each axes for the vertical annotations.
    if _stacked:
        _fig.subplots_adjust(top=0.88, hspace=0.55)
    else:
        _fig.subplots_adjust(top=0.72)

    _mode_tag = 'relative' if _relative else 'absolute'
    _layout_tag = 'stacked' if _stacked else 'overlapped'
    _fig.savefig(out_dir / f'Full spectrum_{picker.value}_{_layout_tag}_{_mode_tag}.svg', transparent=True, bbox_inches='tight')
    _fig.savefig(out_dir / f'Full spectrum_{picker.value}_{_layout_tag}_{_mode_tag}.png', transparent=True, bbox_inches='tight', dpi=300)
    mo.mpl.interactive(_fig)
    return


@app.cell(hide_code=True)
def _(mo):

    mass_list_text = mo.ui.text_area(
        value="231t ([1+1]),327t ([1+2]), 425t ([2+2] fragment), 559h ([3+2]), 637h ([3+3])",
        label="Masses to zoom (m/z; suffix required: h=high, t=typical, l=low; optional title in parentheses)",
        full_width=False,
    )
    mass_layout = mo.ui.radio(
        ["Stacked", "Unstacked"],
        value="Stacked",
        label="Dataset layout",
        inline=True,
    )
    mass_intensity_mode = mo.ui.radio(
        ["Relative", "Absolute"],
        value="Relative",
        label="Intensity",
        inline=True,
    )
    mass_show_legend = mo.ui.checkbox(value=True, label="Show legend")

    mo.vstack([
        mo.md("### Mass-window zooms"),
        mo.hstack([mass_list_text, mass_layout, mass_intensity_mode, mass_show_legend], justify="start"),
    ])

    return mass_intensity_mode, mass_layout, mass_list_text, mass_show_legend


@app.cell(hide_code=True)
def _(
    data_root,
    mass_intensity_mode,
    mass_layout,
    mass_list_text,
    mass_show_legend,
    mo,
    plt,
    read_spec,
    smooth,
    window,
):

    import re

    _mass_tokens = mass_list_text.value.replace(";", ",").replace("\n", ",").split(",")
    _voltage_suffixes = {
        "h": ("high_frag", "High source voltage", "40–60 V"),
        "t": ("standard", "Typical source voltage", "20–50 V"),
        "l": ("low_frag", "Low source voltage", "20 V"),
    }
    _zoom_specs = []
    _bad_mass_tokens = []
    for _token in _mass_tokens:
        _token = _token.strip()
        if not _token:
            continue

        _match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([htlHTL])(?:\s*\((.*?)\))?", _token)
        if not _match:
            _bad_mass_tokens.append(_token)
            continue

        _mass_part, _suffix, _custom_title = _match.groups()
        _suffix = _suffix.lower()
        _voltage, _voltage_label, _voltage_range = _voltage_suffixes[_suffix]
        _mass = float(_mass_part)
        _custom_title = _custom_title.strip() if _custom_title else ""
        _title_base = _custom_title if _custom_title else f"m/z {_mass:g}{_suffix}"

        _zoom_specs.append({
            "mass": _mass,
            "voltage": _voltage,
            "voltage_label": _voltage_label,
            "voltage_range": _voltage_range,
            "suffix": _suffix,
            "label": f"{_mass:g}{_suffix}",
            "title": f"{_title_base} ({_voltage_range})",
        })

    if _bad_mass_tokens:
        raise ValueError(
            "Mass entries must look like 327h, 327t, 327l, or 327h (custom title). "
            "No voltage suffix is assumed. Invalid entries: " + ", ".join(_bad_mass_tokens)
        )

    _seen_specs = set()
    _zoom_specs = [
        _spec for _spec in _zoom_specs
        if not ((_spec["mass"], _spec["voltage"]) in _seen_specs or _seen_specs.add((_spec["mass"], _spec["voltage"])))
    ]

    _zoom_relative = mass_intensity_mode.value == "Relative"

    if not _zoom_specs:
        _mass_plot = mo.md("Enter at least one mass with voltage suffix, e.g. `327h` or `327h (label)`.")
    else:
        _spectra_cache = {}
        def _zoom_spectra(_voltage):
            if _voltage not in _spectra_cache:
                _macrocycle = read_spec(data_root / f"full_spectrum_macrocycle_{_voltage}.csv")
                _reaction = read_spec(data_root / f"full_spectrum_reaction_{_voltage}.csv")
                for _df in (_macrocycle, _reaction):
                    _df.loc[_df["mz"].between(620, 625), "intensity"] = 0.0
                _spectra_cache[_voltage] = [
                    ("Pre-formed macrocycle", _macrocycle, "C0", 1.0),
                    ("Aerosol reaction mixture", _reaction, "C3", 0.85),
                ]
            return _spectra_cache[_voltage]

        _unstacked = mass_layout.value == "Unstacked"
        _ncols = len(_zoom_specs)
        _nrows = 2 if _unstacked else 1
        _fig, _axes = plt.subplots(
            nrows=_nrows,
            ncols=_ncols,
            figsize=(max(4.0 * _ncols, 6.0), 3.2 * _nrows),
            sharex="col" if _unstacked else False,
            sharey=True if _zoom_relative else False,
            squeeze=False,
        )

        for _col, _spec in enumerate(_zoom_specs):
            _mass = _spec["mass"]
            _lo, _hi = _mass - 1, _mass + 4
            _zoom_traces = _zoom_spectra(_spec["voltage"])
            for _row, (_sample, _df, _color, _alpha) in enumerate(_zoom_traces):
                _ax = _axes[_row, _col] if _unstacked else _axes[0, _col]
                _mz = _df["mz"].to_numpy()
                _raw = _df["intensity"].to_numpy(dtype=float)
                _y = smooth(_raw, window.value)
                _mask = (_mz >= _lo) & (_mz <= _hi)
                _x_region = _mz[_mask]
                _y_region = _y[_mask]
                if _zoom_relative and _y_region.size and _y_region.max():
                    # Normalize within this m−1 to m+4 window so each curve's
                    # tallest local peak reaches 1 in every panel.
                    _y_region = _y_region / _y_region.max()
                _ax.plot(_x_region, _y_region, color=_color, alpha=_alpha, label=_sample)

                if _unstacked:
                    if _col == 0:
                        _ax.set_ylabel("Norm. intensity" if _zoom_relative else "Intensity")
                    if _row == 0:
                        _ax.set_title(_spec["title"])
                else:
                    _ax.set_title(_spec["title"])
                    if _col == 0:
                        _ax.set_ylabel("Norm. intensity" if _zoom_relative else "Intensity")

                if _zoom_relative:
                    _ax.set_ylim(0, 1.05)
                _ax.spines["top"].set_visible(False)
                _ax.spines["right"].set_visible(False)

            _axes[-1, _col].set_xlabel("m/z")
            if mass_show_legend.value and (_col == _ncols - 1 or _ncols == 1):
                if _unstacked:
                    for _legend_ax in _axes[:, _col]:
                        _legend_ax.legend(frameon=False, fontsize=9)
                else:
                    _axes[0, _col].legend(frameon=False, fontsize=9)

        _fig.tight_layout()
        _mass_plot = mo.mpl.interactive(_fig)

    panels_fig = _fig
    _mass_plot
    return


if __name__ == "__main__":
    app.run()
