# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "anywidget==0.11.0",
#     "jax==0.11.0",
#     "marimo>=0.23.14",
#     "matplotlib==3.11.1",
#     "numpy==2.5.1",
#     "numpyro==0.21.0",
#     "scipy==1.18.0",
#     "seaborn==0.13.2",
#     "traitlets==5.15.1",
# ]
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import base64
    import importlib
    import inspect
    import io
    import json
    import math
    import pickle
    import shutil
    import urllib.request
    from gzip import GzipFile
    from pathlib import Path

    import anywidget
    import jax
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import traitlets
    from numpyro.infer import Predictive, SVI, Trace_ELBO, autoguide
    from numpyro.optim import Adam
    from scipy.signal import find_peaks

    import marimo as mo

    return (
        Adam,
        GzipFile,
        Path,
        Predictive,
        SVI,
        Trace_ELBO,
        anywidget,
        autoguide,
        base64,
        find_peaks,
        importlib,
        inspect,
        io,
        jax,
        json,
        math,
        mo,
        np,
        pickle,
        plt,
        shutil,
        sns,
        traitlets,
        urllib,
    )


@app.cell(hide_code=True)
def _(mo, plt, sns):
    plt.rcParams["figure.dpi"] = 200
    mo._runtime.context.get_context().marimo_config["runtime"]["output_max_bytes"] = 250_000_000
    sns.set_theme(context="talk", style="ticks", font="Arial",
                  rc={"svg.fonttype": "none", "savefig.format": "svg"})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Analysis of mass-spectrometry data
    """)
    return


@app.cell(hide_code=True)
def _(Path):
    MODEL_HOME = Path("./src/models").resolve()
    OUTPUT_HOME = Path("./out").resolve()
    return MODEL_HOME, OUTPUT_HOME


@app.cell(hide_code=True)
def _(Path, json, shutil, urllib):
    DATA_DOI = "https://doi.org/10.5281/zenodo.17609106"
    DATA_ARCHIVE = Path("./Final dataset.zip").resolve()
    DATA_HOME = Path("./Final dataset").resolve()

    if not DATA_HOME.exists():
        if not DATA_ARCHIVE.exists():
            with urllib.request.urlopen(DATA_DOI) as _doi_response:
                _dataset_record_id = _doi_response.url.rstrip("/").rsplit("/", 1)[-1]
            with urllib.request.urlopen(
                f"https://zenodo.org/api/records/{_dataset_record_id}"
            ) as _dataset_response:
                _dataset_record = json.load(_dataset_response)
            _dataset_url = next(
                _file["links"]["self"]
                for _file in _dataset_record["files"]
                if _file["key"] == "Final dataset.zip"
            )
            urllib.request.urlretrieve(_dataset_url, DATA_ARCHIVE)
        shutil.unpack_archive(DATA_ARCHIVE, DATA_HOME)
    return (DATA_HOME,)


@app.cell(hide_code=True)
def _(DATA_HOME, MODEL_HOME, importlib, mo):
    _experiment_paths = sorted(DATA_HOME.glob("*/*.pkgz"), key=lambda path: path.stem)
    all_experiments = {path.stem: path for path in _experiment_paths}

    _model_paths = sorted(MODEL_HOME.glob("model*.py"), key=lambda path: path.stem)
    models = {
        path.stem: importlib.import_module(f"src.models.{path.stem}")
        for path in _model_paths
    }

    mo.md(
        f"Found **{len(all_experiments):,} experiments** and "
        f"**{len(models):,} models**."
    )
    return all_experiments, models


@app.cell(hide_code=True)
def load_experiment(GzipFile, Path, pickle):
    def load_experiment(path: Path) -> dict:
        """Load and cache one gzipped mass-spectrometry experiment."""
        with GzipFile(path, "rb") as stream:
            return pickle.load(stream)

    return (load_experiment,)


@app.cell(hide_code=True)
def bin_masses(math, np):
    def bin_masses(data: dict, width: float = 1.0) -> dict:
        """Aggregate intensities into mass bins CENTRED on nominal masses.

        The previous version binned over [m-1, m) and labelled each bin by its
        *upper* edge. That did two bad things: it reported every ion 1 Da too
        high (the m/z 149 / 279 phthalate background came out as 150 / 280), and
        the bin edges cut through the peaks themselves -- on ZL-09G the m/z 149
        ion had 45% of its intensity in one bin and 55% in the next, so a single
        species appeared as two adjacent channels.

        Centring the bins on [m - w/2, m + w/2) puts each ion in one channel:
        capture for the strong ZL-09G ions goes 55% -> 100% (m/z 149),
        52% -> 100% (158), 82% -> 96% (79), 80% -> 94% (98).
        """
        masses = np.asarray(data["masses"])
        intensities = np.asarray(data["intensities"])
        centres = np.arange(
            math.floor(masses.min() + 0.5), math.ceil(masses.max() - 0.5) + 1, width
        )
        edges = np.append(centres - width / 2, centres[-1] + width / 2)
        index = np.searchsorted(edges, masses, side="right") - 1
        inside = (index >= 0) & (index < len(centres))
        index, columns = index[inside], intensities[:, inside]
        counts = np.bincount(index, minlength=len(centres))
        starts = np.clip(
            np.concatenate([[0], np.cumsum(counts)[:-1]]), 0, columns.shape[1] - 1
        )
        binned = np.add.reduceat(columns, starts, axis=1)
        binned[:, counts == 0] = 0
        return {**data, "masses": centres, "intensities": binned}

    return (bin_masses,)


@app.cell(hide_code=True)
def smooth_trace(np):
    def smooth_trace(values, window: int):
        """Centered N-point moving average with edge padding."""
        trace = np.asarray(values, dtype=float)
        window = min(int(window), trace.size)
        if window % 2 == 0:
            window -= 1
        if window <= 1 or trace.size < 2:
            return trace
        pad = window // 2
        padded = np.pad(trace, pad, mode="edge")
        return np.convolve(padded, np.ones(window) / window, mode="valid")


    return (smooth_trace,)


@app.cell(hide_code=True)
def peak_base_bounds(np, smooth_trace):
    def peak_base_bounds(
        trace,
        peaks,
        detect_window: int = 15,
        rise_tolerance: float = 0.10,
        return_fraction: float = 0.05,
        max_base_level: float = 0.5,
        width_limit: float = 4.0,
    ):
        """Locate integration bases either side of every peak.

        Returns ``(start, stop, left_level, right_level)`` per peak, where the
        levels are the denoised signal at each base — anchoring the baseline to
        single raw samples lets noise spikes tilt the tie line above the trace
        and yield negative areas.

        Flanks are followed on a smoothed copy, since scan-to-scan noise in a
        raw XIC stops a naive walk after one or two points. Each apex is first
        snapped to the corresponding maximum of that copy. A flank ends when the
        trace turns back up by more than ``rise_tolerance`` of the peak height
        *and* the valley has fallen below ``max_base_level`` (so shoulders of a
        doublet are not mistaken for bases), or once it returns to within
        ``return_fraction`` of the baseline. Arms are capped at ``width_limit``
        times the median half-width of all peaks, which keeps peaks riding on a
        broad hump from running away.
        """
        trace = np.asarray(trace, dtype=float)
        peaks = np.atleast_1d(peaks).astype(int)
        detect = smooth_trace(trace, detect_window)
        floor = float(np.percentile(detect, 5))

        apexes = []
        half_widths = []
        for peak in peaks:
            apex = peak
            for step in (-1, 1):
                index = peak
                while 0 <= index + step < detect.size and detect[index + step] > detect[index]:
                    index += step
                if detect[index] > detect[apex]:
                    apex = index
            apexes.append(apex)
            half_level = floor + 0.5 * (detect[apex] - floor)
            arms = []
            for step in (-1, 1):
                index = apex
                while 0 <= index + step < detect.size and detect[index + step] > half_level:
                    index += step
                arms.append(abs(index - apex) + 1)
            half_widths.append(max(arms))
        reach = max(3, int(width_limit * float(np.median(half_widths)))) if half_widths else 3

        bounds = []
        for peak, apex in zip(peaks, apexes):
            height = detect[apex] - floor
            stop_level = floor + return_fraction * height
            base_level = floor + max_base_level * height
            slack = rise_tolerance * height
            edges = []
            for step in (-1, 1):
                valley = index = apex
                while 0 <= index + step < detect.size and abs(index + step - apex) <= reach:
                    index += step
                    if detect[index] <= detect[valley]:
                        valley = index
                    if detect[index] <= stop_level:
                        valley = index
                        break
                    if detect[index] > detect[valley] + slack and detect[valley] <= base_level:
                        break
                edges.append(valley)
            start = min(edges[0], peak)
            stop = max(edges[1], peak) + 1
            bounds.append((start, stop, float(detect[start]), float(detect[stop - 1])))
        return bounds


    return (peak_base_bounds,)


@app.cell(hide_code=True)
def annotate_chromatogram_peaks(find_peaks, math, np, peak_base_bounds):
    def annotate_chromatogram_peaks(
        ax,
        times,
        values,
        *,
        plot_times=None,
        scale: float = 1.0,
        offset: float = 0.0,
        color=None,
        label_fontsize: float = 8.75,
        peak_threshold: float = 0.5,
        peak_spacing: float = 100.0,
        peak_bases: bool = False,
        integration_scans: int = 20,
        detect_window: int = 15,
        rise_tolerance: float = 0.10,
        return_fraction: float = 0.05,
        width_limit: float = 4.0,
    ):
        """Mark, shade and label the integrated peaks of one chromatogram.

        Decorates an existing trace rather than drawing it, so the caller keeps
        control of the line itself. ``scale`` and ``offset`` map signal units
        onto the axes, which lets stacked XIC lanes share this code with a TIC
        plotted in raw units. Returns the peak indices and their areas.
        """
        times = np.asarray(times, dtype=float)
        values = np.asarray(values, dtype=float)
        if plot_times is None:
            plot_times = times
        deltas = np.diff(times)
        positive = deltas[deltas > 0]
        median_step = float(np.median(positive)) if positive.size else 1.0
        detection = values / (float(values.max()) or 1.0)
        drawn = values / scale + offset
        peaks = find_peaks(
            detection,
            height=peak_threshold,
            distance=max(1, math.ceil(peak_spacing / median_step)),
        )[0]
        if peak_bases and peaks.size:
            bounds = peak_base_bounds(
                values,
                peaks,
                detect_window=detect_window,
                rise_tolerance=rise_tolerance,
                return_fraction=return_fraction,
                width_limit=width_limit,
            )
        else:
            bounds = [
                (
                    max(0, int(peak) - 1),
                    min(int(peak) + integration_scans, values.size),
                    0.0,
                    0.0,
                )
                for peak in peaks
            ]

        areas = []
        for peak, (start, stop, left_level, right_level) in zip(peaks, bounds):
            window = slice(start, stop)
            window_times = times[window]
            window_values = values[window]
            if peak_bases and window_values.size > 1:
                base = np.interp(
                    window_times,
                    window_times[[0, -1]],
                    [left_level, right_level],
                )
            else:
                base = np.zeros_like(window_values)
            area = (
                float(
                    np.trapezoid(
                        np.maximum(window_values - base, 0.0), window_times
                    )
                )
                if window_values.size > 1
                else float(window_values[0] * median_step)
            )
            areas.append(area)

            base_curve = base / scale + offset
            ax.fill_between(
                plot_times[window],
                base_curve,
                np.maximum(drawn[window], base_curve),
                color=color,
                alpha=0.24,
                zorder=1,
            )
            if peak_bases and window_values.size > 1:
                ax.plot(
                    plot_times[window],
                    base_curve,
                    color=color,
                    linewidth=1.0,
                    linestyle="--",
                    zorder=2,
                )
            ax.scatter(
                plot_times[peak],
                drawn[peak],
                color=color,
                edgecolor="black",
                linewidth=0.5,
                s=24,
                zorder=3,
            )
            ax.annotate(
                f"A={area:.2e}",
                xy=(plot_times[peak], drawn[peak]),
                xytext=(6, -7),
                textcoords="offset points",
                fontsize=label_fontsize,
                color=color,
                ha="left",
                va="top",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.7,
                },
                zorder=4,
            )
        return peaks, areas


    return (annotate_chromatogram_peaks,)


@app.cell(hide_code=True)
def MatplotlibRangeBrush(anywidget, traitlets):
    class MatplotlibRangeBrush(anywidget.AnyWidget):
        image = traitlets.Unicode().tag(sync=True)
        width = traitlets.Int().tag(sync=True)
        height = traitlets.Int().tag(sync=True)
        axes_bounds = traitlets.List(traitlets.Float()).tag(sync=True)
        times = traitlets.List(traitlets.Float()).tag(sync=True)
        selection_start = traitlets.Int(0).tag(sync=True)
        selection_stop = traitlets.Int(1).tag(sync=True)

        _esm = r"""
        const SVG_NS = "http://www.w3.org/2000/svg";

        function svgNode(name, attrs = {}) {
          const node = document.createElementNS(SVG_NS, name);
          for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
          return node;
        }

        function render({ model, el }) {
          const controller = new AbortController();
          const { signal } = controller;
          el.classList.add("mpl-range-brush");

          const instruction = document.createElement("div");
          instruction.className = "mpl-range-instruction";
          instruction.textContent = "Click for one scan · drag horizontally for a scan range";
          const stage = document.createElement("div");
          stage.className = "mpl-range-stage";
          const image = document.createElement("img");
          image.alt = "Total-ion chromatogram";
          image.draggable = false;
          const svg = svgNode("svg");
          const selection = svgNode("rect", { class: "mpl-range-selection" });
          const leftHandle = svgNode("line", { class: "mpl-range-handle" });
          const rightHandle = svgNode("line", { class: "mpl-range-handle" });
          const overlay = svgNode("rect", { class: "mpl-range-overlay" });
          svg.append(selection, leftHandle, rightHandle, overlay);
          stage.append(image, svg);
          el.append(instruction, stage);

          let dragging = false;
          let anchor = 0;
          let preview = null;

          function dimensions() {
            const width = model.get("width");
            const height = model.get("height");
            const [left, top, right, bottom] = model.get("axes_bounds");
            return { width, height, left, top, right, bottom };
          }

          function nearestPosition(event) {
            const times = model.get("times");
            const { width, left, right } = dimensions();
            const rect = svg.getBoundingClientRect();
            const px = ((event.clientX - rect.left) / rect.width) * width;
            const ratio = Math.max(0, Math.min(1, (px - left) / (right - left)));
            const target = times[0] + ratio * (times[times.length - 1] - times[0]);
            let low = 0;
            let high = times.length - 1;
            while (low < high) {
              const mid = Math.floor((low + high) / 2);
              if (times[mid] < target) low = mid + 1;
              else high = mid;
            }
            if (low > 0 && Math.abs(times[low - 1] - target) < Math.abs(times[low] - target)) {
              return low - 1;
            }
            return low;
          }

          function xForPosition(position) {
            const times = model.get("times");
            const { left, right } = dimensions();
            const span = times[times.length - 1] - times[0] || 1;
            return left + ((times[position] - times[0]) / span) * (right - left);
          }

          function draw() {
            const times = model.get("times");
            const { width, height, left, top, right, bottom } = dimensions();
            image.src = model.get("image");
            svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
            overlay.setAttribute("x", left);
            overlay.setAttribute("y", top);
            overlay.setAttribute("width", right - left);
            overlay.setAttribute("height", bottom - top);

            let start = model.get("selection_start");
            let stop = model.get("selection_stop");
            if (preview) [start, stop] = preview;
            start = Math.max(0, Math.min(start, times.length - 1));
            stop = Math.max(start + 1, Math.min(stop, times.length));
            const last = stop - 1;
            const leftX = xForPosition(start);
            const rightX = xForPosition(last);
            const halfStep = Math.max((right - left) / Math.max(times.length - 1, 1) / 2, 1.5);
            const brushLeft = Math.max(left, leftX - halfStep);
            const brushRight = Math.min(right, rightX + halfStep);
            selection.setAttribute("x", brushLeft);
            selection.setAttribute("y", top);
            selection.setAttribute("width", Math.max(3, brushRight - brushLeft));
            selection.setAttribute("height", bottom - top);
            for (const [handle, x] of [[leftHandle, brushLeft], [rightHandle, brushRight]]) {
              handle.setAttribute("x1", x);
              handle.setAttribute("x2", x);
              handle.setAttribute("y1", top);
              handle.setAttribute("y2", bottom);
            }
            const count = stop - start;
          }

          svg.addEventListener("pointerdown", (event) => {
            event.preventDefault();
            dragging = true;
            anchor = nearestPosition(event);
            preview = [anchor, anchor + 1];
            svg.setPointerCapture(event.pointerId);
            draw();
          }, { signal });
          svg.addEventListener("pointermove", (event) => {
            if (!dragging) return;
            const current = nearestPosition(event);
            const first = Math.min(anchor, current);
            const last = Math.max(anchor, current);
            preview = [first, last + 1];
            draw();
          }, { signal });
          svg.addEventListener("pointerup", (event) => {
            if (!dragging) return;
            dragging = false;
            const current = nearestPosition(event);
            const first = Math.min(anchor, current);
            const last = Math.max(anchor, current);
            preview = null;
            model.set("selection_start", first);
            model.set("selection_stop", last + 1);
            model.save_changes();
            draw();
          }, { signal });
          svg.addEventListener("pointercancel", () => {
            dragging = false;
            preview = null;
            draw();
          }, { signal });

          model.on("change:image", draw);
          model.on("change:selection_start", draw);
          model.on("change:selection_stop", draw);
          draw();
          return () => controller.abort();
        }

        export default { render };
        """

        _css = r"""
        .mpl-range-brush { width: 100%; max-width: 100%; }
        .mpl-range-stage { position: relative; width: 100%; line-height: 0; }
        .mpl-range-stage img { display: block; width: 100%; height: auto; user-select: none; }
        .mpl-range-stage svg { position: absolute; inset: 0; width: 100%; height: 100%; }
        .mpl-range-overlay { fill: transparent; cursor: crosshair; touch-action: none; }
        .mpl-range-selection { fill: #f28e2b; fill-opacity: 0.22; pointer-events: none; }
        .mpl-range-handle { stroke: #e07000; stroke-width: 2; vector-effect: non-scaling-stroke; pointer-events: none; }
        .mpl-range-instruction { margin: 0 0 4px; font-size: 0.84rem; opacity: 0.72; }
        .mpl-range-status { margin-top: 4px; font: 600 0.86rem/1.35 ui-monospace, SFMono-Regular, monospace; }
        @media (prefers-color-scheme: dark) {
          .mpl-range-selection { fill: #ffad55; }
          .mpl-range-handle { stroke: #ffad55; }
        }
        """

    return (MatplotlibRangeBrush,)


@app.cell(hide_code=True)
def _(mo):
    spectrum_mass_range = mo.ui.range_slider(
        0, 100, step=0.5, value=[0, 100], show_value=True,
        label="Mass range (%)", full_width=True,
    )
    spectrum_peak_threshold = mo.ui.slider(
        0, 100, step=1, value=10, show_value=True,
        label="Peak threshold (%)", full_width=True,
    )
    spectrum_min_peak_distance = mo.ui.number(
        0.1, 100, step=0.1, value=1,
        label="Minimum peak distance (m/z)", full_width=True,
    )
    spectrum_prominence = mo.ui.number(
        0, 100, step=0.1, value=0, label="Peak prominence", full_width=True,
    )
    chromatogram_time_range = mo.ui.range_slider(
        0, 100, step=0.5, value=[0, 100], show_value=True,
        label="Time range (%)", full_width=True,
    )
    chromatogram_smoothing = mo.ui.number(
        1, 101, step=2, value=1, label="Smoothing (pts)", full_width=True,
    )
    spectrum_xic_mzs = mo.ui.text(
        placeholder="e.g. 115.4 (Amine), 149.1 (Aldehyde hemiacetal), 231.1 ([1 + 1])",
        label="Extracted-ion m/z values",
        full_width=True,
    )
    spectrum_xic_delta = mo.ui.number(
        0.01, 10, step=0.01, value=0.5,
        label="Extracted-ion tolerance", full_width=True,
    )
    chromatogram_integration_scans = mo.ui.number(
        1, 1000, step=1, value=20,
        label="Scans integrated", full_width=True,
    )
    chromatogram_peak_bases = mo.ui.switch(value=False, label="Peak bases")
    chromatogram_peak_threshold = mo.ui.slider(
        1, 100, step=1, value=50, show_value=True,
        label="Peak threshold (%)", full_width=True,
    )
    chromatogram_peak_spacing = mo.ui.number(
        1, 1000, step=1, value=100,
        label="Minimum peak spacing (s)", full_width=True,
    )
    chromatogram_base_smoothing = mo.ui.number(
        3, 101, step=2, value=15,
        label="Base detection smoothing (pts)", full_width=True,
    )
    chromatogram_base_rise = mo.ui.slider(
        0, 50, step=1, value=10, show_value=True,
        label="Base rise tolerance (%)", full_width=True,
    )
    chromatogram_base_return = mo.ui.slider(
        0, 100, step=1, value=5, show_value=True,
        label="Base return to baseline (%)", full_width=True,
    )
    chromatogram_base_width = mo.ui.number(
        1.0, 20.0, step=0.5, value=4.0,
        label="Max width (x half-width)", full_width=True,
    )
    spectrum_show_peak_intensities = mo.ui.checkbox(
        value=False, label="Label peak intensities"
    )
    spectrum_rotate_labels = mo.ui.checkbox(value=False, label="Rotate peak labels")
    spectrum_series = mo.ui.checkbox(value=False, label="Plot scans individually")
    spectrum_average = mo.ui.checkbox(
        value=False, label="Average selected scans (unchecked: sum)"
    )
    spectrum_truncate_start_time = mo.ui.checkbox(
        value=False, label="Start selected time range at zero"
    )
    return (
        chromatogram_base_return,
        chromatogram_base_rise,
        chromatogram_base_smoothing,
        chromatogram_base_width,
        chromatogram_integration_scans,
        chromatogram_peak_bases,
        chromatogram_peak_spacing,
        chromatogram_peak_threshold,
        chromatogram_smoothing,
        chromatogram_time_range,
        spectrum_average,
        spectrum_mass_range,
        spectrum_min_peak_distance,
        spectrum_peak_threshold,
        spectrum_prominence,
        spectrum_rotate_labels,
        spectrum_series,
        spectrum_show_peak_intensities,
        spectrum_truncate_start_time,
        spectrum_xic_delta,
        spectrum_xic_mzs,
    )


@app.cell(hide_code=True)
def chromatogram_peak_settings(
    chromatogram_base_return,
    chromatogram_base_rise,
    chromatogram_base_smoothing,
    chromatogram_base_width,
    chromatogram_integration_scans,
    chromatogram_peak_bases,
    chromatogram_peak_spacing,
    chromatogram_peak_threshold,
    chromatogram_smoothing,
):
    chromatogram_peak_settings = {
        "peak_threshold": chromatogram_peak_threshold.value / 100,
        "peak_spacing": float(chromatogram_peak_spacing.value),
        "peak_bases": chromatogram_peak_bases.value,
        "integration_scans": int(chromatogram_integration_scans.value),
        "detect_window": max(
            int(chromatogram_base_smoothing.value),
            int(chromatogram_smoothing.value),
        ),
        "rise_tolerance": chromatogram_base_rise.value / 100,
        "return_fraction": chromatogram_base_return.value / 100,
        "width_limit": float(chromatogram_base_width.value),
    }
    return (chromatogram_peak_settings,)


@app.cell(hide_code=True)
def _(
    MatplotlibRangeBrush,
    all_experiments,
    annotate_chromatogram_peaks,
    base64,
    chromatogram_peak_settings,
    chromatogram_smoothing,
    chromatogram_time_range,
    io,
    load_experiment,
    mo,
    np,
    plt,
    sample_picker,
    smooth_trace,
    spectrum_truncate_start_time,
):
    _selector_sample = sample_picker.value
    _selector_data = load_experiment(all_experiments[_selector_sample])
    _selector_times = np.asarray(_selector_data["times"])
    _selector_intensities = np.asarray(_selector_data["intensities"])
    _selector_plot_times = (
        _selector_times - _selector_times[0]
        if spectrum_truncate_start_time.value
        else _selector_times
    )

    _selector_full_tic = smooth_trace(
        _selector_intensities.sum(axis=1), chromatogram_smoothing.value
    )
    _selector_start_pct, _selector_end_pct = chromatogram_time_range.value
    _selector_span = float(_selector_plot_times[-1] - _selector_plot_times[0])
    _selector_start_time = float(_selector_plot_times[0]) + (
        _selector_span * _selector_start_pct / 100
    )
    _selector_end_time = float(_selector_plot_times[0]) + (
        _selector_span * _selector_end_pct / 100
    )
    _selector_mask = (_selector_plot_times >= _selector_start_time) & (
        _selector_plot_times <= _selector_end_time
    )
    if int(_selector_mask.sum()) < 2:
        _selector_mask = np.ones_like(_selector_plot_times, dtype=bool)
    _selector_indices = np.flatnonzero(_selector_mask)
    tic_range_offset = int(_selector_indices[0])
    _selector_times = _selector_times[_selector_mask]
    _selector_plot_times = _selector_plot_times[_selector_mask]
    _selector_tic = _selector_full_tic[_selector_mask]

    _selector_fig, _selector_ax = plt.subplots(figsize=(12.3, 3.2))
    (_selector_line,) = _selector_ax.plot(_selector_plot_times, _selector_tic)
    annotate_chromatogram_peaks(
        _selector_ax,
        _selector_times,
        _selector_tic,
        plot_times=_selector_plot_times,
        color=_selector_line.get_color(),
        **chromatogram_peak_settings,
    )
    _selector_ax.set(ylabel="TIC", xlabel="Time (s)")
    _selector_fig.tight_layout()
    _selector_fig.canvas.draw()
    _selector_width, _selector_height = _selector_fig.canvas.get_width_height()
    _selector_bbox = _selector_ax.get_window_extent()
    _selector_axes_bounds = [
        float(_selector_bbox.x0),
        float(_selector_height - _selector_bbox.y1),
        float(_selector_bbox.x1),
        float(_selector_height - _selector_bbox.y0),
    ]
    _selector_buffer = io.BytesIO()
    _selector_fig.savefig(_selector_buffer, format="png")
    _selector_png = _selector_buffer.getvalue()
    _selector_svg_buffer = io.BytesIO()
    _selector_fig.savefig(_selector_svg_buffer, format="svg")
    _selector_image = "data:image/png;base64," + base64.b64encode(
        _selector_png
    ).decode("ascii")
    plt.close(_selector_fig)
    _selector_middle = len(_selector_times) // 2

    tic_range_selector = mo.ui.anywidget(
        MatplotlibRangeBrush(
            image=_selector_image,
            width=int(_selector_width),
            height=int(_selector_height),
            axes_bounds=_selector_axes_bounds,
            times=[float(value) for value in _selector_plot_times],
            selection_start=_selector_middle,
            selection_stop=_selector_middle + 1,
        )
    )

    tic_download = mo.download(
        data=_selector_svg_buffer.getvalue(),
        filename=f"{_selector_sample}_TIC.svg",
        mimetype="image/svg+xml",
        label="Save TIC (SVG)",
    )
    return tic_download, tic_range_offset, tic_range_selector


@app.cell(hide_code=True)
def _(all_experiments, mo):
    sample_picker = mo.ui.dropdown(
        list(all_experiments),
        value=next(iter(all_experiments)),
        searchable=True,
        label="Sample to analyse",
        full_width=True,
    )

    mo.vstack(
        [
            mo.md("## Interactive exploration"),
            sample_picker,
        ],
        gap=0.75,
    )
    return (sample_picker,)


@app.cell(hide_code=True)
def spectrum_result(
    OUTPUT_HOME,
    all_experiments,
    annotate_chromatogram_peaks,
    chromatogram_base_return,
    chromatogram_base_rise,
    chromatogram_base_smoothing,
    chromatogram_base_width,
    chromatogram_integration_scans,
    chromatogram_peak_bases,
    chromatogram_peak_settings,
    chromatogram_peak_spacing,
    chromatogram_peak_threshold,
    chromatogram_smoothing,
    chromatogram_time_range,
    find_peaks,
    load_experiment,
    math,
    mo,
    np,
    plt,
    sample_picker,
    smooth_trace,
    sns,
    spectrum_average,
    spectrum_mass_range,
    spectrum_min_peak_distance,
    spectrum_peak_threshold,
    spectrum_prominence,
    spectrum_rotate_labels,
    spectrum_series,
    spectrum_show_peak_intensities,
    spectrum_truncate_start_time,
    spectrum_xic_delta,
    spectrum_xic_mzs,
    tic_download,
    tic_range_offset,
    tic_range_selector,
):
    _spec_sample = sample_picker.value
    _spec_data = load_experiment(all_experiments[_spec_sample])
    _spec_times = np.asarray(_spec_data["times"])
    _spec_all_masses = np.asarray(_spec_data["masses"])
    _spec_intensities = np.asarray(_spec_data["intensities"])
    _spec_selector_state = tic_range_selector.value
    _spec_scan_start = max(
        0,
        min(
            tic_range_offset + int(_spec_selector_state["selection_start"]),
            len(_spec_times) - 1,
        ),
    )
    _spec_scan_stop = max(
        _spec_scan_start + 1,
        min(
            tic_range_offset + int(_spec_selector_state["selection_stop"]),
            len(_spec_times),
        ),
    )
    _spec_scan_last = _spec_scan_stop - 1

    _spec_controls = mo.vstack(
        [
            mo.md("### Spectrum explorer"),
            mo.hstack(
                [
                    spectrum_mass_range,
                    chromatogram_time_range,
                    spectrum_peak_threshold,
                    spectrum_min_peak_distance,
                    chromatogram_smoothing,
                ],
                widths=[2, 2, 2, 1, 1],
                align="end",
            ),
            mo.accordion(
                {
                    "Extracted-ion chromatograms": mo.vstack(
                        [
                            mo.hstack(
                                [spectrum_xic_mzs, spectrum_xic_delta],
                                widths=[4, 1],
                                align="end",
                            ),
                        ],
                        gap=0.5,
                    ),
                    "Chromatogram peaks and integration": mo.vstack(
                        [
                            mo.hstack(
                                [
                                    chromatogram_peak_threshold,
                                    chromatogram_peak_spacing,
                                    chromatogram_integration_scans,
                                    chromatogram_peak_bases,
                                ],
                                widths=[2, 1, 1, 1],
                                align="end",
                            ),
                            mo.hstack(
                                [
                                    chromatogram_base_return,
                                    chromatogram_base_rise,
                                    chromatogram_base_smoothing,
                                    chromatogram_base_width,
                                ],
                                widths=[2, 2, 1, 1],
                                align="end",
                            ),
                        ],
                        gap=0.5,
                    ),
                    "Advanced display and peak settings": mo.vstack(
                        [
                            spectrum_prominence,
                            mo.hstack(
                                [
                                    spectrum_show_peak_intensities,
                                    spectrum_rotate_labels,
                                    spectrum_series,
                                    spectrum_average,
                                    spectrum_truncate_start_time,
                                ],
                                justify="start",
                                wrap=True,
                            ),
                        ]
                    ),
                }
            ),
        ],
        gap=1,
    )

    _spec_mass_start_pct, _spec_mass_end_pct = spectrum_mass_range.value
    _spec_mass_start = _spec_all_masses[0] + (
        _spec_all_masses[-1] - _spec_all_masses[0]
    ) * _spec_mass_start_pct / 100
    _spec_mass_end = _spec_all_masses[0] + (
        _spec_all_masses[-1] - _spec_all_masses[0]
    ) * _spec_mass_end_pct / 100
    _spec_mass_mask = (_spec_all_masses >= _spec_mass_start) & (
        _spec_all_masses <= _spec_mass_end
    )
    _spec_masses = _spec_all_masses[_spec_mass_mask]
    mo.stop(
        len(_spec_masses) < 2,
        mo.vstack(
            [_spec_controls, mo.callout("Select a wider mass range.", kind="warn")]
        ),
    )

    _spec_scans = _spec_intensities[
        _spec_scan_start:_spec_scan_stop, _spec_mass_mask
    ]
    _spec_scan = (
        _spec_scans.mean(axis=0)
        if spectrum_average.value
        else _spec_scans.sum(axis=0)
    )
    _spec_tic = _spec_scans.sum()
    _spec_scan_max = float(_spec_scan.max()) if _spec_scan.size else 0.0
    _spec_threshold = spectrum_peak_threshold.value / 100 * _spec_scan_max
    _spec_mass_step = float(np.median(np.diff(_spec_masses)))
    _spec_distance_bins = max(
        1,
        math.ceil(float(spectrum_min_peak_distance.value) / _spec_mass_step),
    )
    _spec_peaks = find_peaks(
        _spec_scan,
        height=_spec_threshold,
        distance=_spec_distance_bins,
        prominence=float(spectrum_prominence.value),
    )[0]
    _spec_xic_entries = []
    for _spec_xic_token in spectrum_xic_mzs.value.split(","):
        _spec_xic_token = _spec_xic_token.strip()
        if not _spec_xic_token:
            continue
        _spec_xic_number, _, _spec_xic_description = _spec_xic_token.partition("(")
        _spec_xic_number = _spec_xic_number.strip()
        _spec_xic_description = _spec_xic_description.strip().rstrip(")").strip()
        try:
            _spec_xic_mz_value = float(_spec_xic_number)
        except ValueError:
            continue
        _spec_xic_entries.append(
            (
                _spec_xic_mz_value,
                f"m/z {_spec_xic_number}"
                + (f" ({_spec_xic_description})" if _spec_xic_description else ""),
            )
        )
    _spec_xic_values = [_spec_entry[0] for _spec_entry in _spec_xic_entries]

    _spec_xic_count = len(_spec_xic_values)
    _spec_xic_fig = None
    _spec_area_fig = None
    _spec_xic_peak_records = []
    if _spec_xic_values:
        _spec_xic_fig, _spec_xic_ax = plt.subplots(
            figsize=(12.3, 1.85 * _spec_xic_count)
        )
    _spec_scan_fig, _spec_scan_ax = plt.subplots(figsize=(12.3, 4.37))
    _spec_plot_times = (
        _spec_times - _spec_times[0]
        if spectrum_truncate_start_time.value
        else _spec_times
    )

    if _spec_xic_values:
        for _spec_offset, (_spec_mz, _spec_xic_label) in enumerate(
            _spec_xic_entries
        ):
            _spec_xic_mask = (
                (_spec_all_masses > _spec_mz - spectrum_xic_delta.value)
                & (_spec_all_masses < _spec_mz + spectrum_xic_delta.value)
            )
            _spec_xic = smooth_trace(
                _spec_intensities[:, _spec_xic_mask].sum(axis=1),
                chromatogram_smoothing.value,
            )
            _spec_xic_scale = float(_spec_xic.max()) or 1.0
            _spec_xic_normalized = _spec_xic / _spec_xic_scale
            _spec_xic_baseline = 1.2 * _spec_offset
            (_spec_xic_line,) = _spec_xic_ax.plot(
                _spec_plot_times,
                _spec_xic_normalized + _spec_xic_baseline,
                label=_spec_xic_label,
                zorder=2,
            )
            _spec_xic_color = _spec_xic_line.get_color()
            _spec_xic_ax.axhline(
                _spec_xic_baseline,
                color="black",
                linewidth=0.8,
                linestyle="-",
                alpha=0.7,
                zorder=0,
            )
            for _spec_xic_fraction in np.arange(0.1, 1.01, 0.1):
                _spec_xic_ax.axhline(
                    _spec_xic_baseline + _spec_xic_fraction,
                    color="0.45",
                    linewidth=0.55,
                    linestyle=":",
                    alpha=0.45,
                    zorder=0,
                )
            _spec_xic_peaks, _spec_xic_areas = annotate_chromatogram_peaks(
                _spec_xic_ax,
                _spec_times,
                _spec_xic,
                plot_times=_spec_plot_times,
                scale=_spec_xic_scale,
                offset=_spec_xic_baseline,
                color=_spec_xic_color,
                **chromatogram_peak_settings,
            )
            _spec_xic_peak_records.append(
                (
                    np.asarray(_spec_plot_times)[_spec_xic_peaks],
                    np.asarray(_spec_xic_areas, dtype=float),
                    _spec_xic_color,
                    _spec_xic_label,
                )
            )
        _spec_xic_tick_fractions = np.arange(0.2, 1.01, 0.2)
        _spec_xic_tick_positions = [
            1.2 * _spec_ion_index + _spec_tick_fraction
            for _spec_ion_index in range(_spec_xic_count)
            for _spec_tick_fraction in _spec_xic_tick_fractions
        ]
        _spec_xic_tick_labels = [
            f"{round(100 * _spec_tick_fraction):d}%"
            for _spec_ion_index in range(_spec_xic_count)
            for _spec_tick_fraction in _spec_xic_tick_fractions
        ]
        _spec_time_start_pct, _spec_time_end_pct = chromatogram_time_range.value
        _spec_time_span = float(_spec_plot_times[-1] - _spec_plot_times[0])
        _spec_time_start = float(_spec_plot_times[0]) + (
            _spec_time_span * _spec_time_start_pct / 100
        )
        _spec_time_end = float(_spec_plot_times[0]) + (
            _spec_time_span * _spec_time_end_pct / 100
        )
        if _spec_time_end <= _spec_time_start:
            _spec_time_end = _spec_time_start + max(_spec_time_span, 1.0) * 1e-3
        _spec_xic_ax.set(
            xlim=(_spec_time_start, _spec_time_end),
            xlabel="Time (s)",
            ylabel="Normalized XIC",
            yticks=_spec_xic_tick_positions,
            yticklabels=_spec_xic_tick_labels,
            ylim=(-0.05, 1.2 * (_spec_xic_count - 1) + 1.05),
        )
        _spec_xic_ax.tick_params(
            axis="y",
            which="major",
            left=False,
            right=True,
            labelleft=False,
            labelright=True,
            labelsize=8.25,
            length=4,
        )
        _spec_xic_ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            fontsize=12.5,
            ncol=max(1, math.ceil(_spec_xic_count / 3)),
            frameon=False,
            borderaxespad=0,
            borderpad=0,
            labelspacing=0.3,
            handlelength=1.2,
            handletextpad=0.4,
            columnspacing=1.0,
        )

    if _spec_xic_values:
        _spec_area_fig, _spec_area_ax = plt.subplots(figsize=(12.3, 6.0))
        for (
            _spec_area_times,
            _spec_area_values,
            _spec_area_color,
            _spec_area_label,
        ) in _spec_xic_peak_records:
            _spec_area_normalized = _spec_area_values / (
                float(_spec_area_values.max()) if _spec_area_values.size else 1.0
            )
            _spec_area_ax.plot(
                _spec_area_times,
                _spec_area_normalized,
                linestyle=":",
                linewidth=0.9,
                alpha=0.55,
                color=_spec_area_color,
                zorder=2,
            )
            _spec_area_ax.plot(
                _spec_area_times,
                _spec_area_normalized,
                linestyle="none",
                marker="o",
                markersize=4.5,
                color=_spec_area_color,
                label=_spec_area_label,
                zorder=3,
            )
        _spec_area_ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7, zorder=1)
        _spec_area_tick_fractions = np.arange(0.0, 1.01, 0.1)
        for _spec_area_tick_fraction in _spec_area_tick_fractions:
            _spec_area_ax.axhline(
                _spec_area_tick_fraction,
                color="0.45",
                linewidth=0.55,
                linestyle=":",
                alpha=0.45,
                zorder=0,
            )
        _spec_area_ax.set(
            xlim=(_spec_time_start, _spec_time_end),
            ylim=(-0.03, 1.03),
            xlabel="Time (s)",
            ylabel="Normalized peak area",
            yticks=_spec_area_tick_fractions,
            yticklabels=[
                f"{round(100 * _spec_area_tick_fraction):d}%"
                for _spec_area_tick_fraction in _spec_area_tick_fractions
            ],
        )
        _spec_area_ax.tick_params(
            axis="y",
            which="major",
            left=False,
            right=True,
            labelleft=False,
            labelright=True,
            labelsize=8.25,
            length=4,
        )
        _spec_area_ax.legend(
            loc="lower right",
            fontsize=12.5,
            ncol=max(1, math.ceil(_spec_xic_count / 3)),
            labelspacing=0.3,
            handlelength=1.2,
            handletextpad=0.4,
            columnspacing=1.0,
            framealpha=0.85,
        )

    if spectrum_series.value and len(_spec_scans) > 1:
        _spec_colors = sns.color_palette("crest", n_colors=len(_spec_scans))
        for _spec_color, _spec_single_scan in zip(_spec_colors, _spec_scans):
            _spec_scan_ax.plot(_spec_masses, _spec_single_scan, color=_spec_color)
    else:
        _spec_scan_ax.plot(_spec_masses, _spec_scan)
    _spec_scan_ax.scatter(
        _spec_masses[_spec_peaks],
        _spec_scan[_spec_peaks],
        color="tab:red",
        s=30,
        zorder=10,
        marker="x",
        linewidth=1,
    )
    _spec_label_offset = max(
        (_spec_scan_max - float(_spec_scan.min())) * 0.02,
        1e-12,
    )
    for _spec_peak in _spec_peaks:
        _spec_label = f"{_spec_masses[_spec_peak]:.2f}"
        if spectrum_show_peak_intensities.value:
            _spec_label += f"\n{_spec_scan[_spec_peak]:.2e}"
        _spec_scan_ax.text(
            _spec_masses[_spec_peak],
            _spec_scan[_spec_peak] + _spec_label_offset,
            _spec_label,
            color="tab:red",
            fontsize=10,
            horizontalalignment="center",
            rotation=90 if spectrum_rotate_labels.value else 0,
        )
    _spec_aggregation = "Average" if spectrum_average.value else "Sum"
    if len(_spec_scans) == 1:
        _spec_legend_entries = [
            f"Scan {_spec_scan_start}",
            f"t = {_spec_plot_times[_spec_scan_start]:.1f} s",
            f"TIC {_spec_tic:.2e}",
        ]
    else:
        _spec_legend_entries = [
            f"{_spec_aggregation} of {len(_spec_scans)} scans",
            f"Scans {_spec_scan_start}–{_spec_scan_last}",
            f"{_spec_plot_times[_spec_scan_start]:.1f}–"
            f"{_spec_plot_times[_spec_scan_last]:.1f} s",
            f"Integrated TIC {_spec_tic:.2e}",
        ]
    _spec_scan_ax.set(ylabel="Intensity", xlabel="m/z")
    _spec_scan_ax.legend(
        handles=[plt.Line2D([], [], linestyle="none")],
        labels=[" · ".join(_spec_legend_entries)],
        loc="upper right",
        fontsize=10,
        handlelength=0,
        handletextpad=0,
        borderpad=0.4,
        framealpha=0.85,
    )
    if _spec_scan_max > 0:
        _spec_scan_ax.set_ylim(
            0,
            _spec_scan_max * (1.3 if spectrum_rotate_labels.value else 1.15),
        )
    if _spec_xic_fig is not None:
        _spec_xic_fig.tight_layout()
    if _spec_area_fig is not None:
        _spec_area_fig.tight_layout()
    _spec_scan_fig.tight_layout()

    _spec_filename = (
        f"scan{_spec_scan_start}"
        if len(_spec_scans) == 1
        else f"scan{_spec_scan_start}-{_spec_scan_last}"
    )


    def _export_spectrum(_):
        _directory = OUTPUT_HOME / _spec_sample
        _directory.mkdir(parents=True, exist_ok=True)
        _path = _directory / f"{_spec_filename}.csv"
        np.savetxt(
            _path,
            np.column_stack([_spec_masses, _spec_scan]),
            delimiter=",",
            header="m/z,intensity",
            comments="",
        )


    _spec_export = mo.ui.button(label="Export CSV", on_click=_export_spectrum)
    mo.vstack(
        [
            _spec_controls,
            tic_range_selector,
            mo.hstack([tic_download], justify="end"),
            *([_spec_xic_fig] if _spec_xic_fig is not None else []),
            *([_spec_area_fig] if _spec_area_fig is not None else []),
            _spec_scan_fig,
            _spec_export,
        ],
        gap=0.5,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Probabilistic component model

    Choose a dataset and model, then click **Fit model** to run SVI.
    """)
    return


@app.cell(hide_code=True)
def _(mo, models):
    fit_model = mo.ui.dropdown(
        list(models),
        value="model13" if "model13" in models else next(iter(models)),
        searchable=True,
        label="Model",
        full_width=True,
    )
    fit_time_range = mo.ui.range_slider(
        0, 100, step=0.5, value=[0, 100], show_value=True,
        label="Time range (%)", full_width=True,
    )
    fit_mass_range = mo.ui.range_slider(
        0, 1000, step=1, value=[0, 1000], show_value=True,
        label="Mass range (m/z)", full_width=True,
    )
    fit_n_steps = mo.ui.number(
        100, 100000, step=100, value=10000, label="SVI steps", full_width=True,
    )
    fit_n_components = mo.ui.number(
        2, 10, step=1, value=5, label="Components", full_width=True,
    )
    fit_auto_guide = mo.ui.checkbox(value=True, label="Use AutoNormal guide")
    fit_run = mo.ui.run_button(label="Fit model", kind="success")
    fit_seed = mo.ui.number(
        0, 9999, step=1, value=0, label="Seed", full_width=True,
    )
    return (
        fit_auto_guide,
        fit_mass_range,
        fit_model,
        fit_n_components,
        fit_n_steps,
        fit_run,
        fit_seed,
        fit_time_range,
    )


@app.cell(hide_code=True)
def _(
    Adam,
    Predictive,
    SVI,
    Trace_ELBO,
    all_experiments,
    autoguide,
    bin_masses,
    fit_auto_guide,
    fit_mass_range,
    fit_model,
    fit_n_components,
    fit_n_steps,
    fit_run,
    fit_seed,
    fit_time_range,
    inspect,
    jax,
    load_experiment,
    mo,
    models,
    np,
    plt,
    sample_picker,
    sns,
):
    _fit_controls = mo.vstack(
        [
            fit_model,
            mo.hstack([fit_time_range, fit_mass_range], widths="equal", align="end"),
            mo.hstack(
                [fit_n_steps, fit_n_components, fit_seed, fit_auto_guide, fit_run],
                widths=[2, 2, 2, 2, 1],
                align="end",
            ),
        ],
        gap=1,
    )

    mo.stop(
        not fit_run.value,
        mo.vstack(
            [
                _fit_controls,
                mo.callout(
                    "Configure the model and click **Fit model** to run SVI.",
                    kind="info",
                ),
            ]
        ),
    )

    _fit_sample = sample_picker.value
    _fit_model_name = fit_model.value
    _fit_module = models[_fit_model_name]
    _fit_n_steps = int(fit_n_steps.value)
    _fit_n_components = int(fit_n_components.value)
    _fit_raw = bin_masses(load_experiment(all_experiments[_fit_sample]))
    _fit_all_times = np.asarray(_fit_raw["times"])
    _fit_all_masses = np.asarray(_fit_raw["masses"])
    _fit_all_intensities = np.asarray(_fit_raw["intensities"])
    _fit_start_pct, _fit_end_pct = fit_time_range.value
    _fit_time_start = _fit_all_times[0] + (
        _fit_all_times[-1] - _fit_all_times[0]
    ) * _fit_start_pct / 100
    _fit_time_end = _fit_all_times[0] + (
        _fit_all_times[-1] - _fit_all_times[0]
    ) * _fit_end_pct / 100
    _fit_time_mask = (_fit_all_times >= _fit_time_start) & (
        _fit_all_times <= _fit_time_end
    )
    _fit_mass_start, _fit_mass_end = fit_mass_range.value
    _fit_mass_mask = (_fit_all_masses >= _fit_mass_start) & (
        _fit_all_masses <= _fit_mass_end
    )
    _fit_times = _fit_all_times[_fit_time_mask]
    _fit_masses = _fit_all_masses[_fit_mass_mask]
    _fit_intensities = _fit_all_intensities[_fit_time_mask][:, _fit_mass_mask]
    mo.stop(
        _fit_intensities.size == 0,
        mo.vstack([_fit_controls, mo.callout("The selected time or mass range contains no observations.", kind="warn")]),
    )

    _fit_model_fn = _fit_module.ms_model
    _fit_guide_fn = getattr(_fit_module, "ms_model_guide", None)
    if fit_auto_guide.value or _fit_guide_fn is None:
        _fit_guide_fn = autoguide.AutoNormal(_fit_model_fn)

    # Only hand `times` to models whose signature asks for it (model16 onwards).
    _fit_args = [_fit_masses, _fit_intensities, _fit_n_components]
    if "times" in inspect.signature(_fit_model_fn).parameters:
        _fit_args.append(_fit_times)

    _fit_seed = int(fit_seed.value)
    _fit_svi = SVI(
        model=_fit_model_fn,
        guide=_fit_guide_fn,
        optim=Adam(0.002),
        loss=Trace_ELBO(),
    )
    _fit_svi_result = _fit_svi.run(
        jax.random.PRNGKey(_fit_seed),
        _fit_n_steps,
        *_fit_args,
        stable_update=True,
        progress_bar=True,
    )

    # Request only the sites that get plotted. The `deviation` deterministic in
    # models 13-15 is a full n_scans x n_masses array *per posterior sample*.
    _fit_trace = Predictive(
        _fit_model_fn,
        guide=_fit_guide_fn,
        params=_fit_svi_result.params,
        num_samples=25,
        return_sites=["component_intensities", "component_weights", "mz_intensities"],
    )(jax.random.PRNGKey(_fit_seed + 1), *_fit_args)

    # Components are exchangeable, so label switching makes two runs incomparable
    # for no reason. Order them by weighted mean retention time.
    _fit_weight_samples = np.asarray(_fit_trace["component_weights"])
    _fit_spectra_samples = np.asarray(_fit_trace["component_intensities"])
    _fit_mean_weights = _fit_weight_samples.mean(axis=0)
    _fit_centroids = (_fit_mean_weights * _fit_times[:, None]).sum(0) / np.maximum(
        _fit_mean_weights.sum(0), 1e-12
    )
    _fit_order = np.argsort(_fit_centroids)
    _fit_trace = {
        "component_weights": _fit_weight_samples[:, :, _fit_order],
        "component_intensities": _fit_spectra_samples[:, _fit_order, :],
        "mz_intensities": np.asarray(_fit_trace["mz_intensities"]),
    }

    # Reconstruction is compared in ABSOLUTE intensity units, so that additive
    # models (model18) and compositional ones (model13-17) are judged on the same
    # scale. Compositional models predict a per-scan fraction, so they get to use
    # the observed TIC; additive models already carry their own amplitude.
    _fit_expected = _fit_trace["mz_intensities"].mean(0)
    _fit_compositional = bool(
        np.allclose(_fit_expected.sum(axis=1), 1.0, rtol=0.05)
    )
    if _fit_compositional:
        _fit_reconstruction = _fit_intensities.sum(axis=1, keepdims=True) * _fit_expected
    else:
        _fit_reconstruction = _fit_expected * (
            _fit_intensities.mean() * _fit_intensities.shape[1]
        )

    # Baseline to beat: one fixed spectrum for the whole run, scaled by the
    # observed TIC. Anything at or below this has learned no chromatography.
    _fit_total_ss = ((_fit_intensities - _fit_intensities.mean()) ** 2).sum()
    _fit_r2 = 1 - (
        (_fit_intensities - _fit_reconstruction) ** 2
    ).sum() / _fit_total_ss
    _fit_baseline_r2 = 1 - (
        (
            _fit_intensities
            - _fit_intensities.sum(axis=1, keepdims=True)
            * (_fit_intensities.sum(0) / _fit_intensities.sum())
        )
        ** 2
    ).sum() / _fit_total_ss
    _fit_losses = np.asarray(_fit_svi_result.losses)
    _fit_nonfinite = int((~np.isfinite(_fit_losses)).sum())
    _fit_tail_drop = float(
        (_fit_losses[int(0.9 * len(_fit_losses))] - _fit_losses[-1])
        / max(abs(_fit_losses[-1]), 1e-12)
    )

    _fit_colors = sns.color_palette("deep", n_colors=_fit_n_components)
    _fit_fig, (_fit_loss_ax, _fit_weights_ax, _fit_components_ax) = plt.subplots(
        figsize=(15, 10), nrows=3
    )
    _fit_weights_ax.set_prop_cycle(plt.cycler(color=_fit_colors))
    _fit_components_ax.set_prop_cycle(plt.cycler(color=_fit_colors))
    _fit_loss_ax.plot(np.log1p(_fit_losses - _fit_losses.min()))
    _fit_loss_ax.set(xlabel="SVI step", ylabel="log(1 + loss - min)")

    _fit_tic = _fit_intensities.sum(axis=1)
    _fit_tic = _fit_tic / (float(_fit_tic.max()) or 1.0)
    _fit_offsets = 1.2 * np.arange(_fit_n_components)
    for _fit_weight_sample in _fit_trace["component_weights"]:
        _fit_scale = float(np.max(_fit_weight_sample)) or 1.0
        _fit_weights_ax.plot(
            _fit_times,
            _fit_weight_sample / _fit_scale + _fit_offsets,
            alpha=0.1,
        )
    _fit_weights_ax.plot(
        _fit_times, _fit_tic + _fit_n_components * 1.2, color="black"
    )
    _fit_weights_ax.hlines(
        1.2 * np.arange(_fit_n_components + 1),
        *_fit_weights_ax.get_xlim(),
        color="black",
        linewidth=0.5,
        linestyle="--",
    )
    _fit_weights_ax.set(xlabel="Time (s)", ylabel="Contribution", yticks=[])

    for _fit_component_sample in _fit_trace["component_intensities"].transpose(0, 2, 1):
        _fit_logged = np.log1p(np.maximum(_fit_component_sample, 0))
        _fit_scales = np.maximum(_fit_logged.max(axis=0), 1e-12)
        _fit_components_ax.plot(
            _fit_masses,
            _fit_logged / _fit_scales + _fit_offsets,
            alpha=0.1,
        )
    _fit_components_ax.hlines(
        _fit_offsets,
        *_fit_components_ax.get_xlim(),
        color="black",
        linewidth=0.5,
        linestyle="--",
    )
    _fit_components_ax.set(xlabel="m/z", ylabel="Intensity", yticks=[])
    _fit_fig.tight_layout()

    fit_results = {
        "sample_id": _fit_sample,
        "model": _fit_model_name,
        "seed": _fit_seed,
        "n_steps": _fit_n_steps,
        "n_components": _fit_n_components,
        "times": _fit_times,
        "masses": _fit_masses,
        "intensities": _fit_intensities,
        "trace": _fit_trace,
        "params": _fit_svi_result.params,
        "losses": _fit_losses,
        "reconstruction": _fit_reconstruction,
        "compositional": _fit_compositional,
        "r2": float(_fit_r2),
        "baseline_r2": float(_fit_baseline_r2),
    }

    mo.vstack(
        [
            _fit_controls,
            mo.md(
                f"**{_fit_model_name}** on **{_fit_sample}** &middot; K={_fit_n_components} "
                f"&middot; seed {_fit_seed}<br>"
                f"R&sup2; **{_fit_r2:.4f}** (constant mean-spectrum baseline "
                f"{_fit_baseline_r2:.4f}) &middot; final ELBO {_fit_losses[-1]:.4e} "
                f"&middot; ELBO drop over last 10% of steps {_fit_tail_drop:.2e} "
                f"&middot; {_fit_nonfinite} non-finite steps"
            ),
            _fit_fig,
        ],
        gap=1,
    )
    return (fit_results,)


@app.cell(hide_code=True)
def _(fit_results, mo):
    component_selection = mo.ui.multiselect(
        {
            f"Component {index + 1}": index
            for index in range(fit_results["n_components"])
        },
        value=[
            f"Component {index + 1}"
            for index in range(fit_results["n_components"])
        ],
        label="Components",
        full_width=True,
    )
    component_mass_range = mo.ui.range_slider(
        0, 100, step=0.5, value=[0, 100], show_value=True,
        label="Mass range (%)", full_width=True,
    )
    component_peak_threshold = mo.ui.slider(
        0, 100, step=1, value=10, show_value=True,
        label="Peak threshold (%)", full_width=True,
    )
    component_min_peak_distance = mo.ui.number(
        0.1, 100, step=0.1, value=10,
        label="Minimum peak distance (m/z)", full_width=True,
    )
    component_rotate_labels = mo.ui.checkbox(value=False, label="Rotate labels")
    return (
        component_mass_range,
        component_min_peak_distance,
        component_peak_threshold,
        component_rotate_labels,
        component_selection,
    )


@app.cell(hide_code=True)
def _(
    component_mass_range,
    component_min_peak_distance,
    component_peak_threshold,
    component_rotate_labels,
    component_selection,
    find_peaks,
    fit_results,
    math,
    mo,
    np,
    plt,
    sns,
):
    _component_controls = mo.vstack(
        [
            mo.md("### Inferred components"),
            component_selection,
            mo.hstack(
                [
                    component_mass_range,
                    component_peak_threshold,
                    component_min_peak_distance,
                    component_rotate_labels,
                ],
                widths=[2, 2, 1, 1],
                align="end",
            ),
        ],
        gap=1,
    )

    _component_selected = sorted(component_selection.value)
    mo.stop(
        not _component_selected,
        mo.vstack(
            [_component_controls, mo.callout("Select at least one component.", kind="info")]
        ),
    )
    _component_all_masses = np.asarray(fit_results["masses"])
    _component_start_pct, _component_end_pct = component_mass_range.value
    _component_start = _component_all_masses[0] + (
        _component_all_masses[-1] - _component_all_masses[0]
    ) * _component_start_pct / 100
    _component_end = _component_all_masses[0] + (
        _component_all_masses[-1] - _component_all_masses[0]
    ) * _component_end_pct / 100
    _component_mass_mask = (_component_all_masses >= _component_start) & (
        _component_all_masses <= _component_end
    )
    _component_masses = _component_all_masses[_component_mass_mask]
    mo.stop(
        len(_component_masses) < 2,
        mo.vstack([_component_controls, mo.callout("Select a wider mass range.", kind="warn")]),
    )

    _component_scans = np.asarray(
        fit_results["trace"]["component_intensities"]
    ).mean(axis=0)[:, _component_mass_mask]
    _component_weights = np.asarray(
        fit_results["trace"]["component_weights"]
    ).mean(axis=0)
    _component_times = np.asarray(fit_results["times"])
    _component_mass_step = float(np.median(np.diff(_component_masses)))
    _component_distance = max(
        1,
        math.ceil(float(component_min_peak_distance.value) / _component_mass_step),
    )
    _component_colors = sns.color_palette(
        "deep", n_colors=fit_results["n_components"]
    )
    _component_fig, (_component_mass_ax, _component_time_ax) = plt.subplots(
        nrows=2,
        figsize=(18, max(3.5, 2.8 * len(_component_selected))),
        sharey=True,
    )

    for _component_row, _component_index in enumerate(_component_selected):
        _component_scan = np.log(np.maximum(_component_scans[_component_index], 1e-3))
        _component_scan = _component_scan - _component_scan.min()
        _component_scan = _component_scan / (float(_component_scan.max()) or 1.0)
        _component_weight = np.maximum(_component_weights[:, _component_index], 0)
        _component_weight = _component_weight / (float(_component_weight.max()) or 1.0)
        _component_peaks = find_peaks(
            _component_scan,
            height=component_peak_threshold.value / 100,
            distance=_component_distance,
        )[0]
        _component_offset = 1.3 * _component_row
        _component_color = _component_colors[_component_index]
        _component_mass_ax.plot(
            _component_masses,
            _component_scan + _component_offset,
            color=_component_color,
        )
        _component_time_ax.plot(
            _component_times,
            _component_weight + _component_offset,
            color=_component_color,
        )
        _component_mass_ax.scatter(
            _component_masses[_component_peaks],
            _component_scan[_component_peaks] + _component_offset,
            color="black",
            s=30,
            zorder=10,
            marker="x",
            linewidth=1,
        )
        for _component_peak in _component_peaks:
            _component_mass_ax.text(
                _component_masses[_component_peak],
                _component_scan[_component_peak]
                + _component_offset
                + (0.1 if component_rotate_labels.value else 0.1),
                f"{_component_masses[_component_peak]:.0f}",
                color="black",
                fontsize=12,
                horizontalalignment="center",
                rotation=90 if component_rotate_labels.value else 0,
            )

    _component_guides = 1.2 * np.arange(len(_component_selected))
    _component_mass_ax.hlines(
        _component_guides,
        *_component_mass_ax.get_xlim(),
        color="black",
        linewidth=0.5,
        linestyle="--",
    )
    _component_time_ax.hlines(
        _component_guides,
        *_component_time_ax.get_xlim(),
        color="black",
        linewidth=0.5,
        linestyle="--",
    )
    _component_mass_ax.set(ylabel="Normalized intensity", xlabel="m/z", yticks=[])
    _component_time_ax.set(xlabel="Time (s)")
    _component_fig.tight_layout()


    mo.vstack([_component_controls, _component_fig], gap=1)
    return


if __name__ == "__main__":
    app.run()
