"""THz imaging for a time-domain grid CSV and a Gold pulse CSV."""

# Optional initial paths. Leave empty to choose files in the application.
SAMPLE_FILE = ""
GOLD_FILE = ""

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
from matplotlib.figure import Figure
from matplotlib.colors import Normalize


@dataclass
class ImagingData:
    waveforms: np.ndarray
    gold: np.ndarray
    dt_ps: float
    dx: float
    dy: float

    @classmethod
    def load(cls, sample_path, gold_path):
        with open(sample_path, encoding="utf-8-sig", newline="") as file:
            rows = csv.reader(file)
            header = next(rows, [])
            values = next(rows, [])
            meta = dict(zip(header, values))
            ny, nx = int(meta["No_Y"]), int(meta["No_X"])
            count = int(meta["No_values"])
            dx, dy, dt = (float(meta[key]) for key in ("dx", "dy", "dt/ps"))
            if min(nx, ny) < 1 or count < 4 or not np.isfinite([dt, dx, dy]).all() or dt <= 0:
                raise ValueError("Invalid imaging dimensions or sampling interval.")
            waveforms = np.empty((nx * ny, count))
            index = 0
            for row in rows:
                if not row:
                    continue
                if len(row) != count or index >= len(waveforms):
                    raise ValueError(f"Unexpected imaging data at CSV line {rows.line_num}.")
                waveforms[index] = row
                index += 1
            if index != len(waveforms):
                raise ValueError(f"Expected {len(waveforms)} traces, found {index}.")

        with open(gold_path, encoding="utf-8-sig", newline="") as file:
            gold_data = np.array([
                (float(row["Time_abs/ps"]), float(row["Signal_1/nA"]))
                for row in csv.DictReader(file)
            ])
        if gold_data.shape != (count, 2):
            raise ValueError(f"Gold pulse must contain {count} time samples.")
        times, gold = gold_data.T
        if not np.allclose(np.diff(times), dt, rtol=1e-4, atol=1e-8):
            raise ValueError("Gold and sample time sampling intervals must match.")
        if not np.isfinite(waveforms).all() or not np.isfinite(gold).all():
            raise ValueError("The input contains non-finite amplitudes.")
        return cls(waveforms.reshape(ny, nx, count), gold, dt, dx, dy)

    def window_weights(self, window, start_ps=0, end_ps=None):
        time = np.arange(self.gold.size) * self.dt_ps
        if end_ps is None:
            end_ps = float(time[-1])
        if not 0 <= start_ps < end_ps <= time[-1] + 1e-8:
            raise ValueError("Window start/end must be within the acquired relative time range.")
        mask = (time >= start_ps - 1e-8) & (time <= end_ps + 1e-8)
        count = int(mask.sum())
        if count < 4:
            raise ValueError("The time window must contain at least four samples.")
        functions = {
            "rectangular": lambda n: np.ones(n),
            "hann": np.hanning,
            "hamming": np.hamming,
            "gaussian": lambda n: np.exp(-0.5 * ((np.arange(n) - (n - 1) / 2) / (n / 4)) ** 2),
        }
        weights = np.zeros(self.gold.size)
        weights[mask] = functions[window](count)
        return weights

    def fft(self, window, start_ps=0, end_ps=None, gold_window=None,
            gold_start_ps=None, gold_end_ps=None):
        weights = self.window_weights(window, start_ps, end_ps)
        gold_weights = self.window_weights(
            gold_window if gold_window is not None else window,
            gold_start_ps if gold_start_ps is not None else start_ps,
            gold_end_ps if gold_end_ps is not None else end_ps,
        )
        freq = np.fft.rfftfreq(self.gold.size, self.dt_ps)
        sample_fft = np.abs(np.fft.rfft(
            (self.waveforms - self.waveforms.mean(axis=-1, keepdims=True)) * weights
        ))
        gold_fft = np.abs(np.fft.rfft((self.gold - self.gold.mean()) * gold_weights))
        return freq, sample_fft, gold_fft


@dataclass
class Result:
    maps: dict
    freq: np.ndarray
    spectra: np.ndarray
    gold_fft: np.ndarray
    valid: np.ndarray
    settings: dict


def extract(fft, target, low, high, floor_percent, order, window):
    freq, sample_fft, gold_fft = fft
    if not (0 < target <= freq[-1] and 0 <= low < high <= freq[-1]):
        raise ValueError(f"Choose frequencies within 0–{freq[-1]:g} THz.")
    if not 0 < floor_percent <= 100:
        raise ValueError("Gold floor must be greater than 0% and at most 100%.")
    if gold_fft[1:].max() <= 0:
        raise ValueError("Gold pulse has no non-DC FFT signal.")
    valid = (gold_fft >= gold_fft[1:].max() * floor_percent / 100) & (freq > 0)
    target_idx = int(np.abs(freq - target).argmin())
    if not valid[target_idx]:
        raise ValueError(f"Nearest bin ({freq[target_idx]:g} THz) is DC or below the Gold floor.")

    # Convert acquisition order to image rows before mapping pixels.
    spectra = np.full_like(sample_fft, np.nan)
    np.divide(sample_fft, gold_fft, out=spectra, where=valid)
    if order == "serpentine":
        spectra[1::2] = spectra[1::2, ::-1].copy()

    # T follows the continuous analyzer convention: squared field ratio.
    spectra **= 2
    in_band = (freq >= low - 1e-12) & (freq <= high + 1e-12) & valid
    if not in_band.any():
        raise ValueError("No frequency bins above the Gold floor in the selected band.")
    band_indices = np.flatnonzero(in_band)
    band_values = spectra[..., band_indices]
    min_idx = band_indices[np.argmin(band_values, axis=-1)]
    max_idx = band_indices[np.argmax(band_values, axis=-1)]
    min_T = np.take_along_axis(spectra, min_idx[..., None], axis=-1)[..., 0]
    max_T = np.take_along_axis(spectra, max_idx[..., None], axis=-1)[..., 0]
    maps = {
        "fixed_T": spectra[..., target_idx],
        "min_T_frequency_THz": freq[min_idx],
        "max_T_frequency_THz": freq[max_idx],
        "min_T": min_T,
        "max_T": max_T,
    }
    settings = {
        "requested_frequency_THz": target,
        "actual_frequency_THz": float(freq[target_idx]),
        "extrema_band_THz": [low, high],
        "gold_floor_percent_of_non_dc_max": floor_percent,
        "window": window,
        "scan_order": order,
        "fft_bin_spacing_THz": float(freq[1]),
        "extrema_rule": "Global min/max over valid FFT bins in inclusive band; ties use lowest frequency",
        "T_definition": "abs(FFT(sample) / FFT(Gold)) ** 2",
        "valid_band_bins": int(in_band.sum()),
        "actual_valid_band_limits_THz": [float(freq[in_band][0]), float(freq[in_band][-1])],
        "min_at_valid_band_edge_pixels": int(((min_idx == band_indices[0]) | (min_idx == band_indices[-1])).sum()),
        "max_at_valid_band_edge_pixels": int(((max_idx == band_indices[0]) | (max_idx == band_indices[-1])).sum()),
    }
    return Result(maps, freq, spectra, gold_fft, valid, settings)


def draw_maps(figure, result):
    figure.clear()
    axes = figure.subplots(2, 3).ravel()
    axes[-1].set_visible(False)
    axes = axes[:-1]
    target = result.settings["actual_frequency_THz"]
    titles = [f"{target:g} THz T", "Frequency at min T (THz)",
              "Frequency at max T (THz)", "Minimum T", "Maximum T"]
    for ax, values, title in zip(axes, result.maps.values(), titles):
        cmap = "viridis" if "Frequency" not in title else "plasma"
        image = ax.imshow(values, origin="upper", interpolation="nearest", cmap=cmap)
        ax.set_facecolor("#dddddd")
        ax.set(title=title, xlabel="Column index", ylabel="Row index")
        figure.colorbar(image, ax=ax, shrink=0.72)
    return axes


def export(folder, data, result, sample_path, gold_path):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(12, 8), layout="constrained")
    draw_maps(figure, result)
    figure.savefig(folder / "imaging_maps.png", dpi=200)
    ny, nx = data.waveforms.shape[:2]
    with (folder / "pixel_results.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["row", "column", "scan_index", *result.maps])
        for y, x in np.ndindex(ny, nx):
            scan_x = nx - 1 - x if result.settings["scan_order"] == "serpentine" and y % 2 else x
            values = [map_[y, x] for map_ in result.maps.values()]
            writer.writerow([y, x, y * nx + scan_x, *(
                float(value) if np.isfinite(value) else "" for value in values
            )])
    settings = {
        **result.settings,
        "sample_file": str(sample_path),
        "gold_file": str(gold_path),
        "shape_yx": [ny, nx],
        "samples_per_pixel": data.gold.size,
        "dt_ps": data.dt_ps,
        "dx_as_recorded": data.dx,
        "dy_as_recorded": data.dy,
    }
    (folder / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")


def launch_gui():
    from PyQt6.QtWidgets import (
        QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
        QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QTabWidget,
        QVBoxLayout, QWidget, QSlider, QSpinBox,
    )
    from PyQt6.QtCore import Qt, QTimer
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

    class ImagingAnalyzer(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("THz Imaging Analyzer — 파일 선택 및 주파수 분석")
            self.resize(1250, 950)
            self.paths = {}
            self.data = None
            self.result = None
            self.cached_fft = None
            root = QWidget()
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)

            self.file_labels = {}
            for key, label in (("sample", "샘플 파일 선택 (.tim.csv)"), ("gold", "Gold 파일 선택 (.pulse.csv)")):
                row = QHBoxLayout()
                button = QPushButton(label)
                name = QLabel("No file selected")
                self.file_labels[key] = name
                button.clicked.connect(lambda _, k=key, n=name: self.pick(k, n))
                row.addWidget(button)
                row.addWidget(name, 1)
                layout.addLayout(row)

            controls = QFormLayout()
            layout.addLayout(controls)
            self.target = self.number(controls, "Target frequency (THz)", 0.58)
            self.low = self.number(controls, "Min/max T band start (THz)", 1.5)
            self.high = self.number(controls, "Min/max T band end (THz)", 2.0)
            self.floor = self.number(controls, "Gold amplitude floor (%)", 5, maximum=100)
            self.order = QComboBox()
            self.order.addItems(["raster", "serpentine"])
            controls.addRow("Scan order (verify against instrument)", self.order)
            self.window = QComboBox()
            self.window.addItems(["rectangular", "hann", "hamming", "gaussian"])

            buttons = QHBoxLayout()
            layout.addLayout(buttons)
            run = QPushButton("FFT 분석 / 맵 갱신")
            run.clicked.connect(self.run)
            buttons.addWidget(run)
            self.save_button = QPushButton("결과 저장 (PNG / CSV)")
            self.save_button.setEnabled(False)
            self.save_button.clicked.connect(self.save)
            buttons.addWidget(self.save_button)
            self.status = QLabel("① 샘플과 Gold 파일 선택 → ② FFT 분석 → ③ 주파수별 T 이미지에서 주파수 조절")
            layout.addWidget(self.status)

            tabs = QTabWidget()
            layout.addWidget(tabs, 1)
            self.map_figure, self.map_canvas = self.add_tab(tabs, "Imaging Maps")
            self.pixel_figure, self.pixel_canvas = self.add_tab(tabs, "Pixel Traces and FFT")
            self.pixel_tab = tabs.widget(1)
            self.tabs = tabs
            self.map_canvas.mpl_connect("button_press_event", self.select_pixel)
            self.map_axes = []

            self.frequency_figure, self.frequency_canvas = self.add_tab(tabs, "주파수별 T 이미지")
            self.frequency_panel = tabs.widget(2)
            frequency_layout = self.frequency_panel.layout()
            self.frequency_slider = QSlider(Qt.Orientation.Horizontal)
            self.frequency_slider.setTracking(True)
            self.frequency_spin = QDoubleSpinBox()
            self.frequency_spin.setDecimals(4)
            self.frequency_spin.setValue(0.58)
            self.frequency_label = QLabel("Run FFT to browse frequency maps.")
            frequency_row = QHBoxLayout()
            frequency_row.addWidget(QLabel("Frequency (THz)"))
            frequency_row.addWidget(self.frequency_spin)
            frequency_row.addWidget(self.frequency_slider, 1)
            frequency_layout.insertLayout(0, frequency_row)
            frequency_layout.insertWidget(1, self.frequency_label)
            self.frequency_slider.setEnabled(False)
            self.frequency_spin.setEnabled(False)
            self.frequency_image = None
            self.frequency_axis = None
            self.frequency_slider.valueChanged.connect(self.update_frequency_map)
            self.frequency_spin.valueChanged.connect(self.set_frequency)
            self.frequency_canvas.mpl_connect("button_press_event", self.select_frequency_pixel)

            self.preview_figure, self.preview_canvas = self.add_tab(tabs, "Windowing / T Spectrum")
            preview_layout = tabs.widget(3).layout()
            preview_form = QFormLayout()
            gate_row = QHBoxLayout()
            preview_layout.insertLayout(0, gate_row)
            sample_form, gold_form = QFormLayout(), QFormLayout()
            gate_row.addLayout(sample_form)
            gate_row.addLayout(gold_form)
            sample_form.addRow(QLabel("SAMPLE WINDOW"))
            gold_form.addRow(QLabel("GOLD / REFERENCE WINDOW"))
            sample_form.addRow("Window function", self.window)
            self.gold_window = QComboBox()
            self.gold_window.addItems(["rectangular", "hann", "hamming", "gaussian"])
            gold_form.addRow("Window function", self.gold_window)
            self.window_start = self.number(sample_form, "Start (relative ps)", 0)
            self.window_end = self.number(sample_form, "End (relative ps)", 49.95)
            self.gold_start = self.number(gold_form, "Start (relative ps)", 0)
            self.gold_end = self.number(gold_form, "End (relative ps)", 49.95)
            self.time_controls = (self.window_start, self.window_end, self.gold_start, self.gold_end)
            self.time_sliders = []
            for spin, form in zip(self.time_controls, (sample_form, sample_form, gold_form, gold_form)):
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setEnabled(False)
                form.addRow(slider)
                self.time_sliders.append(slider)
                slider.valueChanged.connect(
                    lambda index, control=spin: control.setValue(index * self.data.dt_ps)
                )
                spin.valueChanged.connect(self.schedule_preview)
            preview_layout.insertLayout(1, preview_form)
            self.preview_row = QSpinBox()
            self.preview_column = QSpinBox()
            preview_form.addRow("Example pixel row", self.preview_row)
            preview_form.addRow("Example pixel column", self.preview_column)
            self.preview_row.valueChanged.connect(self.schedule_preview)
            self.preview_column.valueChanged.connect(self.schedule_preview)
            self.window.currentTextChanged.connect(self.schedule_preview)
            self.gold_window.currentTextChanged.connect(self.schedule_preview)
            self.floor.valueChanged.connect(self.schedule_preview)
            self.order.currentTextChanged.connect(self.schedule_preview)
            self.preview_timer = QTimer(self)
            self.preview_timer.setSingleShot(True)
            self.preview_timer.setInterval(120)
            self.preview_timer.timeout.connect(self.draw_window_preview)
            self.preview_notice = QLabel("Select both files to preview the time window and T spectrum.")
            self.preview_notice.setWordWrap(True)
            preview_layout.insertWidget(2, self.preview_notice)
            apply_window = QPushButton("Apply window to all pixels / Update maps")
            apply_window.clicked.connect(self.run)
            preview_layout.insertWidget(3, apply_window)
            for key, path in (("sample", SAMPLE_FILE), ("gold", GOLD_FILE)):
                if path:
                    self.paths[key] = Path(path).expanduser()
                    self.file_labels[key].setText(str(self.paths[key]))
            if len(self.paths) == 2:
                self.status.setText("Configured input files selected. Click Run FFT / Update Maps.")


        def number(self, form, label, default, maximum=1000):
            spin = QDoubleSpinBox()
            spin.setDecimals(4)
            spin.setRange(0, maximum)
            spin.setSingleStep(0.02)
            spin.setValue(default)
            form.addRow(label, spin)
            return spin

        def add_tab(self, tabs, label):
            panel = QWidget()
            layout = QVBoxLayout(panel)
            figure = Figure(layout="constrained")
            canvas = FigureCanvasQTAgg(figure)
            layout.addWidget(NavigationToolbar2QT(canvas, panel))
            layout.addWidget(canvas)
            tabs.addTab(panel, label)
            return figure, canvas

        def pick(self, key, label):
            path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV files (*.csv)")
            if path:
                self.paths[key] = path
                label.setText(Path(path).name)
                self.data = self.cached_fft = self.result = None
                self.map_axes = []
                self.frequency_image = self.frequency_axis = None
                self.frequency_slider.setEnabled(False)
                self.frequency_spin.setEnabled(False)
                self.frequency_label.setText("FFT 분석 후 주파수를 조절할 수 있습니다.")
                for figure, canvas in (
                    (self.map_figure, self.map_canvas), (self.pixel_figure, self.pixel_canvas),
                    (self.frequency_figure, self.frequency_canvas)
                ):
                    figure.clear()
                    canvas.draw_idle()
                self.save_button.setEnabled(False)
                self.status.setText("Run FFT to process the selected files.")
                self.preview_figure.clear()
                self.preview_canvas.draw_idle()
                if len(self.paths) == 2:
                    try:
                        self.data = ImagingData.load(self.paths["sample"], self.paths["gold"])
                        self.configure_window_panel()
                        self.tabs.setCurrentIndex(3)
                    except (OSError, ValueError, KeyError, csv.Error) as error:
                        QMessageBox.critical(self, "Input Error", str(error))


        def run(self):
            self.save_button.setEnabled(False)
            try:
                if len(self.paths) != 2:
                    raise ValueError("Select both CSV files first.")
                if self.data is None:
                    self.data = ImagingData.load(self.paths["sample"], self.paths["gold"])
                    self.configure_window_panel()
                window = self.window.currentText()
                window_key = (
                    window, self.window_start.value(), self.window_end.value(),
                    self.gold_window.currentText(), self.gold_start.value(), self.gold_end.value(),
                )
                if self.cached_fft is None or window_key != self.cached_fft[0]:
                    self.cached_fft = (window_key, self.data.fft(*window_key))
                self.result = extract(
                    self.cached_fft[1], self.target.value(), self.low.value(), self.high.value(),
                    self.floor.value(), self.order.currentText(), window,
                )
                self.result.settings.update({
                    "time_window_start_ps": self.window_start.value(),
                    "time_window_end_ps": self.window_end.value(),
                    "gold_window": self.gold_window.currentText(),
                    "gold_time_window_start_ps": self.gold_start.value(),
                    "gold_time_window_end_ps": self.gold_end.value(),
                    "time_origin": "Relative to first sample; independent sample and Gold windows",
                    "processing": "Subtract full trace mean; apply selected window inside gate; zero outside",
                })
                self.map_axes = draw_maps(self.map_figure, self.result)
                self.map_canvas.draw_idle()
                self.configure_frequency_panel()
                self.tabs.setCurrentWidget(self.frequency_panel)
                ny, nx = self.data.waveforms.shape[:2]
                self.draw_pixel(ny // 2, nx // 2)
                settings = self.result.settings
                self.status.setText(
                    f"{nx} × {ny} pixels | {settings['actual_frequency_THz']:g} THz FFT bin | "
                    f"{settings['valid_band_bins']} valid frequency bins | Click a map to inspect a pixel."
                )
                self.save_button.setEnabled(True)
            except (OSError, ValueError, KeyError, csv.Error) as error:
                QMessageBox.critical(self, "Analysis Error", str(error))

        def configure_window_panel(self):
            ny, nx, count = self.data.waveforms.shape
            for control, maximum in ((self.preview_row, ny - 1), (self.preview_column, nx - 1)):
                control.blockSignals(True)
                control.setRange(0, maximum)
                control.setValue(maximum // 2)
                control.blockSignals(False)
            end = (count - 1) * self.data.dt_ps
            for spin, slider, initial in zip(
                    self.time_controls, self.time_sliders, (0, end, 0, end)):
                spin.blockSignals(True)
                spin.setRange(0, end)
                spin.setSingleStep(self.data.dt_ps)
                spin.setValue(initial)
                spin.blockSignals(False)
                slider.blockSignals(True)
                slider.setRange(0, count - 1)
                slider.setValue(round(initial / self.data.dt_ps))
                slider.blockSignals(False)
                slider.setEnabled(True)
            self.draw_window_preview()

        def schedule_preview(self, *args):
            if self.data is not None:
                self.save_button.setEnabled(False)
                self.preview_notice.setText("Preview settings changed. Apply window to update all maps and enable export.")
                self.preview_timer.start()

        def draw_window_preview(self):
            if self.data is None:
                return
            data = self.data
            for spin, slider in zip(self.time_controls, self.time_sliders):
                slider.blockSignals(True)
                slider.setValue(round(spin.value() / data.dt_ps))
                slider.blockSignals(False)
            try:
                weights = data.window_weights(
                    self.window.currentText(), self.window_start.value(), self.window_end.value()
                )
                gold_weights = data.window_weights(
                    self.gold_window.currentText(), self.gold_start.value(), self.gold_end.value()
                )
            except ValueError as error:
                self.preview_notice.setText(str(error))
                self.preview_figure.clear()
                self.preview_canvas.draw_idle()
                return
            y, x = self.preview_row.value(), self.preview_column.value()
            scan_x = data.waveforms.shape[1] - 1 - x if self.order.currentText() == "serpentine" and y % 2 else x
            sample = data.waveforms[y, scan_x]
            sample = sample - sample.mean()
            gold = data.gold - data.gold.mean()
            sample_processed, gold_processed = sample * weights, gold * gold_weights
            freq = np.fft.rfftfreq(data.gold.size, data.dt_ps)
            gold_fft = np.abs(np.fft.rfft(gold_processed))
            sample_fft = np.abs(np.fft.rfft(sample_processed))
            valid = (freq > 0) & (gold_fft > 0) & (
                gold_fft >= gold_fft[1:].max() * self.floor.value() / 100
            )
            T = np.full_like(freq, np.nan)
            np.divide(sample_fft, gold_fft, out=T, where=valid)
            T **= 2
            self.preview_figure.clear()
            raw_ax, processed_ax, spectrum_ax = self.preview_figure.subplots(3, 1)
            time = np.arange(data.gold.size) * data.dt_ps
            raw_ax.plot(time, sample, label="Pixel (mean removed)", lw=0.8)
            raw_ax.plot(time, gold, label="Gold (mean removed)", lw=0.8, alpha=0.7)
            raw_ax.axvspan(self.window_start.value(), self.window_end.value(),
                           color="tab:blue", alpha=0.12, label="Sample gate")
            raw_ax.axvspan(self.gold_start.value(), self.gold_end.value(),
                           color="tab:orange", alpha=0.12, label="Gold gate")
            raw_ax.set(xlabel="Relative time (ps)", ylabel="Signal", title=f"Example pixel row {y}, column {x}")
            raw_ax.legend(fontsize=7)
            processed_ax.plot(time, sample_processed, label="Windowed pixel", lw=0.8)
            processed_ax.plot(time, gold_processed, label="Windowed Gold", lw=0.8, alpha=0.7)
            processed_ax.set(xlabel="Relative time (ps)", ylabel="Signal")
            processed_ax.legend(fontsize=7)
            spectrum_ax.plot(freq, T, lw=0.9)
            spectrum_ax.set(xlabel="Frequency (THz)", ylabel="T = |sample / Gold|²",
                            xlim=(0, min(freq[-1], max(3, self.high.value() * 1.2))))
            self.preview_canvas.draw_idle()

        def configure_frequency_panel(self):
            freq = self.result.freq
            self.frequency_slider.blockSignals(True)
            self.frequency_slider.setRange(0, len(freq) - 1)
            self.frequency_slider.setValue(int(np.abs(freq - self.frequency_spin.value()).argmin()))
            self.frequency_slider.blockSignals(False)
            self.frequency_spin.blockSignals(True)
            self.frequency_spin.setRange(float(freq[0]), float(freq[-1]))
            self.frequency_spin.setSingleStep(float(freq[1] - freq[0]))
            self.frequency_spin.blockSignals(False)
            self.frequency_slider.setEnabled(True)
            self.frequency_spin.setEnabled(True)
            # Rebuild the colorbar once per FFT run, then reuse its image on slider moves.
            self.frequency_figure.clear()
            self.frequency_axis = self.frequency_figure.add_subplot(111)
            self.frequency_axis.set_facecolor("#dddddd")
            self.frequency_image = self.frequency_axis.imshow(
                self.result.spectra[..., 0], origin="upper",
                interpolation="nearest", cmap="viridis",
            )
            self.frequency_axis.set(xlabel="Column index", ylabel="Row index")
            self.frequency_figure.colorbar(self.frequency_image, ax=self.frequency_axis, label="T")
            self.update_frequency_map(self.frequency_slider.value())

        def set_frequency(self, value):
            if self.result is None:
                return
            index = int(np.abs(self.result.freq - value).argmin())
            self.frequency_slider.blockSignals(True)
            self.frequency_slider.setValue(index)
            self.frequency_slider.blockSignals(False)
            self.update_frequency_map(index)

        def update_frequency_map(self, index):
            if self.result is None or self.frequency_image is None:
                return
            frequency = float(self.result.freq[index])
            self.frequency_spin.blockSignals(True)
            self.frequency_spin.setValue(frequency)
            self.frequency_spin.blockSignals(False)
            values = self.result.spectra[..., index]
            self.frequency_image.set_data(values)
            finite = values[np.isfinite(values)]
            if finite.size:
                low, high = float(finite.min()), float(finite.max())
                if low == high:
                    margin = max(abs(low) * 0.01, 1e-6)
                    low, high = low - margin, high + margin
                # Reset first to keep the attached colorbar bounds consistent.
                self.frequency_image.set_norm(Normalize(low, high))
                message = f"T range {finite.min():.4g}–{finite.max():.4g}. Color scale adjusts per frequency."
            else:
                self.frequency_image.set_clim(0, 1)
                message = "Masked: DC or Gold amplitude below the configured floor."
            self.frequency_axis.set_title(f"T at {frequency:g} THz")
            self.frequency_label.setText(
                f"FFT bin {index} | {frequency:g} THz | {message}"
            )
            self.frequency_canvas.draw_idle()

        def select_frequency_pixel(self, event):
            if (self.result is not None and event.inaxes is self.frequency_axis
                    and event.button == 1 and not self.frequency_canvas.toolbar.mode):
                self.draw_pixel(round(event.ydata), round(event.xdata))
                self.tabs.setCurrentWidget(self.pixel_tab)

        def select_pixel(self, event):
            if event.inaxes in self.map_axes and event.button == 1 and not self.map_canvas.toolbar.mode:
                self.draw_pixel(round(event.ydata), round(event.xdata))
                self.tabs.setCurrentWidget(self.pixel_tab)

        def draw_pixel(self, y, x):
            data, result = self.data, self.result
            ny, nx = data.waveforms.shape[:2]
            if not (0 <= y < ny and 0 <= x < nx):
                return
            for control, value in ((self.preview_row, y), (self.preview_column, x)):
                control.blockSignals(True)
                control.setValue(value)
                control.blockSignals(False)
            self.draw_window_preview()
            scan_x = nx - 1 - x if result.settings["scan_order"] == "serpentine" and y % 2 else x
            self.pixel_figure.clear()
            td, fft, ratio = self.pixel_figure.subplots(3, 1)
            time = np.arange(data.gold.size) * data.dt_ps
            td.plot(time, data.waveforms[y, scan_x], label="Pixel")
            td.plot(time, data.gold, label="Gold", alpha=0.75)
            td.set(title=f"Row {y}, column {x}", xlabel="Time from first sample (ps)", ylabel="Signal")
            td.legend(loc="upper right")
            freq = result.freq
            fft.plot(freq, self.cached_fft[1][1][y, scan_x], label="Pixel")
            fft.plot(freq, result.gold_fft, label="Gold", alpha=0.75)
            fft.set(xlabel="Frequency (THz)", ylabel="|FFT|")
            fft.legend(loc="upper right")
            ratio.plot(freq, result.spectra[y, x])
            low, high = result.settings["extrema_band_THz"]
            ratio.axvspan(low, high, alpha=0.15)
            ratio.axvline(result.settings["actual_frequency_THz"], color="gray", ls="--")
            for kind, color in (("min", "tab:blue"), ("max", "tab:red")):
                frequency = result.maps[f"{kind}_T_frequency_THz"][y, x]
                value = result.maps[f"{kind}_T"][y, x]
                ratio.scatter(frequency, value, color=color,
                              label=f"{kind} T = {value:.4g} at {frequency:g} THz")
            ratio.legend(loc="upper right", fontsize=8)
            ratio.set(xlabel="Frequency (THz)", ylabel="T = |sample / Gold|²",
                      xlim=(0, min(freq[-1], 1.5 * max(high, result.settings["actual_frequency_THz"]))))
            self.pixel_canvas.draw_idle()

        def save(self):
            folder = QFileDialog.getExistingDirectory(self, "Select Export Parent Folder")
            if folder:
                from tempfile import mkdtemp
                try:
                    destination = mkdtemp(prefix="thz_imaging_", dir=folder)
                    export(destination, self.data, self.result, self.paths["sample"], self.paths["gold"])
                    self.preview_figure.savefig(Path(destination) / "windowing_preview.png", dpi=200)
                    frequency_index = self.frequency_slider.value()
                    np.savetxt(
                        Path(destination) / "selected_frequency_T.csv",
                        self.result.spectra[..., frequency_index], delimiter=",",
                        header=f"T at {self.result.freq[frequency_index]:g} THz; rows=y, columns=x",
                    )
                    self.frequency_figure.savefig(
                        Path(destination) / "selected_frequency_T.png", dpi=200
                    )
                    QMessageBox.information(self, "Export Complete", destination)
                except OSError as error:
                    QMessageBox.critical(self, "Export Error", str(error))

    app = QApplication(sys.argv[:1])
    window = ImagingAnalyzer()
    window.show()
    return app.exec()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true", help="Open the interactive application.")
    parser.add_argument("--sample", type=Path, default=Path(SAMPLE_FILE).expanduser() if SAMPLE_FILE else None)
    parser.add_argument("--gold", type=Path, default=Path(GOLD_FILE).expanduser() if GOLD_FILE else None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frequency", type=float, default=0.58)
    parser.add_argument("--band", type=float, nargs=2, default=(1.5, 2.0))
    parser.add_argument("--gold-floor", type=float, default=5)
    parser.add_argument("--order", choices=("raster", "serpentine"), default="raster")
    parser.add_argument("--window", choices=("rectangular", "hann", "hamming", "gaussian"), default="rectangular")
    args = parser.parse_args()
    if args.gui or len(sys.argv) == 1:
        return launch_gui()
    if not all((args.sample, args.gold, args.output)):
        parser.error("Provide --sample, --gold and --output together for batch export.")
    try:
        data = ImagingData.load(args.sample, args.gold)
        result = extract(
            data.fft(args.window), args.frequency, *args.band,
            args.gold_floor, args.order, args.window,
        )
        export(args.output, data, result, args.sample, args.gold)
        print(json.dumps(result.settings, indent=2))
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.exit(1, f"Analysis failed: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
