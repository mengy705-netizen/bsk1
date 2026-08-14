from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import queue
import sys
import threading
import traceback
from typing import Any

import numpy as np

from core.geometry import WingGeometry, build_sensor_coordinates, make_grid
from core.inference import ModelRunner
from core.mesh_io import load_prepared_mesh, prepare_mesh
from core.model_registry import read_active_manifest, read_pending_manifest
from core.sensor_layout import (
    export_sensor_coordinates_csv,
    load_sensor_coordinates_file,
    read_active_sensor_layout,
    save_active_sensor_coordinates,
)
from core.wing_geometry_registry import (
    read_active_wing_geometry,
    save_active_wing_geometry,
)
from core.reconstruction import StrainReconstructor
from core.keras_v3_backend import (
    KerasV3TrainingConfig,
    TrainingCancelled,
    activate_existing_keras_v3,
    activate_pending_keras_v3,
    train_keras_v3,
)
from core.pretrain_backend import (
    PretrainConfig,
    PretrainCancelled,
    activate_existing_pretrain,
    read_active_pretrain,
    train_pretrained_models,
)
from core.external_model_registry import (
    clear_active_external_model,
    read_active_external_model,
)
from core.seven_model_group import (
    SLOT_DEFINITIONS as SEVEN_MODEL_SLOTS,
    auto_match_seven_model_directory,
    install_seven_model_group,
    inspect_seven_model_group,
    preview_seven_model_group,
)
from core.visualization import create_figure, draw_result


BASE_DIR = Path(__file__).resolve().parent


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or BASE_DIR / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["base_dir"] = str(BASE_DIR)
    return config


def runtime_config(
    config: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    active_model = read_active_manifest(BASE_DIR)
    active_layout = read_active_sensor_layout(BASE_DIR)
    active_geometry = read_active_wing_geometry(BASE_DIR)
    runtime = deepcopy(config)
    # 后台模型、机翼三维模型和传感器布置分别独立管理。
    geometry_meta = active_geometry.get("geometry", {}) if active_geometry else {}
    if not geometry_meta and active_layout:
        # 兼容旧版本：旧传感器布置文件可能仍包含几何信息。
        geometry_meta = active_layout.get("geometry", {})
    if all(key in geometry_meta for key in ("span_mm", "root_chord_mm", "tip_chord_mm")):
        runtime["geometry"]["span_mm"] = float(geometry_meta["span_mm"])
        runtime["geometry"]["root_chord_mm"] = float(geometry_meta["root_chord_mm"])
        runtime["geometry"]["tip_chord_mm"] = float(geometry_meta["tip_chord_mm"])
    if active_layout and active_layout.get("sensor_coordinates_mm"):
        runtime["sensor_coordinates_mm"] = active_layout["sensor_coordinates_mm"]
    return runtime, active_model, active_layout, active_geometry


def make_context(config: dict[str, Any]):
    geometry_cfg = config["geometry"]
    geometry = WingGeometry(
        span_mm=float(geometry_cfg["span_mm"]),
        root_chord_mm=float(geometry_cfg["root_chord_mm"]),
        tip_chord_mm=float(geometry_cfg["tip_chord_mm"]),
    )
    custom = config.get("sensor_coordinates_mm")
    if custom is not None:
        sensor_coords_3d = np.asarray(custom, dtype=float)
        if sensor_coords_3d.shape != (48, 3):
            raise ValueError("当前传感器布置文件不是48×3坐标。")
        sensor_coords = sensor_coords_3d[:, :2].copy()
    else:
        sensor_coords = build_sensor_coordinates(
            geometry,
            config["sensors"]["span_fractions"],
            config["sensors"]["chord_fractions"],
        )
        sensor_coords_3d = np.column_stack(
            [sensor_coords, np.zeros((sensor_coords.shape[0],), dtype=float)]
        )
    grid_z, grid_x, mask = make_grid(
        geometry,
        int(geometry_cfg["grid_span_points"]),
        int(geometry_cfg["grid_chord_points"]),
    )
    return geometry, sensor_coords, sensor_coords_3d, grid_z, grid_x, mask


def generate_demo_strains(sensor_coords: np.ndarray, geometry: WingGeometry) -> np.ndarray:
    load_z = 0.64 * geometry.span_mm
    load_x = 0.42 * float(geometry.local_chord(load_z))
    z = sensor_coords[:, 0]
    x = sensor_coords[:, 1]
    root_bending = 150.0 * (1.0 - z / geometry.span_mm) ** 1.3
    local = 310.0 * np.exp(
        -0.5
        * (
            ((z - load_z) / (0.12 * geometry.span_mm)) ** 2
            + ((x - load_x) / (0.13 * geometry.root_chord_mm)) ** 2
        )
    )
    chord_gradient = 45.0 * (x / np.maximum(geometry.local_chord(z), 1e-9) - 0.5)
    deterministic_noise = 5.0 * np.sin(np.arange(sensor_coords.shape[0]) * 1.37)
    return root_bending + local + chord_gradient + deterministic_noise


def render_demo(output_path: Path) -> None:
    base_config = load_config()
    config, active, active_layout, active_geometry = runtime_config(base_config)
    geometry, sensor_coords, sensor_coords_3d, grid_z, grid_x, mask = make_context(config)
    strain = generate_demo_strains(sensor_coords, geometry)
    prediction = ModelRunner(config).predict(strain, sensor_coords, "Demo")
    field = StrainReconstructor(config).reconstruct(
        sensor_coords,
        strain,
        prediction,
        grid_z,
        grid_x,
        "Demo load-guided RBF",
    )
    mesh_source = active_geometry or active_layout
    mesh = load_prepared_mesh(mesh_source.get("mesh_npz_path_resolved")) if mesh_source else None
    figure = create_figure(config)
    draw_result(
        figure,
        geometry,
        sensor_coords,
        strain,
        grid_z,
        grid_x,
        mask,
        field,
        prediction,
        config,
        mesh_data=mesh,
        sensor_coords_3d=sensor_coords_3d,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")


def run_gui(preview_output: Path | None = None) -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    class ChineseNavigationToolbar(NavigationToolbar2Tk):
        toolitems = (
            ("主页", "恢复初始视图", "home", "home"),
            ("后退", "返回上一个视图", "back", "back"),
            ("前进", "前往下一个视图", "forward", "forward"),
            (None, None, None, None),
            ("平移", "按住鼠标左键平移，右键缩放", "move", "pan"),
            ("框选缩放", "拖动矩形区域进行缩放", "zoom_to_rect", "zoom"),
            ("布局", "调整图形子区布局", "subplots", "configure_subplots"),
            (None, None, None, None),
            ("保存", "保存当前图形", "filesave", "save_figure"),
        )

    model_mode_map = {
        "当前外部模型": "Active External",
        "当前激活迁移模型": "Active AE-FST",
        "手动选择迁移模型包": "AE-FST Bundle",
        "演示模型": "Demo",
        "自定义 Python 接口": "Custom Python",
        "Joblib / sklearn 模型": "Joblib / sklearn",
        "TorchScript 模型": "TorchScript",
    }
    reconstruction_mode_map = {
        "LC-Kriging（内置用户代码）": "User LC-Kriging",
        "演示：载荷引导 RBF": "Demo load-guided RBF",
        "自定义 Python（LC-Kriging）": "Custom Python",
    }

    class WingMonitorApp:
        def __init__(self, root: tk.Tk):
            self.root = root
            self.base_config = load_config()
            self.config, self.active_manifest, self.active_layout_manifest, self.active_geometry_manifest = runtime_config(self.base_config)
            self._rebuild_context()
            self.root.title(self.base_config.get("app_title", "机翼应变监测与模型管理系统"))
            self.root.geometry("1540x900")
            self.root.minsize(1220, 740)

            self.training_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
            self.training_thread: threading.Thread | None = None
            self.cancel_event: threading.Event | None = None
            self.current_figure = create_figure(self.config)
            self.current_prediction: dict[str, float] | None = None

            active_exists = bool(self.active_manifest and self.active_manifest.get("bundle_path_resolved"))
            self.active_external_manifest = read_active_external_model(BASE_DIR)
            external_exists = bool(
                self.active_external_manifest
                and self.active_external_manifest.get("model_path_resolved")
            )
            self.model_mode = tk.StringVar(
                value=(
                    "当前外部模型"
                    if external_exists
                    else ("当前激活迁移模型" if active_exists else "演示模型")
                )
            )
            default_model_path = (
                self.active_external_manifest.get("model_path_resolved", "")
                if external_exists
                else (
                    self.active_manifest.get("bundle_path_resolved", "")
                    if active_exists
                    else str(BASE_DIR / "adapters" / "user_model.py")
                )
            )
            self.model_path = tk.StringVar(value=default_model_path)
            self.reconstruction_path = tk.StringVar(
                value=str(BASE_DIR / "adapters" / "user_reconstruction.py")
            )
            self.reconstruction_mode = tk.StringVar(value="LC-Kriging（内置用户代码）")
            self.status_text = tk.StringVar(value="就绪：请导入48通道应变数据，或使用演示数据。")
            self.load_x_text = tk.StringVar(value="—")
            self.load_z_text = tk.StringVar(value="—")
            self.load_n_text = tk.StringVar(value="—")
            self.damage_text = tk.StringVar(value="—")
            self.damage_x_text = tk.StringVar(value="—")
            self.damage_z_text = tk.StringVar(value="—")
            self.damage_size_text = tk.StringVar(value="—")

            self.pretrain_paths = {
                "sim_u": tk.StringVar(),
                "sim_d": tk.StringVar(),
            }
            self.pretrain_existing_path = tk.StringVar()
            self.pretrain_bp_trials = tk.IntVar(value=10)
            self.pretrain_bp_epochs = tk.IntVar(value=200)
            self.pretrain_cnn_epochs = tk.IntVar(value=250)
            self.pretrain_batch_size = tk.IntVar(value=128)
            self.pretrain_progress = tk.DoubleVar(value=0.0)
            self.pretrain_status = tk.StringVar(value="第1步：请选择仿真未损伤和仿真损伤数据。")
            self.pretrain_model_text = tk.StringVar()
            self.active_pretrain_manifest = read_active_pretrain(BASE_DIR)

            self.transfer_paths = {
                "sim_u": tk.StringVar(),
                "sim_d": tk.StringVar(),
                "act_u": tk.StringVar(),
                "act_d": tk.StringVar(),
            }
            self.manual_model_path = tk.StringVar()
            self.latent_dim = tk.IntVar(value=16)
            self.ae_epochs = tk.IntVar(value=200)
            self.pretrain_epochs = tk.IntVar(value=200)
            self.finetune_epochs = tk.IntVar(value=60)
            self.cls_epochs = tk.IntVar(value=150)
            self.batch_size = tk.IntVar(value=256)
            self.training_progress = tk.DoubleVar(value=0.0)
            self.training_status = tk.StringVar(value="第2步：请先配置预训练模型，再选择四类迁移学习数据。")
            self.active_model_text = tk.StringVar()
            self.pending_model_text = tk.StringVar()
            self.pending_manifest = read_pending_manifest(BASE_DIR)

            # 外部七模型组合：1个分类 + 3个无损回归 + 3个损伤回归。
            self.seven_model_rows = {
                slot: {
                    "model": tk.StringVar(),
                    "input_scaler": tk.StringVar(),
                    "output_scaler": tk.StringVar(),
                    "status": tk.StringVar(value="未选择"),
                }
                for slot in SEVEN_MODEL_SLOTS
            }
            self.external_damage_threshold = tk.DoubleVar(value=0.5)
            self.external_position_unit = tk.StringVar(value="m")
            self.external_damage_size_unit = tk.StringVar(value="mm")
            self.external_copy_into_software = tk.BooleanVar(value=True)
            self.external_status = tk.StringVar(value="请逐项选择七个模型；归一化文件可留空。")
            self.external_model_text = tk.StringVar()
            self.external_test_text = tk.StringVar(value="尚未执行七模型检查。")

            # 机翼三维模型管理状态；与迁移学习、后台识别模型完全独立。
            self.wing_geometry_path = tk.StringVar()
            self.wing_span_axis = tk.StringVar(value="X")
            self.wing_chord_axis = tk.StringVar(value="Y")
            self.wing_geometry_unit = tk.StringVar(value="mm")
            self.wing_reverse_span = tk.BooleanVar(value=False)
            self.wing_geometry_status = tk.StringVar(value="请选择机翼三维模型文件。")
            self.wing_geometry_info = tk.StringVar(value="当前未加载待保存模型。")
            self.wing_prepared_mesh = None
            self.wing_work_mesh = None
            self.wing_figure = None
            self.wing_canvas = None

            # 传感器布置编辑器状态；只使用“机翼三维模型”模块中的当前生效模型。
            self.selected_sensor_id = tk.IntVar(value=1)
            self.sensor_z_var = tk.DoubleVar(value=0.0)
            self.sensor_x_var = tk.DoubleVar(value=0.0)
            self.sensor_y_var = tk.DoubleVar(value=0.0)
            self.sensor_step_var = tk.DoubleVar(value=1.0)
            self.sensor_layout_status = tk.StringVar(value="请先在“机翼三维模型”页面设置当前机翼，再逐点调整48个传感器。")
            self.sensor_geometry_text = tk.StringVar(value="当前未配置机翼三维模型。")
            self.sensor_work_mesh = None
            self.sensor_coords_edit = np.asarray(self.sensor_coords_3d, dtype=float).copy()
            self.sensor_figure = None
            self.sensor_canvas = None
            self.sensor_ax_3d = None
            self.sensor_ax_plan = None

            self._build_style()
            self._build_ui()
            self._refresh_pretrain_model_card()
            self._refresh_active_model_card()
            self._refresh_pending_model_card()
            self._refresh_external_model_card()
            self.set_demo_data()
            self.run_analysis()
            self.root.after(120, self._poll_training_queue)

        def _rebuild_context(self) -> None:
            (
                self.geometry,
                self.sensor_coords,
                self.sensor_coords_3d,
                self.grid_z,
                self.grid_x,
                self.mask,
            ) = make_context(self.config)
            self.mesh_data = None
            mesh_source = self.active_geometry_manifest or self.active_layout_manifest
            if mesh_source and mesh_source.get("mesh_npz_path_resolved"):
                self.mesh_data = load_prepared_mesh(mesh_source.get("mesh_npz_path_resolved"))
            self.model_runner = ModelRunner(self.config)
            self.reconstructor = StrainReconstructor(self.config)

        def _reload_active_context(self) -> None:
            self.config, self.active_manifest, self.active_layout_manifest, self.active_geometry_manifest = runtime_config(self.base_config)
            self._rebuild_context()
            self.active_external_manifest = read_active_external_model(BASE_DIR)
            if self.active_external_manifest and self.active_external_manifest.get("model_path_resolved"):
                self.model_path.set(self.active_external_manifest["model_path_resolved"])
                self.model_mode.set("当前外部模型")
            elif self.active_manifest and self.active_manifest.get("bundle_path_resolved"):
                self.model_path.set(self.active_manifest["bundle_path_resolved"])
                self.model_mode.set("当前激活迁移模型")
            self._refresh_pretrain_model_card()
            self._refresh_active_model_card()
            self.pending_manifest = read_pending_manifest(BASE_DIR)
            self._refresh_pending_model_card()
            self._refresh_external_model_card()

        def _build_style(self) -> None:
            self.root.option_add("*Font", ("Microsoft YaHei UI", 10))
            style = ttk.Style(self.root)
            try:
                style.theme_use("vista")
            except tk.TclError:
                pass
            style.configure("TLabel", font=("Microsoft YaHei UI", 10))
            style.configure("TButton", font=("Microsoft YaHei UI", 10))
            style.configure("TEntry", font=("Microsoft YaHei UI", 10))
            style.configure("TCombobox", font=("Microsoft YaHei UI", 10))
            style.configure("TLabelframe.Label", font=("Microsoft YaHei UI", 10, "bold"))
            style.configure("Header.TLabel", font=("Microsoft YaHei UI", 15, "bold"))
            style.configure("SubHeader.TLabel", font=("Microsoft YaHei UI", 11, "bold"))
            style.configure("Result.TLabel", font=("Microsoft YaHei UI", 11, "bold"))
            style.configure("Run.TButton", font=("Microsoft YaHei UI", 11, "bold"), padding=8)
            style.configure("Update.TButton", font=("Microsoft YaHei UI", 11, "bold"), padding=10)
            style.configure("Train.TButton", font=("Microsoft YaHei UI", 11, "bold"), padding=10)

        def _build_ui(self) -> None:
            self.notebook = ttk.Notebook(self.root)
            self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
            self.monitor_tab = ttk.Frame(self.notebook)
            self.pretrain_tab = ttk.Frame(self.notebook)
            self.transfer_tab = ttk.Frame(self.notebook)
            self.external_model_tab = ttk.Frame(self.notebook)
            self.geometry_tab = ttk.Frame(self.notebook)
            self.sensor_tab = ttk.Frame(self.notebook)
            self.notebook.add(self.monitor_tab, text="  在线监测与应变云图  ")
            self.notebook.add(self.pretrain_tab, text="  第1步：模型预训练  ")
            self.notebook.add(self.transfer_tab, text="  第2步：迁移学习与更新  ")
            self.notebook.add(self.external_model_tab, text="  七模型组合导入  ")
            self.notebook.add(self.geometry_tab, text="  机翼三维模型  ")
            self.notebook.add(self.sensor_tab, text="  传感器布置  ")
            self._build_monitor_tab()
            self._build_pretrain_tab()
            self._build_transfer_tab()
            self._build_external_model_tab()
            self._build_geometry_tab()
            self._build_sensor_tab()

        def _build_monitor_tab(self) -> None:
            outer = ttk.Panedwindow(self.monitor_tab, orient=tk.HORIZONTAL)
            outer.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            controls = ttk.Frame(outer, width=380)
            display = ttk.Frame(outer)
            outer.add(controls, weight=0)
            outer.add(display, weight=1)

            ttk.Label(controls, text="机翼应变场监测与载荷识别", style="Header.TLabel").pack(
                anchor="w", padx=8, pady=(5, 10)
            )

            data_box = ttk.LabelFrame(controls, text="1. 48通道应变数据输入")
            data_box.pack(fill=tk.X, padx=6, pady=5)
            button_row = ttk.Frame(data_box)
            button_row.pack(fill=tk.X, padx=6, pady=5)
            ttk.Button(button_row, text="导入CSV", command=self.open_csv).pack(side=tk.LEFT, padx=2)
            ttk.Button(button_row, text="演示数据", command=self.set_demo_data).pack(side=tk.LEFT, padx=2)
            ttk.Button(button_row, text="清空", command=self.clear_data).pack(side=tk.LEFT, padx=2)
            self.strain_text = tk.Text(
                data_box,
                height=7,
                width=42,
                wrap=tk.WORD,
                font=("Microsoft YaHei UI", 9),
            )
            self.strain_text.pack(fill=tk.X, padx=6, pady=(0, 6))
            ttk.Label(
                data_box,
                text="支持逗号、空格、Tab或换行分隔；必须输入正好48个数值。",
                wraplength=340,
            ).pack(anchor="w", padx=6, pady=(0, 5))

            model_box = ttk.LabelFrame(controls, text="2. 载荷与损伤识别模型")
            model_box.pack(fill=tk.X, padx=6, pady=5)
            ttk.Combobox(
                model_box,
                textvariable=self.model_mode,
                state="readonly",
                values=list(model_mode_map.keys()),
            ).pack(fill=tk.X, padx=6, pady=5)
            path_row = ttk.Frame(model_box)
            path_row.pack(fill=tk.X, padx=6, pady=(0, 6))
            ttk.Entry(path_row, textvariable=self.model_path).pack(
                side=tk.LEFT, fill=tk.X, expand=True
            )
            ttk.Button(path_row, text="浏览", command=self.browse_model).pack(
                side=tk.LEFT, padx=(4, 0)
            )

            recon_box = ttk.LabelFrame(controls, text="3. 应变场重构方法")
            recon_box.pack(fill=tk.X, padx=6, pady=5)
            ttk.Combobox(
                recon_box,
                textvariable=self.reconstruction_mode,
                state="readonly",
                values=list(reconstruction_mode_map.keys()),
            ).pack(fill=tk.X, padx=6, pady=5)
            recon_path_row = ttk.Frame(recon_box)
            recon_path_row.pack(fill=tk.X, padx=6, pady=(0, 6))
            ttk.Entry(recon_path_row, textvariable=self.reconstruction_path).pack(
                side=tk.LEFT, fill=tk.X, expand=True
            )
            ttk.Button(
                recon_path_row, text="浏览", command=self.browse_reconstruction
            ).pack(side=tk.LEFT, padx=(4, 0))
            ttk.Label(
                recon_box,
                text="内置LC-Kriging直接使用当前48点手动坐标；仅自定义Python模式需要选择代码文件。",
                wraplength=340,
            ).pack(anchor="w", padx=6, pady=(0, 5))

            ttk.Button(
                controls,
                text="预测载荷并重构应变云图",
                style="Run.TButton",
                command=self.run_analysis,
            ).pack(fill=tk.X, padx=6, pady=(10, 6))
            ttk.Button(controls, text="保存组合结果图", command=self.save_figure).pack(
                fill=tk.X, padx=6, pady=(0, 8)
            )

            result_box = ttk.LabelFrame(controls, text="模型输出结果")
            result_box.pack(fill=tk.X, padx=6, pady=5)
            rows = [
                ("加载弦向坐标 x：", self.load_x_text),
                ("加载展向坐标 z：", self.load_z_text),
                ("载荷大小：", self.load_n_text),
                ("损伤概率：", self.damage_text),
                ("损伤弦向坐标：", self.damage_x_text),
                ("损伤展向坐标：", self.damage_z_text),
                ("损伤尺寸：", self.damage_size_text),
            ]
            for row_index, (label, variable) in enumerate(rows):
                ttk.Label(result_box, text=label).grid(
                    row=row_index, column=0, sticky="w", padx=6, pady=2
                )
                ttk.Label(result_box, textvariable=variable, style="Result.TLabel").grid(
                    row=row_index, column=1, sticky="e", padx=6, pady=2
                )
            result_box.columnconfigure(1, weight=1)

            ttk.Label(
                controls,
                textvariable=self.status_text,
                wraplength=350,
                relief=tk.GROOVE,
                padding=6,
            ).pack(fill=tk.X, padx=6, pady=(8, 5))

            canvas = FigureCanvasTkAgg(self.current_figure, master=display)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            toolbar = ChineseNavigationToolbar(canvas, display, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(fill=tk.X)
            self.canvas = canvas

        def _path_selector(self, parent, row: int, label: str, variable, command) -> None:
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=6)
            ttk.Entry(parent, textvariable=variable).grid(
                row=row, column=1, sticky="ew", padx=(0, 4), pady=6
            )
            ttk.Button(parent, text="选择文件", command=command).grid(
                row=row, column=2, sticky="ew", padx=(0, 8), pady=6
            )

        def _build_pretrain_tab(self) -> None:
            outer = ttk.Panedwindow(self.pretrain_tab, orient=tk.HORIZONTAL)
            outer.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
            left = ttk.Frame(outer, width=650)
            right = ttk.Frame(outer)
            outer.add(left, weight=1)
            outer.add(right, weight=1)

            ttk.Label(
                left, text="BP分类与CNN多任务回归预训练模型", style="Header.TLabel"
            ).pack(anchor="w", padx=8, pady=(5, 10))
            ttk.Label(
                left,
                text="模型组成：BP损伤分类器 + CNN无损载荷回归分支 + CNN损伤参数回归分支。",
                wraplength=610, justify=tk.LEFT,
            ).pack(fill=tk.X, padx=8, pady=(0, 8))

            data_box = ttk.LabelFrame(left, text="1. 仿真训练数据")
            data_box.pack(fill=tk.X, padx=8, pady=6)
            data_box.columnconfigure(1, weight=1)
            self._path_selector(
                data_box, 0, "仿真未损伤：", self.pretrain_paths["sim_u"],
                lambda: self.browse_pretrain_data("sim_u", "选择仿真未损伤数据"),
            )
            self._path_selector(
                data_box, 1, "仿真损伤：", self.pretrain_paths["sim_d"],
                lambda: self.browse_pretrain_data("sim_d", "选择仿真损伤数据"),
            )
            ttk.Label(
                data_box,
                text="未损伤标签：load_x、load_z/load_y、load_size；损伤标签：hole_x、hole_z/hole_y、hole_d。",
                wraplength=600, justify=tk.LEFT,
            ).grid(row=2, column=0, columnspan=3, sticky="w", padx=8, pady=(2, 8))

            param_box = ttk.LabelFrame(left, text="2. 预训练配置")
            param_box.pack(fill=tk.X, padx=8, pady=6)
            params = [
                ("BP搜索次数", self.pretrain_bp_trials),
                ("BP训练轮数", self.pretrain_bp_epochs),
                ("CNN训练轮数", self.pretrain_cnn_epochs),
                ("批大小", self.pretrain_batch_size),
            ]
            for index, (label, variable) in enumerate(params):
                row = index // 2
                column = (index % 2) * 2
                ttk.Label(param_box, text=label).grid(
                    row=row, column=column, sticky="w", padx=(8, 3), pady=7
                )
                ttk.Spinbox(
                    param_box, from_=1, to=2000, textvariable=variable, width=9
                ).grid(row=row, column=column + 1, sticky="w", padx=(0, 18), pady=7)

            action_box = ttk.LabelFrame(left, text="3. 预训练模型操作")
            action_box.pack(fill=tk.X, padx=8, pady=(8, 6))
            existing_row = ttk.Frame(action_box)
            existing_row.pack(fill=tk.X, padx=8, pady=(8, 4))
            ttk.Label(existing_row, text="已有预训练模型：").pack(side=tk.LEFT)
            ttk.Entry(existing_row, textvariable=self.pretrain_existing_path).pack(
                side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4)
            )
            ttk.Button(
                existing_row, text="选择", command=self.browse_existing_pretrain
            ).pack(side=tk.LEFT)

            action_row = ttk.Frame(action_box)
            action_row.pack(fill=tk.X, padx=8, pady=(4, 5))
            self.start_pretrain_button = ttk.Button(
                action_row, text="开始预训练", style="Train.TButton", command=self.start_pretraining
            )
            self.start_pretrain_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
            ttk.Button(
                action_row, text="使用已有预训练模型", style="Update.TButton",
                command=self.activate_existing_pretrain_model,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
            self.cancel_pretrain_button = ttk.Button(
                action_box, text="取消任务", command=self.cancel_training, state=tk.DISABLED
            )
            self.cancel_pretrain_button.pack(fill=tk.X, padx=8, pady=(0, 8))

            model_box = ttk.LabelFrame(right, text="当前预训练模型")
            model_box.pack(fill=tk.X, padx=8, pady=(8, 6))
            ttk.Label(
                model_box, textvariable=self.pretrain_model_text, wraplength=600, justify=tk.LEFT
            ).pack(fill=tk.X, padx=8, pady=8)

            progress_box = ttk.LabelFrame(right, text="预训练进度")
            progress_box.pack(fill=tk.X, padx=8, pady=6)
            ttk.Progressbar(
                progress_box, variable=self.pretrain_progress, maximum=100.0, mode="determinate"
            ).pack(fill=tk.X, padx=8, pady=(10, 5))
            ttk.Label(
                progress_box, textvariable=self.pretrain_status, wraplength=600, justify=tk.LEFT
            ).pack(fill=tk.X, padx=8, pady=(0, 10))

            log_box = ttk.LabelFrame(right, text="预训练日志")
            log_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
            self.pretrain_log = tk.Text(
                log_box, height=18, wrap=tk.WORD, state=tk.DISABLED,
                font=("Microsoft YaHei UI", 9),
            )
            scroll = ttk.Scrollbar(log_box, command=self.pretrain_log.yview)
            self.pretrain_log.configure(yscrollcommand=scroll.set)
            self.pretrain_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)
            scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)

        def _build_transfer_tab(self) -> None:
            outer = ttk.Panedwindow(self.transfer_tab, orient=tk.HORIZONTAL)
            outer.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
            left = ttk.Frame(outer, width=650)
            right = ttk.Frame(outer)
            outer.add(left, weight=1)
            outer.add(right, weight=1)

            ttk.Label(
                left,
                text="AE引导知识蒸馏迁移模型（Keras v3）",
                style="Header.TLabel",
            ).pack(anchor="w", padx=8, pady=(5, 6))
            ttk.Label(
                left,
                text="第2步：调用当前BP-CNN预训练模型作为教师模型，完成AE特征迁移、知识蒸馏和目标域微调。",
                wraplength=610, justify=tk.LEFT,
            ).pack(fill=tk.X, padx=8, pady=(0, 8))

            data_box = ttk.LabelFrame(left, text="1. 迁移学习数据")
            data_box.pack(fill=tk.X, padx=8, pady=6)
            data_box.columnconfigure(1, weight=1)
            self._path_selector(
                data_box, 0, "仿真未损伤：", self.transfer_paths["sim_u"],
                lambda: self.browse_transfer_data("sim_u", "选择仿真未损伤数据"),
            )
            self._path_selector(
                data_box, 1, "仿真损伤：", self.transfer_paths["sim_d"],
                lambda: self.browse_transfer_data("sim_d", "选择仿真损伤数据"),
            )
            self._path_selector(
                data_box, 2, "实际未损伤：", self.transfer_paths["act_u"],
                lambda: self.browse_transfer_data("act_u", "选择实际未损伤数据"),
            )
            self._path_selector(
                data_box, 3, "实际损伤：", self.transfer_paths["act_d"],
                lambda: self.browse_transfer_data("act_d", "选择实际损伤数据"),
            )
            button_row = ttk.Frame(data_box)
            button_row.grid(row=4, column=0, columnspan=3, sticky="ew", padx=8, pady=(2, 8))
            ttk.Button(
                button_row, text="载入示例数据", command=self.load_transfer_demo_paths
            ).pack(side=tk.LEFT)
            ttk.Label(button_row, text="数据格式：CSV / TXT / XLSX").pack(side=tk.LEFT, padx=10)

            param_box = ttk.LabelFrame(left, text="2. 训练配置")
            param_box.pack(fill=tk.X, padx=8, pady=6)
            params = [
                ("潜在维度", self.latent_dim),
                ("AE轮数", self.ae_epochs),
                ("回归轮数", self.pretrain_epochs),
                ("分类轮数", self.cls_epochs),
                ("微调轮数", self.finetune_epochs),
                ("批大小", self.batch_size),
            ]
            for index, (label, variable) in enumerate(params):
                row = index // 3
                column = (index % 3) * 2
                ttk.Label(param_box, text=label).grid(
                    row=row, column=column, sticky="w", padx=(8, 3), pady=7
                )
                ttk.Spinbox(
                    param_box, from_=1, to=2000, textvariable=variable, width=8
                ).grid(row=row, column=column + 1, sticky="w", padx=(0, 14), pady=7)

            action_box = ttk.LabelFrame(left, text="3. 模型操作")
            action_box.pack(fill=tk.X, padx=8, pady=(8, 6))
            model_path_row = ttk.Frame(action_box)
            model_path_row.pack(fill=tk.X, padx=8, pady=(8, 4))
            ttk.Label(model_path_row, text="已有模型目录：").pack(side=tk.LEFT)
            ttk.Entry(model_path_row, textvariable=self.manual_model_path).pack(
                side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4)
            )
            ttk.Button(
                model_path_row, text="选择", command=self.browse_existing_model
            ).pack(side=tk.LEFT)

            action_row = ttk.Frame(action_box)
            action_row.pack(fill=tk.X, padx=8, pady=(4, 5))
            self.start_training_button = ttk.Button(
                action_row, text="开始迁移学习", style="Train.TButton", command=self.start_training
            )
            self.start_training_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
            self.update_model_button = ttk.Button(
                action_row, text="更新后台模型", style="Update.TButton",
                command=self.update_backend_model, state=tk.NORMAL
            )
            self.update_model_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
            self.cancel_training_button = ttk.Button(
                action_box, text="取消训练", command=self.cancel_training, state=tk.DISABLED
            )
            self.cancel_training_button.pack(fill=tk.X, padx=8, pady=(0, 5))
            ttk.Label(
                action_box,
                text="后台模型更新仅切换识别模型文件；机翼三维模型和传感器布置由各自独立页面管理。",
                wraplength=610,
                justify=tk.LEFT,
            ).pack(fill=tk.X, padx=8, pady=(0, 8))

            teacher_box = ttk.LabelFrame(right, text="当前预训练教师模型")
            teacher_box.pack(fill=tk.X, padx=8, pady=(8, 6))
            ttk.Label(
                teacher_box, textvariable=self.pretrain_model_text, wraplength=600, justify=tk.LEFT
            ).pack(fill=tk.X, padx=8, pady=8)

            active_box = ttk.LabelFrame(right, text="当前生效迁移模型")
            active_box.pack(fill=tk.X, padx=8, pady=6)
            ttk.Label(
                active_box, textvariable=self.active_model_text, wraplength=600, justify=tk.LEFT
            ).pack(fill=tk.X, padx=8, pady=8)

            pending_box = ttk.LabelFrame(right, text="待更新模型")
            pending_box.pack(fill=tk.X, padx=8, pady=6)
            ttk.Label(
                pending_box, textvariable=self.pending_model_text, wraplength=600, justify=tk.LEFT
            ).pack(fill=tk.X, padx=8, pady=8)

            progress_box = ttk.LabelFrame(right, text="任务进度")
            progress_box.pack(fill=tk.X, padx=8, pady=6)
            self.progress_bar = ttk.Progressbar(
                progress_box, variable=self.training_progress, maximum=100.0, mode="determinate"
            )
            self.progress_bar.pack(fill=tk.X, padx=8, pady=(10, 5))
            ttk.Label(
                progress_box, textvariable=self.training_status, wraplength=600, justify=tk.LEFT
            ).pack(fill=tk.X, padx=8, pady=(0, 10))

            log_box = ttk.LabelFrame(right, text="运行日志")
            log_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
            self.training_log = tk.Text(
                log_box, height=18, wrap=tk.WORD, state=tk.DISABLED,
                font=("Microsoft YaHei UI", 9),
            )
            scroll = ttk.Scrollbar(log_box, command=self.training_log.yview)
            self.training_log.configure(yscrollcommand=scroll.set)
            self.training_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)
            scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)

        def _build_external_model_tab(self) -> None:
            outer = ttk.Panedwindow(self.external_model_tab, orient=tk.HORIZONTAL)
            outer.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
            left = ttk.Frame(outer, width=1050)
            right = ttk.Frame(outer, width=430)
            outer.add(left, weight=3)
            outer.add(right, weight=1)

            ttk.Label(left, text="七模型组合导入", style="Header.TLabel").pack(
                anchor="w", padx=8, pady=(5, 3)
            )
            ttk.Label(
                left,
                text="模型组成：1个BP损伤分类模型、3个无损载荷回归模型、3个有损损伤回归模型。每个模型的输入/输出归一化文件均为可选项，留空表示模型直接使用原始48通道输入或直接输出物理量。",
                wraplength=1000,
                justify=tk.LEFT,
            ).pack(fill=tk.X, padx=8, pady=(0, 7))

            toolbar = ttk.Frame(left)
            toolbar.pack(fill=tk.X, padx=8, pady=(0, 5))
            ttk.Button(
                toolbar, text="选择模型目录并自动匹配", command=self.auto_match_seven_models
            ).pack(side=tk.LEFT)
            ttk.Button(
                toolbar, text="清空全部", command=self.clear_seven_model_form
            ).pack(side=tk.LEFT, padx=(6, 0))
            ttk.Label(
                toolbar,
                text="支持：.keras / .h5 / .joblib / .pkl / .pt / .pth / .py",
            ).pack(side=tk.LEFT, padx=(15, 0))

            table = ttk.LabelFrame(left, text="1. 七个模型及归一化文件")
            table.pack(fill=tk.X, padx=8, pady=5)
            table.columnconfigure(1, weight=3)
            table.columnconfigure(3, weight=2)
            table.columnconfigure(5, weight=2)

            headers = [
                ("模型功能", 0),
                ("模型文件", 1),
                ("", 2),
                ("输入归一化（可选）", 3),
                ("", 4),
                ("输出归一化（可选）", 5),
                ("", 6),
                ("状态", 7),
            ]
            for text, column in headers:
                ttk.Label(table, text=text, style="SubHeader.TLabel").grid(
                    row=0, column=column, sticky="w", padx=4, pady=(7, 5)
                )

            for row_index, (slot, definition) in enumerate(SEVEN_MODEL_SLOTS.items(), start=1):
                variables = self.seven_model_rows[slot]
                ttk.Label(table, text=definition["label"]).grid(
                    row=row_index, column=0, sticky="w", padx=(7, 4), pady=4
                )
                ttk.Entry(table, textvariable=variables["model"]).grid(
                    row=row_index, column=1, sticky="ew", padx=(0, 3), pady=4
                )
                ttk.Button(
                    table,
                    text="选择",
                    width=5,
                    command=lambda s=slot: self.browse_seven_model_file(s, "model"),
                ).grid(row=row_index, column=2, padx=(0, 6), pady=4)
                ttk.Entry(table, textvariable=variables["input_scaler"]).grid(
                    row=row_index, column=3, sticky="ew", padx=(0, 3), pady=4
                )
                ttk.Button(
                    table,
                    text="选择",
                    width=5,
                    command=lambda s=slot: self.browse_seven_model_file(s, "input_scaler"),
                ).grid(row=row_index, column=4, padx=(0, 6), pady=4)

                if slot == "classifier":
                    ttk.Label(table, text="不需要", foreground="#666666").grid(
                        row=row_index, column=5, sticky="w", padx=4, pady=4
                    )
                    ttk.Label(table, text="").grid(row=row_index, column=6)
                else:
                    ttk.Entry(table, textvariable=variables["output_scaler"]).grid(
                        row=row_index, column=5, sticky="ew", padx=(0, 3), pady=4
                    )
                    ttk.Button(
                        table,
                        text="选择",
                        width=5,
                        command=lambda s=slot: self.browse_seven_model_file(s, "output_scaler"),
                    ).grid(row=row_index, column=6, padx=(0, 6), pady=4)
                ttk.Label(table, textvariable=variables["status"]).grid(
                    row=row_index, column=7, sticky="w", padx=(2, 7), pady=4
                )

            ttk.Label(
                table,
                text="说明：输入归一化文件应具有 transform 方法；回归输出归一化文件应具有 inverse_transform 方法。模型内部已包含归一化时，对应位置直接留空。",
                wraplength=980,
                justify=tk.LEFT,
            ).grid(row=8, column=0, columnspan=8, sticky="w", padx=7, pady=(5, 8))

            settings = ttk.LabelFrame(left, text="2. 判定与单位")
            settings.pack(fill=tk.X, padx=8, pady=5)
            ttk.Label(settings, text="损伤判定阈值：").grid(row=0, column=0, padx=(8, 4), pady=7, sticky="w")
            ttk.Spinbox(
                settings, from_=0.01, to=0.99, increment=0.01,
                textvariable=self.external_damage_threshold, width=8,
            ).grid(row=0, column=1, padx=(0, 16), pady=7, sticky="w")
            ttk.Label(settings, text="x/z模型输出单位：").grid(row=0, column=2, padx=(0, 4), pady=7, sticky="w")
            ttk.Combobox(
                settings, textvariable=self.external_position_unit,
                values=["m", "mm"], state="readonly", width=7,
            ).grid(row=0, column=3, padx=(0, 16), pady=7, sticky="w")
            ttk.Label(settings, text="损伤尺寸输出单位：").grid(row=0, column=4, padx=(0, 4), pady=7, sticky="w")
            ttk.Combobox(
                settings, textvariable=self.external_damage_size_unit,
                values=["mm", "m"], state="readonly", width=7,
            ).grid(row=0, column=5, padx=(0, 16), pady=7, sticky="w")
            ttk.Checkbutton(
                settings,
                text="将模型文件复制到软件目录",
                variable=self.external_copy_into_software,
            ).grid(row=0, column=6, padx=(4, 8), pady=7, sticky="w")

            actions = ttk.LabelFrame(left, text="3. 模型操作")
            actions.pack(fill=tk.X, padx=8, pady=5)
            ttk.Button(
                actions, text="检查七个模型", command=self.check_external_model
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 4), pady=9)
            ttk.Button(
                actions, text="导入并应用", command=self.import_and_apply_external_model
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4, pady=9)
            ttk.Button(
                actions, text="取消当前外部模型", command=self.clear_external_model
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 8), pady=9)
            ttk.Label(
                left, textvariable=self.external_status, wraplength=1000, justify=tk.LEFT
            ).pack(fill=tk.X, padx=10, pady=(2, 8))

            active_box = ttk.LabelFrame(right, text="当前在线模型")
            active_box.pack(fill=tk.X, padx=8, pady=6)
            ttk.Label(
                active_box, textvariable=self.external_model_text,
                wraplength=395, justify=tk.LEFT,
            ).pack(fill=tk.X, padx=8, pady=9)

            test_box = ttk.LabelFrame(right, text="七模型检查结果")
            test_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
            ttk.Label(
                test_box, textvariable=self.external_test_text,
                wraplength=395, justify=tk.LEFT,
            ).pack(fill=tk.X, padx=8, pady=9)

            guide_box = ttk.LabelFrame(right, text="输出对应关系")
            guide_box.pack(fill=tk.X, padx=8, pady=6)
            guide_text = (
                "分类模型 → 损伤概率\n\n"
                "判定为无损：\n"
                "• 载荷位置 x\n• 载荷位置 z\n• 载荷大小\n\n"
                "判定为有损：\n"
                "• 损伤位置 x\n• 损伤位置 z\n• 损伤尺寸\n\n"
                "模型编号和48个传感器通道顺序必须与训练时完全一致。"
            )
            ttk.Label(
                guide_box, text=guide_text, wraplength=395, justify=tk.LEFT,
            ).pack(fill=tk.X, padx=8, pady=9)

        def _build_geometry_tab(self) -> None:
            from matplotlib.figure import Figure

            outer = ttk.Panedwindow(self.geometry_tab, orient=tk.HORIZONTAL)
            outer.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
            controls = ttk.Frame(outer, width=520)
            display = ttk.Frame(outer)
            outer.add(controls, weight=0)
            outer.add(display, weight=1)

            ttk.Label(
                controls, text="机翼三维模型管理", style="Header.TLabel"
            ).pack(anchor="w", padx=8, pady=(5, 10))
            ttk.Label(
                controls,
                text="该模块仅管理机翼几何，不参与迁移学习，也不会修改后台识别模型或48点传感器坐标。",
                wraplength=490, justify=tk.LEFT,
            ).pack(fill=tk.X, padx=8, pady=(0, 8))

            import_box = ttk.LabelFrame(controls, text="1. 模型文件")
            import_box.pack(fill=tk.X, padx=8, pady=5)
            row = ttk.Frame(import_box)
            row.pack(fill=tk.X, padx=8, pady=(8, 5))
            ttk.Entry(row, textvariable=self.wing_geometry_path).pack(
                side=tk.LEFT, fill=tk.X, expand=True
            )
            ttk.Button(row, text="选择模型", command=self.browse_wing_geometry).pack(
                side=tk.LEFT, padx=(4, 0)
            )
            ttk.Label(
                import_box, text="内置读取：STL（含二进制/ASCII）、OBJ、OFF、ASCII PLY；无需安装 trimesh。",
                wraplength=470,
            ).pack(anchor="w", padx=8, pady=(0, 8))

            coord_box = ttk.LabelFrame(controls, text="2. 坐标与单位")
            coord_box.pack(fill=tk.X, padx=8, pady=5)
            settings = ttk.Frame(coord_box)
            settings.pack(fill=tk.X, padx=8, pady=8)
            for label, variable, values in (
                ("展向轴", self.wing_span_axis, ["X", "Y", "Z"]),
                ("弦向轴", self.wing_chord_axis, ["X", "Y", "Z"]),
                ("模型单位", self.wing_geometry_unit, ["mm", "cm", "m"]),
            ):
                ttk.Label(settings, text=label).pack(side=tk.LEFT, padx=(0, 3))
                ttk.Combobox(
                    settings, textvariable=variable, values=values,
                    state="readonly", width=5
                ).pack(side=tk.LEFT, padx=(0, 10))
            ttk.Checkbutton(
                coord_box, text="反转展向坐标", variable=self.wing_reverse_span
            ).pack(anchor="w", padx=8, pady=(0, 8))

            action_box = ttk.LabelFrame(controls, text="3. 模型操作")
            action_box.pack(fill=tk.X, padx=8, pady=5)
            ttk.Button(
                action_box, text="加载并预览", command=self.load_wing_geometry_preview
            ).pack(fill=tk.X, padx=8, pady=(8, 4))
            ttk.Button(
                action_box, text="保存并设为当前机翼", style="Update.TButton",
                command=self.save_wing_geometry
            ).pack(fill=tk.X, padx=8, pady=(4, 8))

            info_box = ttk.LabelFrame(controls, text="当前模型信息")
            info_box.pack(fill=tk.X, padx=8, pady=5)
            ttk.Label(
                info_box, textvariable=self.wing_geometry_info, wraplength=480,
                justify=tk.LEFT
            ).pack(fill=tk.X, padx=8, pady=8)
            ttk.Label(
                controls, textvariable=self.wing_geometry_status, wraplength=490,
                relief=tk.GROOVE, padding=6
            ).pack(fill=tk.X, padx=8, pady=(5, 8))

            self.wing_figure = Figure(figsize=(10.5, 7.2), dpi=100, constrained_layout=True)
            self.wing_canvas = FigureCanvasTkAgg(self.wing_figure, master=display)
            self.wing_canvas.draw()
            self.wing_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            toolbar = ChineseNavigationToolbar(self.wing_canvas, display, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(fill=tk.X)
            self.load_active_wing_geometry_into_editor()
            self.draw_wing_geometry_preview()

        def _build_sensor_tab(self) -> None:
            from matplotlib.figure import Figure

            outer = ttk.Panedwindow(self.sensor_tab, orient=tk.HORIZONTAL)
            outer.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
            controls = ttk.Frame(outer, width=560)
            display = ttk.Frame(outer)
            outer.add(controls, weight=0)
            outer.add(display, weight=1)

            ttk.Label(
                controls, text="48通道传感器坐标布置", style="Header.TLabel"
            ).pack(anchor="w", padx=8, pady=(5, 10))

            mesh_box = ttk.LabelFrame(controls, text="1. 当前机翼三维模型")
            mesh_box.pack(fill=tk.X, padx=8, pady=5)
            ttk.Label(
                mesh_box, textvariable=self.sensor_geometry_text, wraplength=520, justify=tk.LEFT
            ).pack(fill=tk.X, padx=8, pady=(8, 5))
            model_actions = ttk.Frame(mesh_box)
            model_actions.pack(fill=tk.X, padx=8, pady=(2, 8))
            ttk.Button(
                model_actions, text="载入当前机翼", command=self.load_active_geometry_into_sensor_editor
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
            ttk.Button(
                model_actions, text="打开机翼模型页面", command=lambda: self.notebook.select(self.geometry_tab)
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

            table_box = ttk.LabelFrame(controls, text="2. 传感器坐标表")
            table_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=5)
            columns = ("id", "z", "x", "y")
            self.sensor_tree = ttk.Treeview(
                table_box, columns=columns, show="headings", height=13, selectmode="browse"
            )
            headings = {
                "id": "编号", "z": "展向 z / mm", "x": "弦向 x / mm", "y": "厚度 y / mm"
            }
            widths = {"id": 55, "z": 110, "x": 110, "y": 110}
            for key in columns:
                self.sensor_tree.heading(key, text=headings[key])
                self.sensor_tree.column(key, width=widths[key], anchor="center")
            tree_scroll = ttk.Scrollbar(table_box, command=self.sensor_tree.yview)
            self.sensor_tree.configure(yscrollcommand=tree_scroll.set)
            self.sensor_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)
            tree_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)
            self.sensor_tree.bind("<<TreeviewSelect>>", self.on_sensor_selected)

            edit_box = ttk.LabelFrame(controls, text="3. 当前传感器精确调整")
            edit_box.pack(fill=tk.X, padx=8, pady=5)
            top_row = ttk.Frame(edit_box)
            top_row.pack(fill=tk.X, padx=8, pady=(8, 4))
            ttk.Label(top_row, text="传感器编号").pack(side=tk.LEFT)
            sensor_selector = ttk.Combobox(
                top_row, textvariable=self.selected_sensor_id,
                values=list(range(1, 49)), state="readonly", width=6
            )
            sensor_selector.pack(side=tk.LEFT, padx=(4, 12))
            sensor_selector.bind("<<ComboboxSelected>>", self.on_sensor_id_changed)
            ttk.Label(top_row, text="步长 / mm").pack(side=tk.LEFT)
            ttk.Spinbox(
                top_row, from_=0.001, to=1000.0, increment=0.1,
                textvariable=self.sensor_step_var, width=9
            ).pack(side=tk.LEFT, padx=4)

            coord_row = ttk.Frame(edit_box)
            coord_row.pack(fill=tk.X, padx=8, pady=4)
            for label, variable in (
                ("z", self.sensor_z_var), ("x", self.sensor_x_var), ("y", self.sensor_y_var)
            ):
                ttk.Label(coord_row, text=f"{label} / mm").pack(side=tk.LEFT, padx=(0, 3))
                ttk.Entry(coord_row, textvariable=variable, width=11).pack(side=tk.LEFT, padx=(0, 10))
            ttk.Button(coord_row, text="应用坐标", command=self.apply_sensor_coordinate).pack(side=tk.LEFT)

            nudge_grid = ttk.Frame(edit_box)
            nudge_grid.pack(fill=tk.X, padx=8, pady=4)
            for axis_index, axis_name in enumerate(("z", "x", "y")):
                ttk.Label(nudge_grid, text=f"{axis_name}轴").grid(
                    row=0, column=axis_index * 3, padx=(0, 3), pady=2
                )
                ttk.Button(
                    nudge_grid, text="−", width=4,
                    command=lambda a=axis_index: self.nudge_sensor(a, -1.0)
                ).grid(row=0, column=axis_index * 3 + 1, padx=2, pady=2)
                ttk.Button(
                    nudge_grid, text="+", width=4,
                    command=lambda a=axis_index: self.nudge_sensor(a, 1.0)
                ).grid(row=0, column=axis_index * 3 + 2, padx=(2, 12), pady=2)
            ttk.Button(
                edit_box, text="吸附到机翼上表面", command=self.snap_selected_sensor
            ).pack(fill=tk.X, padx=8, pady=(4, 8))

            save_box = ttk.LabelFrame(controls, text="4. 布置文件")
            save_box.pack(fill=tk.X, padx=8, pady=5)
            file_row = ttk.Frame(save_box)
            file_row.pack(fill=tk.X, padx=8, pady=(8, 4))
            ttk.Button(file_row, text="导入坐标", command=self.import_sensor_layout).pack(
                side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3)
            )
            ttk.Button(file_row, text="导出CSV", command=self.export_sensor_layout).pack(
                side=tk.LEFT, fill=tk.X, expand=True, padx=3
            )
            ttk.Button(file_row, text="恢复规则布置", command=self.generate_default_sensor_layout).pack(
                side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0)
            )
            ttk.Button(
                save_box, text="保存并应用当前布置", style="Update.TButton",
                command=self.save_sensor_layout
            ).pack(fill=tk.X, padx=8, pady=(4, 8))
            ttk.Label(
                controls, textvariable=self.sensor_layout_status, wraplength=530,
                relief=tk.GROOVE, padding=6
            ).pack(fill=tk.X, padx=8, pady=(5, 8))

            self.sensor_figure = Figure(figsize=(10.5, 7.2), dpi=100, constrained_layout=True)
            self.sensor_canvas = FigureCanvasTkAgg(self.sensor_figure, master=display)
            self.sensor_canvas.draw()
            self.sensor_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            toolbar = ChineseNavigationToolbar(self.sensor_canvas, display, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(fill=tk.X)
            self.sensor_canvas.mpl_connect("button_press_event", self.on_sensor_canvas_click)

            self.refresh_sensor_table()
            self.load_active_layout_into_editor()
            self.draw_sensor_editor()

        def browse_seven_model_file(self, slot: str, kind: str) -> None:
            if kind == "model":
                path = filedialog.askopenfilename(
                    title=f"选择{SEVEN_MODEL_SLOTS[slot]['label']}模型",
                    filetypes=[
                        ("支持的模型", "*.keras *.h5 *.hdf5 *.joblib *.pkl *.pickle *.save *.pt *.pth *.ts *.torchscript *.py"),
                        ("Keras模型", "*.keras *.h5 *.hdf5"),
                        ("Joblib模型", "*.joblib *.pkl *.pickle *.save"),
                        ("TorchScript模型", "*.pt *.pth *.ts *.torchscript"),
                        ("Python适配器", "*.py"),
                        ("所有文件", "*.*"),
                    ],
                )
                target = "model"
            else:
                title = "输入归一化文件" if kind == "input_scaler" else "输出归一化文件"
                path = filedialog.askopenfilename(
                    title=f"选择{SEVEN_MODEL_SLOTS[slot]['label']}的{title}",
                    filetypes=[
                        ("归一化文件", "*.pkl *.joblib *.pickle *.save"),
                        ("所有文件", "*.*"),
                    ],
                )
                target = kind
            if path:
                self.seven_model_rows[slot][target].set(path)
                self.seven_model_rows[slot]["status"].set("已选择")
                self.external_status.set("文件已选择。归一化文件不是必需项，全部模型选择完成后请执行检查。")

        def auto_match_seven_models(self) -> None:
            directory = filedialog.askdirectory(
                title="选择包含七个模型及归一化文件的目录",
                initialdir=str(BASE_DIR),
            )
            if not directory:
                return
            try:
                matched = auto_match_seven_model_directory(directory)
                matched_count = 0
                for slot, values in matched.items():
                    row = self.seven_model_rows[slot]
                    row["model"].set(values.get("model_path") or "")
                    row["input_scaler"].set(values.get("input_scaler_path") or "")
                    row["output_scaler"].set(values.get("output_scaler_path") or "")
                    if values.get("model_path"):
                        row["status"].set("已自动匹配")
                        matched_count += 1
                    else:
                        row["status"].set("需手动选择")
                self.external_status.set(
                    f"自动匹配完成：已找到{matched_count}/7个模型。请逐行核对，未匹配或匹配错误的项目可手动更换。"
                )
            except Exception as exc:
                messagebox.showerror("自动匹配失败", str(exc))

        def clear_seven_model_form(self) -> None:
            for row in self.seven_model_rows.values():
                row["model"].set("")
                row["input_scaler"].set("")
                row["output_scaler"].set("")
                row["status"].set("未选择")
            self.external_test_text.set("尚未执行七模型检查。")
            self.external_status.set("已清空。请逐项选择七个模型；归一化文件可留空。")

        def _collect_seven_model_slots(self) -> dict[str, dict[str, Any]]:
            position_multiplier = 1000.0 if self.external_position_unit.get() == "m" else 1.0
            damage_size_multiplier = 1000.0 if self.external_damage_size_unit.get() == "m" else 1.0
            slots: dict[str, dict[str, Any]] = {}
            for slot in SEVEN_MODEL_SLOTS:
                row = self.seven_model_rows[slot]
                multiplier = 1.0
                if slot in {"intact_load_x", "intact_load_z", "damage_x", "damage_z"}:
                    multiplier = position_multiplier
                elif slot == "damage_size":
                    multiplier = damage_size_multiplier
                slots[slot] = {
                    "model_path": row["model"].get().strip() or None,
                    "input_scaler_path": row["input_scaler"].get().strip() or None,
                    "output_scaler_path": (
                        None if slot == "classifier" else row["output_scaler"].get().strip() or None
                    ),
                    "multiplier": multiplier,
                }
            return slots

        def _external_test_values(self) -> np.ndarray:
            try:
                return self.parse_strains()
            except Exception:
                return generate_demo_strains(self.sensor_coords, self.geometry)

        @staticmethod
        def _format_external_prediction(prediction: dict[str, Any]) -> str:
            probability = prediction.get("damage_probability")
            branch = prediction.get("prediction_branch")
            lines = ["七个模型均已成功加载并完成一次48通道测试。"]
            if probability is not None:
                lines.append(f"分类模型输出：损伤概率 {float(probability):.2%}")
            lines.append(f"当前测试判定：{'有损' if branch == 'damaged' else '无损'}")
            lines.extend([
                "",
                "无损分支测试输出：",
                f"  载荷位置 x：{float(prediction['load_x_mm']):.6g} mm",
                f"  载荷位置 z：{float(prediction['load_z_mm']):.6g} mm",
                f"  载荷大小：{float(prediction['load_n']):.6g} N",
                "",
                "有损分支测试输出：",
                f"  损伤位置 x：{float(prediction['damage_x_mm']):.6g} mm",
                f"  损伤位置 z：{float(prediction['damage_z_mm']):.6g} mm",
                f"  损伤尺寸：{float(prediction['damage_size_mm']):.6g} mm",
            ])
            return "\n".join(lines)

        def _mark_seven_model_status(self, text: str) -> None:
            for row in self.seven_model_rows.values():
                row["status"].set(text)

        def check_external_model(self) -> None:
            try:
                self._mark_seven_model_status("检查中")
                self.external_status.set("正在依次检查1个分类模型和6个回归模型……")
                self.root.update_idletasks()
                prediction = inspect_seven_model_group(
                    slots=self._collect_seven_model_slots(),
                    strain_values=self._external_test_values(),
                    config=self.config,
                    damage_threshold=float(self.external_damage_threshold.get()),
                )
                self._mark_seven_model_status("检查通过")
                self.external_test_text.set(self._format_external_prediction(prediction))
                self.external_status.set("七个模型检查通过，可以导入并应用。")
                messagebox.showinfo(
                    "检查通过",
                    "七个模型均可正常加载，48通道输入、可选归一化文件和单值输出接口均检查通过。",
                )
            except Exception as exc:
                for slot, row in self.seven_model_rows.items():
                    if not row["model"].get().strip():
                        row["status"].set("缺少模型")
                    elif row["status"].get() == "检查中":
                        row["status"].set("待排查")
                self.external_test_text.set("七模型检查失败：\n" + str(exc))
                self.external_status.set("检查失败。请根据错误信息核对相应模型和归一化文件。")
                messagebox.showerror("七模型检查失败", str(exc))

        def import_and_apply_external_model(self) -> None:
            try:
                slots = self._collect_seven_model_slots()
                self.external_status.set("正在检查七个模型……")
                prediction = inspect_seven_model_group(
                    slots=slots,
                    strain_values=self._external_test_values(),
                    config=self.config,
                    damage_threshold=float(self.external_damage_threshold.get()),
                )
                self._mark_seven_model_status("检查通过")
                confirmed = messagebox.askyesno(
                    "确认应用七模型组合",
                    "七个模型均已检查通过。是否设为在线监测当前模型？\n\n"
                    "分类后将自动调用无损三模型或有损三模型。\n"
                    "本操作不会修改机翼三维模型和48点传感器坐标。",
                )
                if not confirmed:
                    self.external_status.set("已取消应用，七个模型文件保持不变。")
                    return
                manifest_path = install_seven_model_group(
                    BASE_DIR,
                    slots=slots,
                    damage_threshold=float(self.external_damage_threshold.get()),
                    copy_into_software=bool(self.external_copy_into_software.get()),
                )
                self.active_external_manifest = read_active_external_model(BASE_DIR)
                self.model_mode.set("当前外部模型")
                self.model_path.set(
                    self.active_external_manifest.get("model_path_resolved", str(manifest_path))
                )
                self._refresh_external_model_card()
                self.external_test_text.set(self._format_external_prediction(prediction))
                self.external_status.set("七模型组合已导入并应用到在线监测。")
                self.run_analysis()
                messagebox.showinfo(
                    "应用完成",
                    "七模型组合已设为当前在线模型。\n\n"
                    "1个分类模型负责判定损伤状态；软件随后自动调用对应的3个回归模型。",
                )
            except Exception as exc:
                self.external_status.set("七模型组合导入失败。")
                messagebox.showerror("导入失败", str(exc))

        def clear_external_model(self) -> None:
            if not read_active_external_model(BASE_DIR):
                messagebox.showinfo("没有外部模型", "当前没有已应用的外部模型。")
                return
            confirmed = messagebox.askyesno(
                "取消当前外部模型",
                "确定取消当前七模型组合吗？\n\n已复制到软件目录的模型文件不会自动删除。",
            )
            if not confirmed:
                return
            clear_active_external_model(BASE_DIR)
            self.active_external_manifest = None
            if self.active_manifest and self.active_manifest.get("bundle_path_resolved"):
                self.model_mode.set("当前激活迁移模型")
                self.model_path.set(self.active_manifest["bundle_path_resolved"])
            else:
                self.model_mode.set("演示模型")
                self.model_path.set(str(BASE_DIR / "adapters" / "user_model.py"))
            self._refresh_external_model_card()
            self.external_status.set("当前七模型组合已取消。")
            self.run_analysis()

        def browse_existing_model(self) -> None:
            path = filedialog.askdirectory(
                title="选择迁移模型目录",
                initialdir=str(BASE_DIR / "models"),
            )
            if path:
                self.manual_model_path.set(path)

        def browse_wing_geometry(self) -> None:
            path = filedialog.askopenfilename(
                title="选择机翼三维模型",
                filetypes=[
                    ("内置支持的三维网格", "*.stl *.obj *.off *.ply"),
                    ("可选格式", "*.glb *.gltf"),
                    ("所有文件", "*.*"),
                ],
            )
            if path:
                self.wing_geometry_path.set(path)

        def load_wing_geometry_preview(self) -> None:
            import shutil

            path = self.wing_geometry_path.get().strip()
            if not path:
                messagebox.showerror("未选择模型", "请先选择机翼三维模型文件。")
                return
            if self.wing_span_axis.get() == self.wing_chord_axis.get():
                messagebox.showerror("坐标轴设置错误", "展向轴和弦向轴不能相同。")
                return
            try:
                work_dir = BASE_DIR / "wing_geometry" / "_work"
                if work_dir.exists():
                    shutil.rmtree(work_dir)
                self.wing_prepared_mesh = prepare_mesh(
                    path,
                    work_dir,
                    span_axis=self.wing_span_axis.get(),
                    chord_axis=self.wing_chord_axis.get(),
                    unit=self.wing_geometry_unit.get(),
                    reverse_span=self.wing_reverse_span.get(),
                    max_faces=20000,
                )
                self.wing_work_mesh = load_prepared_mesh(self.wing_prepared_mesh.npz_path)
                meta = self.wing_prepared_mesh.metadata
                self.wing_geometry_info.set(
                    f"文件：{Path(path).name}\n"
                    f"展向长度：{meta['span_mm']:.3f} mm\n"
                    f"根弦长度：{meta['root_chord_mm']:.3f} mm\n"
                    f"尖弦长度：{meta['tip_chord_mm']:.3f} mm\n"
                    f"最大厚度：{meta['thickness_mm']:.3f} mm\n"
                    f"显示面片：{meta['face_count_display']:,}"
                )
                self.wing_geometry_status.set("模型加载完成。确认坐标方向后，可保存为当前机翼。")
                self.draw_wing_geometry_preview()
            except Exception as exc:
                messagebox.showerror("三维模型读取失败", str(exc))

        def save_wing_geometry(self) -> None:
            if self.wing_prepared_mesh is None:
                answer = messagebox.askyesno(
                    "尚未加载模型",
                    "当前没有待保存的三维模型。是否先按当前参数加载所选模型？",
                )
                if not answer:
                    return
                self.load_wing_geometry_preview()
                if self.wing_prepared_mesh is None:
                    return
            try:
                save_active_wing_geometry(BASE_DIR, self.wing_prepared_mesh)
                self._reload_active_context()
                self.load_active_wing_geometry_into_editor()
                self.load_active_geometry_into_sensor_editor(preserve_coordinates=True)
                self.set_demo_data()
                self.run_analysis()
                self.wing_geometry_status.set("当前机翼三维模型已更新并应用到在线监测。")
                messagebox.showinfo(
                    "机翼模型已更新",
                    "机翼三维模型已保存并设为当前模型。\n\n"
                    "后台识别模型和48点传感器坐标均未改变。\n"
                    "建议进入“传感器布置”页面检查传感器是否仍位于新机翼表面。",
                )
            except Exception as exc:
                messagebox.showerror("保存失败", str(exc))

        def load_active_wing_geometry_into_editor(self) -> None:
            manifest = self.active_geometry_manifest
            if not manifest:
                # 兼容旧版：尝试从旧传感器布置清单读取几何。
                manifest = self.active_layout_manifest
            if not manifest:
                self.wing_geometry_info.set("当前未配置机翼三维模型。")
                self.wing_geometry_status.set("请选择模型文件并加载预览。")
                return
            mesh_path = manifest.get("mesh_npz_path_resolved")
            original = manifest.get("original_geometry_path_resolved")
            meta = manifest.get("geometry", {})
            if original:
                self.wing_geometry_path.set(str(original))
            if meta:
                self.wing_span_axis.set(str(meta.get("span_axis", "Z")))
                self.wing_chord_axis.set(str(meta.get("chord_axis", "X")))
                self.wing_geometry_unit.set(str(meta.get("source_unit", "mm")))
                self.wing_reverse_span.set(bool(meta.get("reverse_span", False)))
                self.wing_geometry_info.set(
                    f"当前模型：{manifest.get('geometry_name', meta.get('source_filename', '机翼模型'))}\n"
                    f"展向长度：{float(meta.get('span_mm', 0.0)):.3f} mm\n"
                    f"根弦长度：{float(meta.get('root_chord_mm', 0.0)):.3f} mm\n"
                    f"尖弦长度：{float(meta.get('tip_chord_mm', 0.0)):.3f} mm\n"
                    f"最大厚度：{float(meta.get('thickness_mm', 0.0)):.3f} mm"
                )
            self.wing_work_mesh = load_prepared_mesh(mesh_path) if mesh_path else None
            self.wing_geometry_status.set("已载入当前生效的机翼三维模型。")

        def draw_wing_geometry_preview(self) -> None:
            if self.wing_figure is None:
                return
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            self.wing_figure.clear()
            ax = self.wing_figure.add_subplot(111, projection="3d")
            mesh = self.wing_work_mesh
            if not mesh:
                ax.text2D(0.5, 0.5, "尚未加载机翼三维模型", transform=ax.transAxes,
                          ha="center", va="center", fontsize=14)
                ax.set_axis_off()
                self.wing_canvas.draw_idle()
                return
            vertices = np.asarray(mesh["vertices"], dtype=float)
            faces = np.asarray(mesh["faces"], dtype=int)
            if len(faces) > 20000:
                pick = np.linspace(0, len(faces) - 1, 20000, dtype=int)
                faces = faces[pick]
            collection = Poly3DCollection(
                vertices[faces], facecolor=(0.45, 0.68, 0.88, 0.68),
                edgecolor=(0.12, 0.18, 0.24, 0.12), linewidth=0.08
            )
            ax.add_collection3d(collection)
            mins = vertices.min(axis=0)
            maxs = vertices.max(axis=0)
            spans = np.maximum(maxs - mins, 1.0)
            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(maxs[1], mins[1])
            ax.set_zlim(mins[2] - 0.12 * spans[2], maxs[2] + 0.20 * spans[2])
            ax.set_box_aspect((spans[0], spans[1], max(spans[2], 0.20 * spans[1])))
            ax.view_init(elev=24, azim=-62)
            ax.set_xlabel("展向 z / mm")
            ax.set_ylabel("弦向 x / mm")
            ax.set_zlabel("厚度 y / mm")
            ax.set_title("机翼三维模型预览")
            self.wing_canvas.draw_idle()

        def load_active_geometry_into_sensor_editor(self, preserve_coordinates: bool = True) -> None:
            manifest = self.active_geometry_manifest
            if not manifest:
                manifest = self.active_layout_manifest
            if not manifest or not manifest.get("mesh_npz_path_resolved"):
                self.sensor_work_mesh = None
                self.sensor_geometry_text.set("当前未配置机翼三维模型。请先在“机翼三维模型”页面设置。")
                self.sensor_layout_status.set("没有可用的当前机翼模型。")
                self.draw_sensor_editor()
                return
            self.sensor_work_mesh = load_prepared_mesh(manifest.get("mesh_npz_path_resolved"))
            meta = manifest.get("geometry", {})
            self.sensor_geometry_text.set(
                f"模型：{manifest.get('geometry_name', meta.get('source_filename', '机翼模型'))} ｜ "
                f"展向 {float(meta.get('span_mm', 0.0)):.2f} mm ｜ "
                f"根弦 {float(meta.get('root_chord_mm', 0.0)):.2f} mm ｜ "
                f"尖弦 {float(meta.get('tip_chord_mm', 0.0)):.2f} mm"
            )
            if not preserve_coordinates:
                self.generate_default_sensor_layout()
            else:
                self.refresh_sensor_table()
                self.select_sensor(int(self.selected_sensor_id.get()))
                self.draw_sensor_editor()
                self.sensor_layout_status.set("已载入当前机翼模型，现有48点坐标保持不变。")

        def nearest_surface_height(self, z_value: float, x_value: float) -> float:
            if not self.sensor_work_mesh:
                return float(self.sensor_y_var.get())
            vertices = np.asarray(self.sensor_work_mesh["vertices"], dtype=float)
            distance = (vertices[:, 0] - float(z_value)) ** 2 + (vertices[:, 1] - float(x_value)) ** 2
            count = min(64, len(vertices))
            nearest = np.argpartition(distance, count - 1)[:count]
            return float(np.max(vertices[nearest, 2]))

        def generate_default_sensor_layout(self) -> None:
            manifest = self.active_geometry_manifest or self.active_layout_manifest
            if manifest:
                meta = manifest.get("geometry", {})
                geometry = WingGeometry(
                    float(meta.get("span_mm", self.geometry.span_mm)),
                    float(meta.get("root_chord_mm", self.geometry.root_chord_mm)),
                    float(meta.get("tip_chord_mm", self.geometry.tip_chord_mm)),
                )
            else:
                geometry = self.geometry
            plan = build_sensor_coordinates(
                geometry,
                self.base_config["sensors"]["span_fractions"],
                self.base_config["sensors"]["chord_fractions"],
            )
            heights = np.asarray(
                [self.nearest_surface_height(z, x) for z, x in plan], dtype=float
            )
            self.sensor_coords_edit = np.column_stack([plan, heights])
            self.refresh_sensor_table()
            self.select_sensor(1)
            self.draw_sensor_editor()
            self.sensor_layout_status.set("已生成8×6规则初始布置，可继续逐点调整。")

        def refresh_sensor_table(self) -> None:
            if not hasattr(self, "sensor_tree"):
                return
            selected = int(self.selected_sensor_id.get())
            for item in self.sensor_tree.get_children():
                self.sensor_tree.delete(item)
            for index, row in enumerate(np.asarray(self.sensor_coords_edit), start=1):
                self.sensor_tree.insert(
                    "", "end", iid=str(index),
                    values=(index, f"{row[0]:.3f}", f"{row[1]:.3f}", f"{row[2]:.3f}")
                )
            if str(selected) in self.sensor_tree.get_children():
                self.sensor_tree.selection_set(str(selected))
                self.sensor_tree.see(str(selected))

        def select_sensor(self, sensor_id: int) -> None:
            sensor_id = int(np.clip(sensor_id, 1, 48))
            self.selected_sensor_id.set(sensor_id)
            row = np.asarray(self.sensor_coords_edit[sensor_id - 1], dtype=float)
            self.sensor_z_var.set(float(row[0]))
            self.sensor_x_var.set(float(row[1]))
            self.sensor_y_var.set(float(row[2]))
            if hasattr(self, "sensor_tree") and str(sensor_id) in self.sensor_tree.get_children():
                current = self.sensor_tree.selection()
                if current != (str(sensor_id),):
                    self.sensor_tree.selection_set(str(sensor_id))
                self.sensor_tree.see(str(sensor_id))

        def on_sensor_selected(self, _event=None) -> None:
            selection = self.sensor_tree.selection()
            if selection:
                sensor_id = int(selection[0])
                self.selected_sensor_id.set(sensor_id)
                row = np.asarray(self.sensor_coords_edit[sensor_id - 1], dtype=float)
                self.sensor_z_var.set(float(row[0]))
                self.sensor_x_var.set(float(row[1]))
                self.sensor_y_var.set(float(row[2]))
                self.draw_sensor_editor()

        def on_sensor_id_changed(self, _event=None) -> None:
            self.select_sensor(int(self.selected_sensor_id.get()))
            self.draw_sensor_editor()

        def apply_sensor_coordinate(self) -> None:
            try:
                sensor_id = int(self.selected_sensor_id.get())
                values = np.asarray(
                    [self.sensor_z_var.get(), self.sensor_x_var.get(), self.sensor_y_var.get()],
                    dtype=float,
                )
                if not np.all(np.isfinite(values)):
                    raise ValueError("坐标必须是有限数值。")
                self.sensor_coords_edit[sensor_id - 1] = values
                self.refresh_sensor_table()
                self.draw_sensor_editor()
                self.sensor_layout_status.set(f"传感器 {sensor_id} 坐标已更新。")
            except Exception as exc:
                messagebox.showerror("坐标输入错误", str(exc))

        def nudge_sensor(self, axis_index: int, direction: float) -> None:
            step = float(self.sensor_step_var.get())
            sensor_id = int(self.selected_sensor_id.get())
            self.sensor_coords_edit[sensor_id - 1, axis_index] += direction * step
            self.select_sensor(sensor_id)
            self.refresh_sensor_table()
            self.draw_sensor_editor()

        def snap_selected_sensor(self) -> None:
            if not self.sensor_work_mesh:
                messagebox.showinfo("未加载模型", "请先加载机翼三维模型。")
                return
            sensor_id = int(self.selected_sensor_id.get())
            z, x = self.sensor_coords_edit[sensor_id - 1, :2]
            self.sensor_coords_edit[sensor_id - 1, 2] = self.nearest_surface_height(z, x)
            self.select_sensor(sensor_id)
            self.refresh_sensor_table()
            self.draw_sensor_editor()
            self.sensor_layout_status.set(f"传感器 {sensor_id} 已吸附到机翼上表面。")

        def draw_sensor_editor(self) -> None:
            if self.sensor_figure is None:
                return
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            from scipy.spatial import ConvexHull

            self.sensor_figure.clear()
            gs = self.sensor_figure.add_gridspec(2, 1, height_ratios=(1.45, 1.0), hspace=0.16)
            self.sensor_ax_3d = self.sensor_figure.add_subplot(gs[0, 0], projection="3d")
            self.sensor_ax_plan = self.sensor_figure.add_subplot(gs[1, 0])
            ax3d = self.sensor_ax_3d
            axp = self.sensor_ax_plan

            vertices = faces = None
            if self.sensor_work_mesh:
                vertices = np.asarray(self.sensor_work_mesh["vertices"], dtype=float)
                faces = np.asarray(self.sensor_work_mesh["faces"], dtype=int)
                if len(faces) > 12000:
                    pick = np.linspace(0, len(faces) - 1, 12000, dtype=int)
                    faces = faces[pick]
                collection = Poly3DCollection(
                    vertices[faces], facecolor=(0.70, 0.74, 0.80, 0.35),
                    edgecolor=(0.15, 0.18, 0.22, 0.10), linewidth=0.08
                )
                ax3d.add_collection3d(collection)
                try:
                    hull = ConvexHull(vertices[:, :2])
                    outline = vertices[hull.vertices, :2]
                    axp.plot(
                        np.r_[outline[:, 0], outline[0, 0]],
                        np.r_[outline[:, 1], outline[0, 1]],
                        color="black", linewidth=1.2
                    )
                except Exception:
                    axp.scatter(vertices[::max(1, len(vertices)//2500), 0],
                                vertices[::max(1, len(vertices)//2500), 1], s=1, alpha=0.15)

            coords = np.asarray(self.sensor_coords_edit, dtype=float)
            selected = int(self.selected_sensor_id.get()) - 1
            colors = np.full((48,), "tab:blue", dtype=object)
            colors[selected] = "crimson"
            sizes = np.full((48,), 26.0)
            sizes[selected] = 90.0
            ax3d.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colors, s=sizes, depthshade=False)
            axp.scatter(coords[:, 0], coords[:, 1], c=colors, s=sizes)
            ax3d.text(
                coords[selected, 0], coords[selected, 1], coords[selected, 2],
                f"  S{selected + 1}\n  ({coords[selected,0]:.2f}, {coords[selected,1]:.2f}, {coords[selected,2]:.2f})",
                color="crimson", fontsize=9, weight="bold"
            )
            axp.annotate(
                f"S{selected + 1}", xy=(coords[selected, 0], coords[selected, 1]),
                xytext=(6, 6), textcoords="offset points", color="crimson", weight="bold"
            )

            if vertices is not None:
                mins = vertices.min(axis=0)
                maxs = vertices.max(axis=0)
            else:
                mins = np.minimum(coords.min(axis=0), 0.0)
                maxs = np.maximum(coords.max(axis=0), 1.0)
            spans = np.maximum(maxs - mins, 1.0)
            ax3d.set_xlim(mins[0], maxs[0])
            ax3d.set_ylim(maxs[1], mins[1])
            ax3d.set_zlim(mins[2] - 0.15 * spans[2], maxs[2] + 0.25 * spans[2])
            ax3d.set_box_aspect((spans[0], spans[1], max(spans[2], 0.22 * spans[1])))
            ax3d.view_init(elev=24, azim=-62)
            ax3d.set_xlabel("展向 z / mm")
            ax3d.set_ylabel("弦向 x / mm")
            ax3d.set_zlabel("厚度 y / mm")
            ax3d.set_title("三维传感器布置")

            axp.set_xlim(mins[0], maxs[0])
            axp.set_ylim(maxs[1], mins[1])
            axp.set_aspect("equal", adjustable="box")
            axp.set_xlabel("展向 z / mm")
            axp.set_ylabel("弦向 x / mm")
            axp.set_title("俯视定位（单击可放置当前传感器）")
            axp.grid(alpha=0.25)
            self.sensor_canvas.draw_idle()

        def on_sensor_canvas_click(self, event) -> None:
            if event.inaxes is not self.sensor_ax_plan or event.xdata is None or event.ydata is None:
                return
            sensor_id = int(self.selected_sensor_id.get())
            z_value = float(event.xdata)
            x_value = float(event.ydata)
            y_value = self.nearest_surface_height(z_value, x_value)
            self.sensor_coords_edit[sensor_id - 1] = [z_value, x_value, y_value]
            self.select_sensor(sensor_id)
            self.refresh_sensor_table()
            self.draw_sensor_editor()
            self.sensor_layout_status.set(
                f"传感器 {sensor_id} 已定位到 z={z_value:.2f}, x={x_value:.2f}, y={y_value:.2f} mm。"
            )

        def import_sensor_layout(self) -> None:
            path = filedialog.askopenfilename(
                title="导入48个传感器坐标",
                filetypes=[("坐标文件", "*.csv *.json"), ("所有文件", "*.*")],
            )
            if not path:
                return
            try:
                self.sensor_coords_edit = load_sensor_coordinates_file(path)
                self.refresh_sensor_table()
                self.select_sensor(1)
                self.draw_sensor_editor()
                self.sensor_layout_status.set(f"已导入传感器坐标：{Path(path).name}")
            except Exception as exc:
                messagebox.showerror("导入失败", str(exc))

        def export_sensor_layout(self) -> None:
            path = filedialog.asksaveasfilename(
                title="导出传感器坐标",
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv")],
            )
            if not path:
                return
            try:
                export_sensor_coordinates_csv(path, self.sensor_coords_edit)
                self.sensor_layout_status.set(f"坐标已导出：{path}")
            except Exception as exc:
                messagebox.showerror("导出失败", str(exc))

        def save_sensor_layout(self) -> None:
            if not self.active_geometry_manifest and not (
                self.active_layout_manifest and self.active_layout_manifest.get("mesh_npz_path_resolved")
            ):
                messagebox.showerror(
                    "无法保存",
                    "当前没有生效的机翼三维模型。请先在“机翼三维模型”页面设置当前机翼。",
                )
                return
            try:
                geometry_manifest = self.active_geometry_manifest or self.active_layout_manifest or {}
                geometry_name = geometry_manifest.get("geometry_name") or geometry_manifest.get("layout_name")
                save_active_sensor_coordinates(
                    BASE_DIR, self.sensor_coords_edit, geometry_name=geometry_name
                )
                self.sensor_layout_status.set("48点传感器坐标已保存并应用到在线监测模块。")
                self._reload_active_context()
                self.load_active_geometry_into_sensor_editor(preserve_coordinates=True)
                self.set_demo_data()
                self.run_analysis()
                messagebox.showinfo(
                    "传感器布置已应用",
                    "48点传感器坐标已保存。\n\n"
                    "机翼三维模型和后台识别模型均未改变。",
                )
            except Exception as exc:
                messagebox.showerror("保存失败", str(exc))

        def load_active_layout_into_editor(self) -> None:
            if self.active_layout_manifest:
                coords = self.active_layout_manifest.get("sensor_coordinates_mm")
                if coords is not None:
                    array = np.asarray(coords, dtype=float)
                    if array.shape == (48, 3):
                        self.sensor_coords_edit = array.copy()
            self.load_active_geometry_into_sensor_editor(preserve_coordinates=True)
            self.refresh_sensor_table()
            self.select_sensor(1)
            self.draw_sensor_editor()
            if self.active_layout_manifest:
                self.sensor_layout_status.set("已载入当前48点传感器坐标和当前机翼模型。")

        def parse_strains(self) -> np.ndarray:
            raw = self.strain_text.get("1.0", tk.END).strip()
            for separator in [",", ";", "\t", "\n"]:
                raw = raw.replace(separator, " ")
            tokens = [token for token in raw.split(" ") if token.strip()]
            try:
                values = np.asarray([float(token) for token in tokens], dtype=float)
            except ValueError as exc:
                raise ValueError("输入内容中包含非数值字符。") from exc
            expected = self.sensor_coords.shape[0]
            if values.size != expected:
                raise ValueError(f"必须输入正好{expected}个应变值，当前读取到{values.size}个。")
            if not np.all(np.isfinite(values)):
                raise ValueError("所有应变值必须为有限数值，不能包含NaN或无穷大。")
            return values

        def set_demo_data(self) -> None:
            values = generate_demo_strains(self.sensor_coords, self.geometry)
            self.strain_text.delete("1.0", tk.END)
            lines = [", ".join(f"{v:.5f}" for v in values[i : i + 6]) for i in range(0, 48, 6)]
            self.strain_text.insert("1.0", "\n".join(lines))
            self.status_text.set("已加载48通道演示应变数据。")

        def clear_data(self) -> None:
            self.strain_text.delete("1.0", tk.END)
            self.status_text.set("输入数据已清空。")

        def open_csv(self) -> None:
            path = filedialog.askopenfilename(
                title="导入应变数据文件",
                filetypes=[("CSV或文本文件", "*.csv *.txt"), ("所有文件", "*.*")],
            )
            if not path:
                return
            try:
                values: list[float] = []
                with open(path, "r", encoding="utf-8-sig", newline="") as handle:
                    for row in csv.reader(handle):
                        for cell in row:
                            cell = cell.strip()
                            if not cell:
                                continue
                            try:
                                values.append(float(cell))
                            except ValueError:
                                continue
                if len(values) < 48:
                    raise ValueError(f"文件中仅找到{len(values)}个数值，少于所需的48个。")
                values = values[:48]
                self.strain_text.delete("1.0", tk.END)
                lines = [", ".join(f"{v:.8g}" for v in values[i : i + 6]) for i in range(0, 48, 6)]
                self.strain_text.insert("1.0", "\n".join(lines))
                self.status_text.set(f"已从 {Path(path).name} 中读取前48个数值。")
            except Exception as exc:
                messagebox.showerror("CSV读取错误", str(exc))

        def browse_model(self) -> None:
            path = filedialog.askopenfilename(
                title="选择模型包或模型适配器",
                filetypes=[
                    ("支持的模型文件", "*.json *.pt *.py *.joblib *.pkl *.pth *.ts"),
                    ("所有文件", "*.*"),
                ],
            )
            if path:
                self.model_path.set(path)

        def browse_reconstruction(self) -> None:
            path = filedialog.askopenfilename(
                title="选择LC-Kriging重构代码",
                filetypes=[("Python文件", "*.py"), ("所有文件", "*.*")],
            )
            if path:
                self.reconstruction_path.set(path)

        def run_analysis(self) -> None:
            try:
                self.status_text.set("正在执行模型推理和应变场重构，请稍候……")
                self.root.update_idletasks()
                values = self.parse_strains()
                prediction = self.model_runner.predict(
                    values,
                    self.sensor_coords,
                    model_mode_map[self.model_mode.get()],
                    self.model_path.get().strip() or None,
                )
                field = self.reconstructor.reconstruct(
                    self.sensor_coords,
                    values,
                    prediction,
                    self.grid_z,
                    self.grid_x,
                    reconstruction_mode_map[self.reconstruction_mode.get()],
                    self.reconstruction_path.get().strip() or None,
                )
                draw_result(
                    self.current_figure,
                    self.geometry,
                    self.sensor_coords,
                    values,
                    self.grid_z,
                    self.grid_x,
                    self.mask,
                    field,
                    prediction,
                    self.config,
                    mesh_data=self.mesh_data,
                    sensor_coords_3d=self.sensor_coords_3d,
                )
                self.canvas.draw_idle()
                self.current_prediction = prediction
                if prediction.get("prediction_branch") == "damaged":
                    self.load_x_text.set("有损状态不适用")
                    self.load_z_text.set("有损状态不适用")
                    self.load_n_text.set("有损状态不适用")
                else:
                    self.load_x_text.set(f"{float(prediction['load_x_mm']):.2f} mm")
                    self.load_z_text.set(f"{float(prediction['load_z_mm']):.2f} mm")
                    self.load_n_text.set(f"{float(prediction['load_n']):.2f} N")
                self.damage_text.set(
                    f"{float(prediction['damage_probability']):.1%}"
                    if "damage_probability" in prediction
                    else "模型未输出"
                )
                self.damage_x_text.set(
                    f"{float(prediction['damage_x_mm']):.2f} mm"
                    if "damage_x_mm" in prediction
                    else "模型未输出"
                )
                self.damage_z_text.set(
                    f"{float(prediction['damage_z_mm']):.2f} mm"
                    if "damage_z_mm" in prediction
                    else "模型未输出"
                )
                self.damage_size_text.set(
                    f"{float(prediction['damage_size_mm']):.2f} mm"
                    if "damage_size_mm" in prediction
                    else "模型未输出"
                )
                if prediction.get("prediction_branch") == "damaged":
                    method = self.reconstructor.last_method or self.reconstruction_mode.get()
                    self.status_text.set(f"分析完成：三维应变云图和损伤参数已显示；重构方法：{method}。")
                else:
                    method = self.reconstructor.last_method or self.reconstruction_mode.get()
                    self.status_text.set(f"分析完成：三维应变云图和载荷参数已显示；重构方法：{method}。")
            except Exception as exc:
                self.status_text.set("分析失败：请检查48通道顺序、模型接口和重构接口。")
                messagebox.showerror(
                    "分析错误", f"{exc}\n\n技术信息：\n{traceback.format_exc(limit=5)}"
                )

        def save_figure(self) -> None:
            path = filedialog.asksaveasfilename(
                title="保存组合结果图",
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            )
            if path:
                self.current_figure.savefig(path, bbox_inches="tight", dpi=300)
                self.status_text.set(f"结果图已保存至：{path}")

        def browse_pretrain_data(self, key: str, title: str) -> None:
            path = filedialog.askopenfilename(
                title=title,
                filetypes=[("数据文件", "*.csv *.txt *.tsv *.xlsx *.xls"), ("所有文件", "*.*")],
            )
            if path:
                self.pretrain_paths[key].set(path)
                if key in self.transfer_paths and not self.transfer_paths[key].get().strip():
                    self.transfer_paths[key].set(path)

        def browse_existing_pretrain(self) -> None:
            path = filedialog.askdirectory(
                title="选择已有BP-CNN预训练模型目录",
                initialdir=str(BASE_DIR / "pretrained_models"),
            )
            if path:
                self.pretrain_existing_path.set(path)

        def _validate_pretrain_inputs(self) -> None:
            missing = []
            for key, label in (("sim_u", "仿真未损伤数据"), ("sim_d", "仿真损伤数据")):
                value = self.pretrain_paths[key].get().strip()
                if not value or not Path(value).exists():
                    missing.append(label)
            if missing:
                raise ValueError("以下预训练输入尚未选择或文件不存在：" + "、".join(missing))
            for label, value in (
                ("BP搜索次数", self.pretrain_bp_trials.get()),
                ("BP训练轮数", self.pretrain_bp_epochs.get()),
                ("CNN训练轮数", self.pretrain_cnn_epochs.get()),
                ("批大小", self.pretrain_batch_size.get()),
            ):
                if int(value) <= 0:
                    raise ValueError(f"{label}必须大于0。")

        def start_pretraining(self) -> None:
            if self.training_thread and self.training_thread.is_alive():
                messagebox.showinfo("任务正在进行", "当前已有训练任务正在运行。")
                return
            try:
                self._validate_pretrain_inputs()
            except Exception as exc:
                messagebox.showerror("输入检查失败", str(exc))
                return
            self.cancel_event = threading.Event()
            self.pretrain_progress.set(0.0)
            self.pretrain_status.set("正在执行第1步：模型预训练……")
            self.start_pretrain_button.configure(state=tk.DISABLED)
            self.cancel_pretrain_button.configure(state=tk.NORMAL)
            self._append_pretrain_log("=" * 58)
            self._append_pretrain_log("预训练任务已启动：BP分类 + CNN多任务回归。")
            cfg = PretrainConfig(
                feature_count=48,
                bp_trials=int(self.pretrain_bp_trials.get()),
                bp_epochs=int(self.pretrain_bp_epochs.get()),
                cnn_epochs=int(self.pretrain_cnn_epochs.get()),
                batch_size=int(self.pretrain_batch_size.get()),
            )
            arguments = {
                "base_dir": BASE_DIR,
                "sim_undamaged_path": self.pretrain_paths["sim_u"].get(),
                "sim_damaged_path": self.pretrain_paths["sim_d"].get(),
                "training_config": cfg,
                "cancel_event": self.cancel_event,
            }

            def worker() -> None:
                try:
                    def callback(fraction: float, message: str) -> None:
                        self.training_queue.put(("pretrain_progress", (fraction, message)))
                    result = train_pretrained_models(**arguments, progress_callback=callback)
                    self.training_queue.put(("pretrain_success", result))
                except PretrainCancelled as exc:
                    self.training_queue.put(("pretrain_cancelled", str(exc)))
                except Exception as exc:
                    self.training_queue.put(("pretrain_error", (str(exc), traceback.format_exc(limit=12))))

            self.training_thread = threading.Thread(target=worker, daemon=True)
            self.training_thread.start()

        def activate_existing_pretrain_model(self) -> None:
            selected = self.pretrain_existing_path.get().strip()
            if not selected:
                selected = filedialog.askdirectory(
                    title="选择已有BP-CNN预训练模型目录",
                    initialdir=str(BASE_DIR / "pretrained_models"),
                )
                if not selected:
                    return
                self.pretrain_existing_path.set(selected)
            try:
                activate_existing_pretrain(BASE_DIR, selected)
                self.active_pretrain_manifest = read_active_pretrain(BASE_DIR)
                self._refresh_pretrain_model_card()
                self.pretrain_status.set("已有预训练模型已设为当前教师模型。")
                messagebox.showinfo("设置完成", "所选BP-CNN预训练模型已用于后续迁移学习。")
            except Exception as exc:
                messagebox.showerror("设置失败", str(exc))

        def browse_transfer_data(self, key: str, title: str) -> None:
            path = filedialog.askopenfilename(
                title=title,
                filetypes=[
                    ("数据文件", "*.csv *.txt *.tsv *.xlsx *.xls"),
                    ("所有文件", "*.*"),
                ],
            )
            if path:
                self.transfer_paths[key].set(path)

        def load_transfer_demo_paths(self) -> None:
            demo = BASE_DIR / "sample_data" / "transfer_demo"
            mapping = {
                "sim_u": demo / "simulation_undamaged.csv",
                "sim_d": demo / "simulation_damaged.csv",
                "act_u": demo / "actual_undamaged.csv",
                "act_d": demo / "actual_damaged.csv",
            }
            for key, path in mapping.items():
                self.transfer_paths[key].set(str(path))
            self.training_status.set("已填入软件自带的四类示例数据路径。")
            self._append_log("已载入示例训练数据。示例仅用于流程测试，不代表正式实验模型。")

        def _validate_training_inputs(self) -> None:
            missing = []
            labels = {
                "sim_u": "仿真未损伤数据",
                "sim_d": "仿真损伤数据",
                "act_u": "实际未损伤数据",
                "act_d": "实际损伤数据",
            }
            for key, label in labels.items():
                value = self.transfer_paths[key].get().strip()
                if not value or not Path(value).exists():
                    missing.append(label)
            if missing:
                raise ValueError("以下输入尚未选择或文件不存在：" + "、".join(missing))
            for label, value in (
                ("潜在维度", self.latent_dim.get()),
                ("AE轮数", self.ae_epochs.get()),
                ("回归学生轮数", self.pretrain_epochs.get()),
                ("分类学生轮数", self.cls_epochs.get()),
                ("有损AE微调轮数", self.finetune_epochs.get()),
                ("批大小", self.batch_size.get()),
            ):
                if int(value) <= 0:
                    raise ValueError(f"{label}必须大于0。")

        def start_training(self) -> None:
            if self.training_thread and self.training_thread.is_alive():
                messagebox.showinfo("训练正在进行", "当前已有一个训练任务正在运行。")
                return
            self.active_pretrain_manifest = read_active_pretrain(BASE_DIR)
            if not self.active_pretrain_manifest:
                messagebox.showerror(
                    "缺少预训练模型",
                    "迁移学习必须先完成第1步预训练。\n\n请先训练BP分类模型和CNN回归模型，或选择已有 pretrain_bundle.json。",
                )
                self.notebook.select(self.pretrain_tab)
                return
            try:
                self._validate_training_inputs()
            except Exception as exc:
                messagebox.showerror("输入检查失败", str(exc))
                return

            self.cancel_event = threading.Event()
            self.training_progress.set(0.0)
            self.training_status.set("正在执行迁移学习……")
            self.start_training_button.configure(state=tk.DISABLED)
            self.cancel_training_button.configure(state=tk.NORMAL)
            self._append_log("=" * 58)
            self._append_log("迁移学习任务已启动。模型：Keras AE-guided KD v3。")

            cfg = KerasV3TrainingConfig(
                feature_count=48,
                latent_dim=int(self.latent_dim.get()),
                batch_size=int(self.batch_size.get()),
                ae_epochs=int(self.ae_epochs.get()),
                reg_epochs=int(self.pretrain_epochs.get()),
                cls_epochs=int(self.cls_epochs.get()),
                finetune_epochs=int(self.finetune_epochs.get()),
                position_scale=1000.0,
                load_scale=1.0,
            )
            arguments = {
                "base_dir": BASE_DIR,
                "sim_undamaged_path": self.transfer_paths["sim_u"].get(),
                "sim_damaged_path": self.transfer_paths["sim_d"].get(),
                "actual_undamaged_path": self.transfer_paths["act_u"].get(),
                "actual_damaged_path": self.transfer_paths["act_d"].get(),
                "training_config": cfg,
                "cancel_event": self.cancel_event,
            }

            def worker() -> None:
                try:
                    def callback(fraction: float, message: str) -> None:
                        self.training_queue.put(("progress", (fraction, message)))

                    result = train_keras_v3(
                        **arguments, progress_callback=callback
                    )
                    self.training_queue.put(("success", result))
                except TrainingCancelled as exc:
                    self.training_queue.put(("cancelled", str(exc)))
                except Exception as exc:
                    self.training_queue.put(
                        ("error", (str(exc), traceback.format_exc(limit=12)))
                    )

            self.training_thread = threading.Thread(target=worker, daemon=True)
            self.training_thread.start()

        def update_backend_model(self) -> None:
            training_running = bool(self.training_thread and self.training_thread.is_alive())
            if training_running:
                proceed = messagebox.askyesno(
                    "迁移学习正在运行",
                    "当前训练仍在运行。你可以独立更新已有后台模型，且不会终止训练。是否继续？",
                )
                if not proceed:
                    return

            self.pending_manifest = read_pending_manifest(BASE_DIR)
            selected_existing = self.manual_model_path.get().strip() or None
            use_pending = selected_existing is None and self.pending_manifest is not None

            if use_pending:
                model_name = self.pending_manifest.get("model_name", "Keras迁移模型")
                confirmed = messagebox.askyesno(
                    "确认更新后台模型",
                    f"将待更新模型“{model_name}”设为当前后台模型？\n\n"
                    "本操作只更新识别模型，不修改机翼三维模型和传感器坐标。",
                )
                if not confirmed:
                    return
            else:
                if selected_existing is None:
                    messagebox.showinfo(
                        "选择已有模型",
                        "当前没有待更新模型。请选择已有迁移模型目录。\n\n"
                        "可选择模型根目录，也可直接选择 models_transfer_v3 或 models_student_v3。",
                    )
                    selected_existing = filedialog.askdirectory(
                        title="选择已有Keras v3模型目录",
                        initialdir=str(BASE_DIR / "models"),
                    )
                    if not selected_existing:
                        self.training_status.set("已取消后台模型更新。")
                        return
                    self.manual_model_path.set(selected_existing)
                confirmed = messagebox.askyesno(
                    "确认更新后台模型",
                    f"确定使用以下模型更新后台吗？\n\n{selected_existing}\n\n"
                    "软件只检查并切换模型文件；三维机翼和传感器布置保持不变。",
                )
                if not confirmed:
                    return

            try:
                self.training_status.set("正在检查并更新后台模型……")
                if use_pending:
                    active_path = activate_pending_keras_v3(BASE_DIR)
                else:
                    active_path = activate_existing_keras_v3(BASE_DIR, selected_existing)
                self._append_log(f"后台模型已更新：{active_path}")
                self.training_status.set("后台模型更新完成。")
                self._reload_active_context()
                self._refresh_pending_model_card()
                messagebox.showinfo(
                    "更新完成",
                    "所选模型已设为当前后台模型。\n"
                    "机翼三维模型和48个传感器坐标未改变。",
                )
            except Exception as exc:
                self.training_status.set("后台模型更新失败。")
                self._append_log("更新失败：" + str(exc))
                self.pending_manifest = read_pending_manifest(BASE_DIR)
                self._refresh_pending_model_card()
                messagebox.showerror(
                    "更新失败",
                    f"{exc}\n\n请不要选择STL文件或只包含三维模型的文件夹；"
                    "模型更新需要Keras模型输出目录。",
                )

        def cancel_training(self) -> None:
            if self.cancel_event is not None:
                self.cancel_event.set()
                self.training_status.set("正在等待当前训练批次安全停止……")
                self._append_log("已请求取消；软件将在当前批次结束后停止，不会更改当前后台模型。")

        def _poll_training_queue(self) -> None:
            try:
                while True:
                    kind, payload = self.training_queue.get_nowait()
                    if kind == "pretrain_progress":
                        fraction, message = payload
                        self.pretrain_progress.set(float(fraction) * 100.0)
                        self.pretrain_status.set(message)
                        self._append_pretrain_log(message)
                    elif kind == "pretrain_success":
                        result = payload
                        self.pretrain_progress.set(100.0)
                        self.pretrain_status.set("预训练完成，模型已设为当前迁移教师模型。")
                        self._append_pretrain_log(f"预训练模型包：{result.bundle_path}")
                        self.active_pretrain_manifest = read_active_pretrain(BASE_DIR)
                        self._refresh_pretrain_model_card()
                        self._finish_training_controls()
                        messagebox.showinfo(
                            "预训练完成",
                            "BP分类模型和CNN回归模型已生成。\n\n现在可以进入第2步执行迁移学习。",
                        )
                    elif kind == "pretrain_cancelled":
                        self.pretrain_status.set(payload)
                        self._append_pretrain_log(payload)
                        self._finish_training_controls()
                    elif kind == "pretrain_error":
                        message, details = payload
                        self.pretrain_status.set("预训练失败。")
                        self._append_pretrain_log("错误：" + message)
                        self._append_pretrain_log(details)
                        self._finish_training_controls()
                        messagebox.showerror("预训练失败", f"{message}\n\n技术信息：\n{details}")
                    elif kind == "progress":
                        fraction, message = payload
                        self.training_progress.set(float(fraction) * 100.0)
                        self.training_status.set(message)
                        self._append_log(message)
                    elif kind == "success":
                        result = payload
                        self.training_progress.set(100.0)
                        self.training_status.set("迁移学习完成，模型等待更新。")
                        self._append_log(f"模型包：{result.bundle_path}")
                        self._append_log(f"训练报告：{result.metadata_path}")
                        self._append_log("模型已生成并通过预测验证，当前后台模型尚未改变。")
                        self._finish_training_controls()
                        self.pending_manifest = read_pending_manifest(BASE_DIR)
                        self._refresh_pending_model_card()
                        messagebox.showinfo(
                            "迁移学习完成",
                            "迁移模型已生成并通过验证。\n\n"
                            "点击“更新后台模型”后，新模型才会在在线监测中生效。",
                        )
                    elif kind == "cancelled":
                        self.training_status.set(payload)
                        self._append_log(payload)
                        self._finish_training_controls()
                    elif kind == "error":
                        message, details = payload
                        self.training_status.set("迁移学习失败，当前后台模型未改变。")
                        self._append_log("错误：" + message)
                        self._append_log(details)
                        self._finish_training_controls()
                        messagebox.showerror(
                            "迁移学习失败",
                            f"{message}\n\n当前后台模型保持不变。\n\n技术信息：\n{details}",
                        )
            except queue.Empty:
                pass
            if self.root.winfo_exists():
                self.root.after(120, self._poll_training_queue)

        def _finish_training_controls(self) -> None:
            if hasattr(self, "start_pretrain_button"):
                self.start_pretrain_button.configure(state=tk.NORMAL)
            if hasattr(self, "cancel_pretrain_button"):
                self.cancel_pretrain_button.configure(state=tk.DISABLED)
            self.start_training_button.configure(state=tk.NORMAL)
            self.cancel_training_button.configure(state=tk.DISABLED)
            self.pending_manifest = read_pending_manifest(BASE_DIR)
            self.update_model_button.configure(state=tk.NORMAL)

        def _append_pretrain_log(self, text: str) -> None:
            if not hasattr(self, "pretrain_log"):
                return
            self.pretrain_log.configure(state=tk.NORMAL)
            self.pretrain_log.insert(tk.END, text.rstrip() + "\n")
            self.pretrain_log.see(tk.END)
            self.pretrain_log.configure(state=tk.DISABLED)

        def _append_log(self, text: str) -> None:
            self.training_log.configure(state=tk.NORMAL)
            self.training_log.insert(tk.END, text.rstrip() + "\n")
            self.training_log.see(tk.END)
            self.training_log.configure(state=tk.DISABLED)

        def _refresh_external_model_card(self) -> None:
            if not hasattr(self, "external_model_text"):
                return
            self.active_external_manifest = read_active_external_model(BASE_DIR)
            manifest = self.active_external_manifest
            if not manifest:
                self.external_model_text.set(
                    "状态：未应用外部模型\n可导入外部训练完成的模型并直接用于在线监测。"
                )
                return
            if manifest.get("model_type") == "seven_model_group":
                slot_count = len(manifest.get("slots") or {})
                self.external_model_text.set(
                    f"模型名称：{manifest.get('model_name', '七模型组合')}\n"
                    "模型组成：1个损伤分类 + 3个无损回归 + 3个损伤回归\n"
                    f"已配置模型：{slot_count}/7\n"
                    f"损伤阈值：{float(manifest.get('damage_threshold', 0.5)):.2f}\n"
                    f"应用时间：{manifest.get('updated_at', '—')}\n"
                    f"模型位置：{manifest.get('model_path_resolved', '—')}\n"
                    "状态：当前在线监测使用七模型分支推理"
                )
            else:
                fields = manifest.get("output_order") or []
                self.external_model_text.set(
                    f"模型名称：{manifest.get('model_name', '外部模型')}\n"
                    f"模型类型：{manifest.get('model_type_label', manifest.get('model_type', '—'))}\n"
                    f"应用时间：{manifest.get('updated_at', '—')}\n"
                    f"输出字段：{', '.join(fields) if fields else '由模型包自动识别'}\n"
                    f"模型位置：{manifest.get('model_path_resolved', '—')}\n"
                    "状态：当前在线监测优先使用该模型"
                )


        def _refresh_pretrain_model_card(self) -> None:
            if not hasattr(self, "pretrain_model_text"):
                return
            self.active_pretrain_manifest = read_active_pretrain(BASE_DIR)
            manifest = self.active_pretrain_manifest
            if not manifest:
                self.pretrain_model_text.set(
                    "状态：未配置预训练模型\n迁移学习暂不可执行。请先完成第1步预训练或选择已有模型。"
                )
                return
            metrics = manifest.get("metrics", {})
            bp = metrics.get("bp_classifier", {}) if isinstance(metrics, dict) else {}
            intact = metrics.get("cnn_intact", {}) if isinstance(metrics, dict) else {}
            damage = metrics.get("cnn_damage", {}) if isinstance(metrics, dict) else {}
            lines = [
                f"模型名称：{manifest.get('model_name', 'BP-CNN预训练模型')}",
                f"模型类型：{manifest.get('model_type', 'BP-CNN-Wing-Pretrain-v1')}",
                f"BP测试准确率：{bp.get('accuracy', '—')}",
                "CNN无损输出：load_x / load_z / load_size",
                "CNN损伤输出：damage_x / damage_z / damage_size",
                f"模型目录：{manifest.get('model_dir_resolved', '—')}",
                "状态：可用于第2步迁移学习",
            ]
            self.pretrain_model_text.set("\n".join(lines))

        def _refresh_active_model_card(self) -> None:
            if not hasattr(self, "active_model_text"):
                return
            manifest = self.active_manifest
            if not manifest:
                self.active_model_text.set(
                    "状态：未配置正式模型\n在线监测当前使用演示模型或手动选择的外部模型。"
                )
                return
            metrics = manifest.get("metrics", {})
            class_acc = metrics.get("damage_classification", {}).get("accuracy")
            load_mae = metrics.get("load_regression", {}).get("mae_mean")
            if class_acc is None:
                class_acc = metrics.get("student_bp_cls", {}).get("accuracy")
            if load_mae is None:
                head_maes = [
                    metrics.get(name, {}).get("mae")
                    for name in ("student_load_x", "student_load_y", "student_load_size")
                ]
                valid_maes = [float(value) for value in head_maes if isinstance(value, (int, float))]
                load_mae = float(np.mean(valid_maes)) if valid_maes else None
            lines = [
                f"模型名称：{manifest.get('model_name', 'AE-FST')}",
                f"模型类型：{manifest.get('model_type', 'AE-FST')}",
                f"生效时间：{manifest.get('updated_at', '—')}",
                f"模型状态：已生效",
                "机翼几何：由“机翼三维模型”模块独立管理",
                "传感器坐标：由“传感器布置”模块独立管理",
                f"损伤分类训练诊断准确率：{class_acc:.2%}" if isinstance(class_acc, (int, float)) else "损伤分类诊断：—",
                f"载荷回归训练诊断平均MAE：{load_mae:.4g}" if isinstance(load_mae, (int, float)) else "载荷回归诊断：—",
                "说明：上述指标是训练诊断，不代替独立测试集验证。",
            ]
            self.active_model_text.set("\n".join(lines))

        def _refresh_pending_model_card(self) -> None:
            if not hasattr(self, "pending_model_text"):
                return
            self.pending_manifest = read_pending_manifest(BASE_DIR)
            if not self.pending_manifest:
                self.pending_model_text.set("状态：无待更新模型\n仍可点击“更新后台模型”，选择已有模型目录进行更新。")
                if hasattr(self, "update_model_button"):
                    self.update_model_button.configure(state=tk.NORMAL)
                return
            lines = [
                f"模型名称：{self.pending_manifest.get('model_name', 'Keras迁移模型')}",
                f"模型类型：{self.pending_manifest.get('model_type', 'Keras-AE-guided-KD-v3')}",
                "模型状态：已验证，待更新",
                f"生成时间：{self.pending_manifest.get('created_at', '—')}",
                "更新范围：仅后台识别模型",
            ]
            self.pending_model_text.set("\n".join(lines))
            if hasattr(self, "update_model_button"):
                self.update_model_button.configure(state=tk.NORMAL)

    root = tk.Tk()
    app = WingMonitorApp(root)

    if preview_output is not None:
        # 预览在线监测页面中的内置LC-Kriging选项。
        app.notebook.select(app.monitor_tab)
        app.reconstruction_mode.set("LC-Kriging（内置用户代码）")
        app.status_text.set(
            "就绪：七模型负责参数识别，内置LC-Kriging使用当前48点坐标重构三维应变场。"
        )

        def capture() -> None:
            try:
                from PIL import ImageGrab

                root.update_idletasks()
                x = root.winfo_rootx()
                y = root.winfo_rooty()
                width = root.winfo_width()
                height = root.winfo_height()
                image = ImageGrab.grab((x, y, x + width, y + height))
                preview_output.parent.mkdir(parents=True, exist_ok=True)
                image.save(preview_output)
            finally:
                root.after(100, root.destroy)

        root.after(1800, capture)

    root.mainloop()



def _hide_console_window() -> None:
    """Hide the console for normal GUI startup while keeping a console-subsystem exe for worker stdout."""
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0)
    except Exception:
        pass


def _run_internal_worker() -> int | None:
    """Dispatch training workers inside a frozen PyInstaller executable.

    Frozen applications cannot execute bundled .py files with ``sys.executable script.py``
    because ``sys.executable`` points to the packaged EXE.  The GUI therefore relaunches
    the same EXE with ``--internal-worker`` and this dispatcher calls the original
    training module entry point.
    """
    if len(sys.argv) < 3 or sys.argv[1] != "--internal-worker":
        return None
    worker = sys.argv[2].strip().lower()
    worker_args = sys.argv[3:]
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0], *worker_args]
        if worker == "pretrain":
            from backend.train_pretrain_bp_cnn import main as worker_main
        elif worker == "transfer":
            from backend.train_transfer_ae_guided_kd_v3 import main as worker_main
        else:
            print(f"[ERROR] 未知内部训练任务：{worker}", flush=True)
            return 2
        worker_main()
        return 0
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        sys.argv = original_argv

def main() -> int:
    worker_code = _run_internal_worker()
    if worker_code is not None:
        return worker_code
    if getattr(sys, "frozen", False):
        _hide_console_window()
    parser = argparse.ArgumentParser(description="机翼应变监测与模型管理系统")
    parser.add_argument("--demo-output", type=Path, help="不启动界面，直接生成一张演示组合图。")
    parser.add_argument("--preview-output", type=Path, help="启动界面并保存模型训练页面预览截图。")
    args = parser.parse_args()
    if args.demo_output:
        render_demo(args.demo_output)
        print(f"演示结果图已保存至：{args.demo_output}")
        return 0
    run_gui(args.preview_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
