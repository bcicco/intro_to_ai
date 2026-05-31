import json
import sys
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

import folium
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.lines import Line2D
from tkcalendar import DateEntry

# global paths
_DIR = Path(__file__).parent
_ROOT = _DIR.parent
sys.path.insert(0, str(_ROOT))

_PROCESSED = Path(__file__).parent / "data" / "processed"
SITES_FILE = _PROCESSED / "boroondara_sites.json"
EDGES_FILE = _PROCESSED / "proposed_edges.json"
WIDE_FILE = _PROCESSED / "scats_wide.parquet"
MODEL_FILE = Path(__file__).parent / "models" / "gru_model.pt"
XGB_MODEL_FILE = Path(__file__).parent / "models" / "xgboost_model.pkl"
LSTM_MODEL_FILE = Path(__file__).parent / "models" / "lstm_model.pt"


from assignment1.search import METHODS  # noqa: E402
from assignment2.build_problem import build_problem
from assignment2.helpers.GRU_helpers import rebuild_scalers, load_model, predict_volumes
from assignment2.helpers.XGBoost_helper import load_xgb_models, predict_volumes_xgb
from assignment2.helpers.LSTM_helpers import load_model as load_model_lstm, predict_volumes as predict_volumes_lstm, rebuild_scalers as rebuild_scalers_lstm
from assignment2.helpers.yen_ksp import yen_k_shortest

COST_FUNCTIONS: dict[str, str] = {
    "quadratic": "quadratic",
    "bpr": "bpr",
}

# ── Static data ────────────────────────────────────────────────────────────────
with open(SITES_FILE) as f:
    _SITES: dict = json.load(f)
with open(EDGES_FILE) as f:
    _EDGES: list = json.load(f)

_SITE_IDS = sorted(_SITES.keys())
_METHODS = list(METHODS.keys())
_COSTS = list(COST_FUNCTIONS.keys())

_EDGE_LOOKUP = {(e["from"], e["to"]): e["roads"][0] for e in _EDGES if e["roads"]}

_ROUTE_HTML = _DIR / "data" / "processed" / "_route_overlay.html"
_GRAPH_HTML = _DIR / "data" / "processed" / "boroondara_graph.html"


def _normalise_id(raw: str) -> str:
    return str(int(str(raw).strip())).zfill(4)


# Folium HTML (for browser use)


def _build_route_html(site_path: list, method: str, cost: float, cost_fn: str) -> Path:
    path_set = set(site_path)
    path_pairs = set(zip(site_path, site_path[1:]))

    valid_coords = [
        (s["latitude"], s["longitude"])
        for s in _SITES.values()
        if s["latitude"] is not None and s["longitude"] is not None
    ]
    centre_lat = sum(c[0] for c in valid_coords) / len(valid_coords)
    centre_lon = sum(c[1] for c in valid_coords) / len(valid_coords)

    m = folium.Map(
        location=[centre_lat, centre_lon], zoom_start=13, tiles="CartoDB positron"
    )

    for e in _EDGES:
        a, b = e["from"], e["to"]
        if a not in _SITES or b not in _SITES:
            continue
        lat_a, lon_a = _SITES[a]["latitude"], _SITES[a]["longitude"]
        lat_b, lon_b = _SITES[b]["latitude"], _SITES[b]["longitude"]
        if None in (lat_a, lon_a, lat_b, lon_b):
            continue
        on_route = (a, b) in path_pairs or (b, a) in path_pairs
        roads_str = (
            ", ".join(e["roads"]) if isinstance(e["roads"], list) else e["roads"]
        )
        folium.PolyLine(
            locations=[[lat_a, lon_a], [lat_b, lon_b]],
            color="#e74c3c" if on_route else "#aaaaaa",
            weight=6 if on_route else 2,
            opacity=0.95 if on_route else 0.4,
            tooltip=f"{a} → {b}  {roads_str}",
        ).add_to(m)

    for site_id, site in sorted(_SITES.items()):
        lat, lon = site["latitude"], site["longitude"]
        if lat is None or lon is None:
            continue
        on_route = site_id in path_set
        is_origin = site_id == site_path[0]
        is_dest = site_id == site_path[-1]
        colour = (
            "green"
            if is_origin
            else "red" if is_dest else "#e74c3c" if on_route else "steelblue"
        )
        radius = 11 if (is_origin or is_dest) else (8 if on_route else 5)
        opacity = 1.0 if on_route else 0.45
        roads_str = ", ".join(sorted(site.get("roads", [])))
        popup_parts = [f"<b>Site {site_id}</b>", f"Roads: {roads_str}"]
        if on_route:
            idx = site_path.index(site_id)
            label = (
                "Origin"
                if is_origin
                else "Destination" if is_dest else f"Step {idx + 1}"
            )
            popup_parts.append(f"<b>{label}</b>")
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=opacity,
            opacity=opacity,
            tooltip=site_id + (" ★" if on_route else ""),
            popup=folium.Popup("<br>".join(popup_parts), max_width=260),
        ).add_to(m)

    cost_str = (
        f"{cost:.1f} s ({cost/60:.1f} min)"
        if cost_fn == "quadratic"
        else f"{cost:.1f} units ({cost/60:.2f} km-equiv)"
    )
    m.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;top:12px;left:56px;z-index:1000;
        background:white;padding:8px 14px;border-radius:6px;
        border:1px solid #bbb;font-family:sans-serif;font-size:14px;
        box-shadow:2px 2px 6px rgba(0,0,0,.2);">
        <b>Route: {site_path[0]} → {site_path[-1]}</b>
        &nbsp;|&nbsp; {method} &nbsp;|&nbsp; {cost_str}
    </div>"""))
    m.get_root().html.add_child(folium.Element("""
    <div style="position:fixed;bottom:30px;left:56px;z-index:1000;
        background:white;padding:8px 14px;border-radius:6px;
        border:1px solid #bbb;font-family:sans-serif;font-size:12px;
        box-shadow:2px 2px 6px rgba(0,0,0,.2);">
        <span style="color:green;font-size:16px;">&#9679;</span> Origin &nbsp;
        <span style="color:red;font-size:16px;">&#9679;</span> Destination &nbsp;
        <span style="color:#e74c3c;">&#9135;&#9135;</span> Route &nbsp;
        <span style="color:steelblue;font-size:16px;">&#9679;</span> Other site
    </div>"""))

    m.save(str(_ROUTE_HTML))
    return _ROUTE_HTML


# ── Matplotlib inline map ──────────────────────────────────────────────────────


def _draw_map(ax, site_path: list | None = None):
    """Draw all edges and sites on ax; highlight route if site_path given."""
    ax.clear()
    ax.set_facecolor("#f5f5f5")

    path_set = set(site_path) if site_path else set()
    path_pairs = set(zip(site_path, site_path[1:])) if site_path else set()

    # Edges
    for e in _EDGES:
        a, b = e["from"], e["to"]
        if a not in _SITES or b not in _SITES:
            continue
        lon_a, lat_a = _SITES[a]["longitude"], _SITES[a]["latitude"]
        lon_b, lat_b = _SITES[b]["longitude"], _SITES[b]["latitude"]
        if None in (lat_a, lon_a, lat_b, lon_b):
            continue
        on_route = (a, b) in path_pairs or (b, a) in path_pairs
        ax.plot(
            [lon_a, lon_b],
            [lat_a, lat_b],
            color="#e74c3c" if on_route else "#cccccc",
            linewidth=2.5 if on_route else 0.8,
            zorder=2 if on_route else 1,
            solid_capstyle="round",
        )

    # Sites
    for site_id, site in _SITES.items():
        lon, lat = site["longitude"], site["latitude"]
        if lon is None or lat is None:
            continue
        is_origin = site_path and site_id == site_path[0]
        is_dest = site_path and site_id == site_path[-1]
        on_route = site_id in path_set

        if is_origin:
            color, size, zorder = "green", 90, 5
        elif is_dest:
            color, size, zorder = "red", 90, 5
        elif on_route:
            color, size, zorder = "#e74c3c", 50, 4
        else:
            color, size, zorder = "steelblue", 12, 3

        ax.scatter(
            lon,
            lat,
            s=size,
            c=color,
            zorder=zorder,
            edgecolors="white" if on_route else "none",
            linewidths=0.8 if on_route else 0,
        )

        if on_route:
            ax.annotate(
                site_id,
                (lon, lat),
                fontsize=6.5,
                ha="center",
                va="bottom",
                xytext=(0, 5),
                textcoords="offset points",
                zorder=6,
                fontweight="bold",
            )

    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)

    if site_path:
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="green",
                markersize=9,
                label="Origin",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=9,
                label="Destination",
            ),
            Line2D([0], [0], color="#e74c3c", linewidth=2, label="Route"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="steelblue",
                markersize=6,
                label="Other site",
            ),
        ]
        ax.legend(handles=legend_elements, loc="lower left", fontsize=7, framealpha=0.9)
    else:
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="steelblue",
                markersize=7,
                label="SCATS site",
            ),
            Line2D([0], [0], color="#cccccc", linewidth=1.5, label="Road segment"),
        ]
        ax.legend(handles=legend_elements, loc="lower left", fontsize=7, framealpha=0.9)

    ax.set_aspect("equal", adjustable="datalim")


# ── Search logic ───────────────────────────────────────────────────────────────


def _run_search(
    origin_id,
    dest_ids,
    method,
    cost_fn,
    query_time_str,
    prediction_model,
    output_widget,
    run_btn,
    on_routes_ready,
):

    def write(text=""):
        output_widget.configure(state="normal")
        output_widget.insert(tk.END, text + "\n")
        output_widget.see(tk.END)
        output_widget.configure(state="disabled")

    try:
        wide_df = pd.read_parquet(WIDE_FILE)
        query_time = (
            pd.Timestamp(query_time_str) if query_time_str else wide_df.index[-1]
        )

        if not query_time_str:
            write(f"No time given; using last timestamp: {query_time}")

        write(f"Loading prediction model: {prediction_model}")

        if prediction_model == "GRU":
            model_result = load_model(MODEL_FILE)
            if model_result is None:
                write(f"WARNING: {MODEL_FILE.name} not found — using distance-only costs.")
                volume_map = {}
            else:
                model, device = model_result
                write(f"GRU model loaded on {device}. Rebuilding scalers…")
                scalers = rebuild_scalers(wide_df)
                write(f"  {len(scalers)} approaches scaled.")
                write(f"Predicting volumes at {query_time}…")
                volume_map = predict_volumes(model, device, scalers, wide_df, query_time)
        elif prediction_model == "XGBoost":
            xgb_models = load_xgb_models(XGB_MODEL_FILE)
            if xgb_models is None:
                write(f"WARNING: {XGB_MODEL_FILE.name} not found — using distance-only costs.")
                volume_map = {}
            else:
                write(f"XGBoost models loaded: {len(xgb_models)}")
                write(f"Predicting volumes at {query_time}…")
                volume_map = predict_volumes_xgb(models=xgb_models, wide_df=wide_df, query_time=query_time)
        elif prediction_model == "LSTM":
            model_result = load_model_lstm(LSTM_MODEL_FILE)

            if model_result is None:
                write(f"WARNING: {LSTM_MODEL_FILE.name} not found — using distance-only costs.")
                volume_map = {}
            else:
                model, device = model_result
                write(f"LSTM model loaded on {device}. Rebuilding scalers…")
                scalers = rebuild_scalers_lstm(wide_df)
                write(f"  {len(scalers)} approaches scaled.")
                write(f"Predicting volumes at {query_time}…")
                volume_map = predict_volumes_lstm(model, device, scalers, wide_df, query_time)
        else:
            write(f"Unknown prediction model '{prediction_model}' — using distance-only costs.")
            volume_map = {}

        predicted = len(volume_map)
        avg_vol = sum(volume_map.values()) / predicted if predicted else 0
        write(f"{predicted} predictions  (avg {avg_vol:.0f} veh/15 min)")

        problem = build_problem(
            origin_id, dest_ids, _SITES, _EDGES, volume_map, cost_fn=cost_fn
        )

        write(f"\nFinding top-5 routes via {method} ({cost_fn}) @ {query_time} …")
        raw_routes = yen_k_shortest(problem, k=5, solver=METHODS[method])

        if not raw_routes:
            write("No path found between those sites.")
            return

        # Convert int node IDs → zero-padded site ID strings
        routes: list[tuple[list[str], float]] = [
            ([str(n).zfill(4) for n in path], cost)
            for path, cost in raw_routes
        ]

        write(f"\n── Top {len(routes)} Routes " + "─" * 44)
        for i, (site_path, cost) in enumerate(routes, 1):
            hops = []
            for j in range(len(site_path) - 1):
                a, b = site_path[j], site_path[j + 1]
                road = _EDGE_LOOKUP.get((a, b)) or _EDGE_LOOKUP.get((b, a)) or "?"
                hops.append(f"    {a} -> {b}  [{road}]")
            if cost_fn == "quadratic":
                cost_str = f"{cost:.1f} s  ({cost / 60:.1f} min)"
            else:
                cost_str = f"{cost:.1f} units  ({cost / 60:.2f} km-equiv)"
            write(f"\nRoute {i}: {site_path[0]} → {site_path[-1]}  |  {cost_str}")
            write("\n".join(hops))

        output_widget.after(0, lambda r=routes: on_routes_ready(r, method, cost_fn))

    except Exception as exc:
        import traceback
        write(f"\nERROR: {exc}")
        write(traceback.format_exc())
    finally:
        output_widget.after(
            0, lambda: run_btn.configure(state="normal", text="Run Search")
        )


# ── GUI ────────────────────────────────────────────────────────────────────────


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Boroondara SCATS Route Search")
        self.geometry("1150x820")
        self.resizable(True, True)
        self._last_site_path = None
        self._last_cost = None
        self._last_method = None
        self._last_cost_fn = None
        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        nb = ttk.Notebook(self)
        nb.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self._notebook = nb

        search_tab = ttk.Frame(nb)
        map_tab = ttk.Frame(nb)
        nb.add(search_tab, text="  Search  ")
        nb.add(map_tab, text="  Map  ")

        self._build_search_tab(search_tab)
        self._build_map_tab(map_tab)

    # ── Search tab ─────────────────────────────────────────────────────────────

    def _build_search_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)
        parent.rowconfigure(3, weight=0)
        pad = {"padx": 8, "pady": 4}

        pf = ttk.LabelFrame(parent, text="Search Parameters")
        pf.grid(row=0, column=0, sticky="ew", padx=10, pady=8)

        ttk.Label(pf, text="Origin site ID:").grid(row=0, column=0, sticky="w", **pad)
        self._origin_var = tk.StringVar(value=_SITE_IDS[0])
        ttk.Combobox(
            pf,
            textvariable=self._origin_var,
            values=_SITE_IDS,
            state="readonly",
            width=10,
        ).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(pf, text="Destination(s):").grid(row=1, column=0, sticky="w", **pad)
        self._dest_var = tk.StringVar(value=_SITE_IDS[-1])
        ttk.Combobox(pf, textvariable=self._dest_var, values=_SITE_IDS, width=18).grid(
            row=1, column=1, sticky="w", **pad
        )
        ttk.Label(pf, text="(comma-separated for multi-dest)", foreground="grey").grid(
            row=1, column=2, sticky="w", **pad
        )

        ttk.Label(pf, text="Method:").grid(row=2, column=0, sticky="w", **pad)
        self._method_var = tk.StringVar(value="AS")
        ttk.Combobox(
            pf,
            textvariable=self._method_var,
            values=_METHODS,
            state="readonly",
            width=10,
        ).grid(row=2, column=1, sticky="w", **pad)

        ttk.Label(pf, text="Prediction model:").grid(row=5, column=0, sticky="w", **pad)
        self._prediction_model_var = tk.StringVar(value="GRU")

        ttk.Combobox(
            pf,
            textvariable=self._prediction_model_var,
            values=["GRU", "XGBoost", "LSTM"],
            state="readonly",
            width=14,
        ).grid(row=5, column=1, sticky="w", **pad)

        ttk.Label(pf, text="Cost function:").grid(row=3, column=0, sticky="w", **pad)
        self._cost_var = tk.StringVar(value=_COSTS[0])
        ttk.Combobox(
            pf, textvariable=self._cost_var, values=_COSTS, state="readonly", width=14
        ).grid(row=3, column=1, sticky="w", **pad)

        ttk.Label(pf, text="Time:").grid(row=4, column=0, sticky="w", **pad)

        time_frame = ttk.Frame(pf)
        time_frame.grid(row=4, column=1, columnspan=2, sticky="w", **pad)

        self._use_last_ts = tk.BooleanVar(value=True)
        self._last_ts_chk = ttk.Checkbutton(
            time_frame,
            text="Use last timestamp in dataset",
            variable=self._use_last_ts,
            command=self._on_timestamp_toggle,
        )
        self._last_ts_chk.pack(side="left")

        self._date_entry = DateEntry(
            time_frame,
            width=12,
            date_pattern="yyyy-mm-dd",
            year=2006, month=10, day=31,
            state="disabled",
        )
        self._date_entry.pack(side="left", padx=(10, 2))

        self._hour_var = tk.StringVar(value="08")
        self._min_var = tk.StringVar(value="00")

        self._hour_spinbox = ttk.Spinbox(
            time_frame, from_=0, to=23, width=3, format="%02.0f",
            textvariable=self._hour_var, state="disabled", wrap=True,
        )
        self._hour_spinbox.pack(side="left")
        ttk.Label(time_frame, text=":").pack(side="left")
        self._min_spinbox = ttk.Spinbox(
            time_frame, from_=0, to=45, increment=15, width=3, format="%02.0f",
            textvariable=self._min_var, state="disabled", wrap=True,
        )
        self._min_spinbox.pack(side="left")

        bf = ttk.Frame(parent)
        bf.grid(row=1, column=0, sticky="ew", padx=10, pady=4)
        self._run_btn = ttk.Button(bf, text="Run Search", command=self._on_run)
        self._run_btn.pack(side="left", padx=4)
        ttk.Button(bf, text="Clear Output", command=self._clear_output).pack(
            side="left", padx=4
        )

        of = ttk.LabelFrame(parent, text="Output")
        of.grid(row=2, column=0, sticky="nsew", padx=10, pady=8)
        of.columnconfigure(0, weight=1)
        of.rowconfigure(0, weight=1)
        self._output = scrolledtext.ScrolledText(
            of, state="disabled", wrap="word", font=("Consolas", 10)
        )
        self._output.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        # ── Routes panel ───────────────────────────────────────────────────────
        rf = ttk.LabelFrame(parent, text="Top Routes  (click to view on map)")
        rf.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 8))
        rf.columnconfigure(0, weight=1)

        self._routes_lb = tk.Listbox(
            rf, height=5, font=("Consolas", 9), activestyle="dotbox",
            selectmode="browse",
        )
        self._routes_lb.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        self._routes_lb.bind("<<ListboxSelect>>", self._on_route_select)

        self._routes: list[tuple[list[str], float]] = []

    # ── Map tab ────────────────────────────────────────────────────────────────

    def _build_map_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        tb = ttk.Frame(parent)
        tb.grid(row=0, column=0, sticky="ew", padx=10, pady=6)
        ttk.Button(tb, text="Show full graph", command=self._show_base_map).pack(
            side="left", padx=4
        )
        ttk.Button(tb, text="Open in browser", command=self._open_in_browser).pack(
            side="left", padx=4
        )
        self._map_status = tk.StringVar(
            value="Run a search to see the route highlighted."
        )
        ttk.Label(tb, textvariable=self._map_status, foreground="grey").pack(
            side="left", padx=12
        )

        # Matplotlib figure embedded in tkinter
        self._fig, self._ax = plt.subplots(figsize=(10, 6), dpi=100)
        self._fig.tight_layout(pad=1.5)

        canvas_frame = ttk.Frame(parent)
        canvas_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 4))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self._canvas = FigureCanvasTkAgg(self._fig, master=canvas_frame)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.grid(row=2, column=0, sticky="ew", padx=10)
        NavigationToolbar2Tk(self._canvas, toolbar_frame)

        # Draw the base graph on startup
        _draw_map(self._ax, site_path=None)
        self._canvas.draw()

    # ── Map actions ────────────────────────────────────────────────────────────

    def _show_base_map(self):
        _draw_map(self._ax, site_path=None)
        self._canvas.draw()
        self._map_status.set("Showing full graph.")

    def _open_in_browser(self):
        if self._last_site_path and _ROUTE_HTML.exists():
            webbrowser.open(_ROUTE_HTML.as_uri())
        elif _GRAPH_HTML.exists():
            webbrowser.open(_GRAPH_HTML.as_uri())
        else:
            messagebox.showwarning(
                "Not found", "No HTML map found. Run visualise_graph.py first."
            )

    def _on_routes_ready(self, routes: list[tuple[list[str], float]], method: str, cost_fn: str):
        self._routes = routes
        self._last_method = method
        self._last_cost_fn = cost_fn

        self._routes_lb.delete(0, tk.END)
        for i, (site_path, cost) in enumerate(routes):
            if cost_fn == "quadratic":
                cost_str = f"{cost / 60:.1f} min"
            else:
                cost_str = f"{cost:.1f} units"
            label = f"{'★ ' if i == 0 else f'{i+1}. '}  {site_path[0]} → {site_path[-1]}   {len(site_path)} stops   {cost_str}"
            self._routes_lb.insert(tk.END, label)

        self._routes_lb.selection_set(0)
        self._show_route(0)

    def _on_route_select(self, _event):
        sel = self._routes_lb.curselection()
        if sel:
            self._show_route(sel[0])

    def _show_route(self, idx: int):
        if not self._routes or idx >= len(self._routes):
            return
        site_path, cost = self._routes[idx]
        method = self._last_method
        cost_fn = self._last_cost_fn

        self._last_site_path = site_path
        self._last_cost = cost

        _draw_map(self._ax, site_path=site_path)
        if cost_fn == "quadratic":
            cost_str = f"{cost:.1f} s ({cost/60:.1f} min)"
        else:
            cost_str = f"{cost:.1f} units ({cost/60:.2f} km-equiv)"
        self._ax.set_title(
            f"Route {idx+1}: {site_path[0]} → {site_path[-1]}  |  {cost_str}",
            fontsize=9,
        )
        self._canvas.draw()
        self._map_status.set(
            f"Route {idx+1}: {site_path[0]} → {site_path[-1]}  ({len(site_path)} stops, {cost_str})"
            "  — click 'Open in browser' for interactive map"
        )

        threading.Thread(
            target=_build_route_html,
            args=(site_path, method, cost, cost_fn),
            daemon=True,
        ).start()

        self._notebook.select(1)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _on_timestamp_toggle(self):
        state = "disabled" if self._use_last_ts.get() else "normal"
        self._date_entry.configure(state=state)
        self._hour_spinbox.configure(state=state)
        self._min_spinbox.configure(state=state)

    def _clear_output(self):
        self._output.configure(state="normal")
        self._output.delete("1.0", tk.END)
        self._output.configure(state="disabled")

    def _on_run(self):
        origin_raw = self._origin_var.get().strip()
        dest_raw = self._dest_var.get().strip()
        method = self._method_var.get().strip().upper()
        cost_fn = self._cost_var.get().strip()
        if self._use_last_ts.get():
            time_str = None
        else:
            date_str = self._date_entry.get_date().strftime("%Y-%m-%d")
            time_str = f"{date_str} {self._hour_var.get().zfill(2)}:{self._min_var.get().zfill(2)}"
        prediction_model = self._prediction_model_var.get().strip()

        if not origin_raw or not dest_raw:
            messagebox.showerror(
                "Input error", "Please set both an origin and destination."
            )
            return

        try:
            origin_id = _normalise_id(origin_raw)
            dest_ids = [_normalise_id(d) for d in dest_raw.split(",")]
        except ValueError:
            messagebox.showerror("Input error", "Site IDs must be numeric.")
            return

        for sid in [origin_id] + dest_ids:
            if sid not in _SITES:
                messagebox.showerror(
                    "Input error",
                    f"Site '{sid}' not found.\nFirst few valid IDs: {', '.join(_SITE_IDS[:8])} …",
                )
                return

        if method not in METHODS:
            messagebox.showerror(
                "Input error",
                f"Unknown method '{method}'. Choose from: {', '.join(_METHODS)}",
            )
            return

        self._clear_output()
        self._run_btn.configure(state="disabled", text="Running…")
        self._map_status.set("Running search…")

        threading.Thread(
            target=_run_search,
            args=(
                origin_id,
                dest_ids,
                method,
                cost_fn,
                time_str,
                prediction_model,
                self._output,
                self._run_btn,
                self._on_routes_ready,
            ),
            daemon=True,
        ).start()


if __name__ == "__main__":
    app = App()
    app.mainloop()
