"""
gui.py – PyQt6 GUI for the AI Search assignment.

Tab 1 – Run Search  : load a .txt from test_data (or browse), pick an
                      algorithm, run it, see results + embedded graph.
Tab 2 – Create File : build a problem file from scratch with a live preview.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QTabWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFileDialog,
    QMessageBox,
    QFrame,
    QSizePolicy,
    QSpacerItem,
)

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from assignment1.search import METHODS  # noqa: E402
from assignment1.helpers import parse_problem, visualise  # noqa: E402
from assignment1.models import Node, Edge, Problem  # noqa: E402

TEST_DATA_DIR = _HERE / "test_data"

# ── colour tokens (Catppuccin Mocha) ─────────────────────────────────────────
BG = "#1e1e2e"
PANEL = "#313244"
SURFACE = "#45475a"
FG = "#cdd6f4"
ACCENT = "#89b4fa"
GREEN = "#a6e3a1"
RED = "#f38ba8"
SUBTEXT = "#a6adc8"

# ── shared stylesheet ─────────────────────────────────────────────────────────
QSS = f"""
/* ── Base ── */
QWidget {{
    background-color: {BG};
    color: {FG};
    font-family: "Segoe UI";
    font-size: 10pt;
}}

/* ── Panels (slightly lighter background) ── */
QFrame#panel {{
    background-color: {PANEL};
    border-radius: 6px;
}}

/* ── Tab bar ── */
QTabWidget::pane {{
    border: none;
    background: {BG};
}}
QTabBar::tab {{
    background: {PANEL};
    color: {FG};
    padding: 8px 18px;
    margin-right: 2px;
    border-radius: 4px 4px 0 0;
    font-weight: bold;
}}
QTabBar::tab:selected {{
    background: {ACCENT};
    color: {BG};
}}
QTabBar::tab:hover:!selected {{
    background: {SURFACE};
}}

/* ── Buttons ── */
QPushButton {{
    background-color: {ACCENT};
    color: {BG};
    font-weight: bold;
    border: none;
    border-radius: 4px;
    padding: 6px 14px;
}}
QPushButton:hover  {{ background-color: {GREEN}; }}
QPushButton:pressed {{ background-color: {GREEN}; }}

QPushButton#danger {{
    background-color: {RED};
}}
QPushButton#danger:hover  {{ background-color: #eba0ac; }}
QPushButton#danger:pressed {{ background-color: #eba0ac; }}

/* ── Line edits & combos ── */
QLineEdit, QComboBox {{
    background-color: {SURFACE};
    color: {FG};
    border: 1px solid {SURFACE};
    border-radius: 4px;
    padding: 4px 6px;
}}
QLineEdit:focus, QComboBox:focus {{
    border: 1px solid {ACCENT};
}}
QComboBox QAbstractItemView {{
    background-color: {SURFACE};
    color: {FG};
    selection-background-color: {ACCENT};
    selection-color: {BG};
    border: none;
}}
QComboBox::drop-down {{ border: none; }}

/* ── Radio buttons ── */
QRadioButton {{
    color: {FG};
    spacing: 6px;
}}
QRadioButton::indicator {{
    width: 14px; height: 14px;
    border-radius: 7px;
    border: 2px solid {ACCENT};
    background: transparent;
}}
QRadioButton::indicator:checked {{
    background: {ACCENT};
}}

/* ── Text / result area ── */
QTextEdit {{
    background-color: {SURFACE};
    color: {GREEN};
    font-family: "Consolas";
    font-size: 9pt;
    border: none;
    border-radius: 4px;
    padding: 4px;
}}

/* ── Table ── */
QTableWidget {{
    background-color: {SURFACE};
    color: {FG};
    gridline-color: {PANEL};
    font-family: "Consolas";
    font-size: 9pt;
    border: none;
    border-radius: 4px;
}}
QTableWidget::item:selected {{
    background-color: {ACCENT};
    color: {BG};
}}
QHeaderView::section {{
    background-color: {PANEL};
    color: {ACCENT};
    font-family: "Segoe UI";
    font-size: 9pt;
    font-weight: bold;
    border: none;
    padding: 4px;
}}

/* ── Scrollbars ── */
QScrollBar:vertical {{
    background: {PANEL};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {SURFACE};
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

QScrollBar:horizontal {{
    background: {PANEL};
    height: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background: {SURFACE};
    border-radius: 4px;
    min-width: 20px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ── Separator line ── */
QFrame[frameShape="4"],
QFrame[frameShape="HLine"] {{
    color: {SURFACE};
    border: none;
    background: {SURFACE};
    max-height: 1px;
}}

/* ── Navigation toolbar (matplotlib) ── */
NavigationToolbar2QT {{
    background-color: {BG};
    border: none;
    spacing: 2px;
}}
NavigationToolbar2QT QToolButton {{
    background-color: transparent;
    color: {FG};
    border: none;
    border-radius: 3px;
    padding: 2px;
}}
NavigationToolbar2QT QToolButton:hover {{
    background-color: {SURFACE};
}}
"""


def _panel(parent: QWidget | None = None) -> QFrame:
    """Return a QFrame styled as a dark panel."""
    f = QFrame(parent)
    f.setObjectName("panel")
    return f


def _header(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {ACCENT}; font-weight: bold; font-size: 11pt;")
    return lbl


def _sep() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet(f"background: {SURFACE}; border: none; max-height: 1px;")
    return line


def _btn(text: str, danger: bool = False) -> QPushButton:
    b = QPushButton(text)
    if danger:
        b.setObjectName("danger")
    return b


def _table(columns: list[str]) -> QTableWidget:
    t = QTableWidget(0, len(columns))
    t.setHorizontalHeaderLabels([c.upper() for c in columns])
    t.verticalHeader().setVisible(False)
    t.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    t.setAlternatingRowColors(False)
    return t


# ──────────────────────────────────────────────────────────────────────────────
class SearchTab(QWidget):
    """Tab 1 – Run a search algorithm on a loaded file."""

    def __init__(self) -> None:
        super().__init__()
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)

        # ── Left control panel ────────────────────────────────────────
        ctrl = _panel()
        ctrl.setFixedWidth(300)
        ctrl_layout = QVBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(14, 14, 14, 14)
        ctrl_layout.setSpacing(6)

        # File selector
        ctrl_layout.addWidget(_header("Test File"))
        self._file_combo = QComboBox()
        self._file_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        ctrl_layout.addWidget(self._file_combo)
        self._refresh_file_list()

        btn_row = QHBoxLayout()
        refresh_btn = _btn("Refresh")
        browse_btn = _btn("Browse…")
        refresh_btn.clicked.connect(self._refresh_file_list)
        browse_btn.clicked.connect(self._browse_file)
        btn_row.addWidget(refresh_btn)
        btn_row.addWidget(browse_btn)
        ctrl_layout.addLayout(btn_row)

        ctrl_layout.addWidget(_sep())

        # Algorithm selector
        ctrl_layout.addWidget(_header("Algorithm"))
        self._algo_group = QButtonGroup(self)
        algo_info = [
            ("DFS", "Depth-First Search"),
            ("BFS", "Breadth-First Search"),
            ("GBFS", "Greedy Best-First"),
            ("AS", "A* Search"),
            ("CUS1", "Custom 1 – Bidirectional BFS"),
            ("CUS2", "Custom 2 – Weighted A*  (w=1.5)"),
        ]
        for key, desc in algo_info:
            rb = QRadioButton(f"{key:<5}  {desc}")
            rb.setProperty("algo_key", key)
            self._algo_group.addButton(rb)
            ctrl_layout.addWidget(rb)
            if key == "BFS":
                rb.setChecked(True)

        ctrl_layout.addWidget(_sep())

        run_btn = _btn("▶   Run Search")
        run_btn.setFixedHeight(36)
        run_btn.clicked.connect(self._run_search)
        ctrl_layout.addWidget(run_btn)

        ctrl_layout.addWidget(_sep())

        # Results
        ctrl_layout.addWidget(_header("Results"))
        self._result_box = QTextEdit()
        self._result_box.setReadOnly(True)
        self._result_box.setMinimumHeight(160)
        ctrl_layout.addWidget(self._result_box, stretch=1)

        root_layout.addWidget(ctrl)

        # ── Right graph panel ─────────────────────────────────────────
        graph_panel = QWidget()
        graph_layout = QVBoxLayout(graph_panel)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        graph_layout.setSpacing(0)

        self._fig = Figure(facecolor=BG)
        self._ax = self._fig.add_subplot(111, facecolor=BG)
        self._ax.axis("off")
        self._ax.text(
            0.5,
            0.5,
            "Load a file and run a search",
            transform=self._ax.transAxes,
            ha="center",
            va="center",
            color=SURFACE,
            fontsize=14,
        )

        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        toolbar = NavToolbar(self._canvas, graph_panel)

        graph_layout.addWidget(toolbar)
        graph_layout.addWidget(self._canvas, stretch=1)
        root_layout.addWidget(graph_panel, stretch=1)

    # ── file helpers ──────────────────────────────────────────────────
    def _refresh_file_list(self) -> None:
        current = self._file_combo.currentText()
        self._file_combo.clear()
        if TEST_DATA_DIR.exists():
            files = sorted(
                f.name for f in TEST_DATA_DIR.iterdir() if f.suffix == ".txt"
            )
            self._file_combo.addItems(files)
            if current in files:
                self._file_combo.setCurrentText(current)

    def _browse_file(self) -> None:
        init = str(TEST_DATA_DIR) if TEST_DATA_DIR.exists() else str(_HERE)
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a problem file", init, "Text files (*.txt);;All files (*)"
        )
        if not path:
            return
        p = Path(path)
        try:
            key = p.relative_to(TEST_DATA_DIR).name
        except ValueError:
            key = str(p)
        if self._file_combo.findText(key) == -1:
            self._file_combo.addItem(key)
        self._file_combo.setCurrentText(key)

    # ── run ───────────────────────────────────────────────────────────
    def _run_search(self) -> None:
        file_val = self._file_combo.currentText().strip()
        if not file_val:
            QMessageBox.warning(self, "No file selected", "Please select a test file.")
            return

        p = Path(file_val)
        file_path = p if p.is_absolute() else TEST_DATA_DIR / file_val
        if not file_path.exists():
            QMessageBox.critical(self, "File not found", f"Cannot find:\n{file_path}")
            return

        checked = self._algo_group.checkedButton()
        if not checked:
            QMessageBox.warning(self, "No algorithm", "Please select an algorithm.")
            return
        method = checked.property("algo_key")

        try:
            problem = parse_problem(str(file_path))
        except Exception as exc:
            QMessageBox.critical(self, "Parse error", str(exc))
            return

        try:
            result = METHODS[method](problem)
        except Exception as exc:
            QMessageBox.critical(self, "Search error", str(exc))
            return

        div = "─" * 28
        if result:
            path, cost, nodes_created = result
            lines = [
                f"File   : {file_val}",
                f"Method : {method}",
                div,
                f"Goal   : node {path[-1]}",
                f"Cost   : {cost}",
                f"Nodes  : {nodes_created} created",
                div,
                "Path:",
                "  " + " → ".join(str(n) for n in path),
            ]
        else:
            path = None
            lines = [
                f"File   : {file_val}",
                f"Method : {method}",
                div,
                "No path found.",
            ]
        self._result_box.setPlainText("\n".join(lines))

        self._ax.clear()
        self._ax.set_facecolor(BG)
        self._fig.set_facecolor(BG)
        visualise(problem, path, ax=self._ax, title=f"{file_val}   [{method}]")
        self._fig.tight_layout()
        self._canvas.draw()

    # called by CreateTab after saving so the combo stays current
    def refresh(self) -> None:
        self._refresh_file_list()


# ──────────────────────────────────────────────────────────────────────────────
class CreateTab(QWidget):
    """Tab 2 – Build a problem file with live graph preview."""

    def __init__(self, search_tab: SearchTab) -> None:
        super().__init__()
        self._search_tab = search_tab

        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Nodes panel ───────────────────────────────────────────────
        nodes_panel = _panel()
        nl = QVBoxLayout(nodes_panel)
        nl.setContentsMargins(12, 12, 12, 12)
        nl.setSpacing(6)

        nl.addWidget(_header("Nodes"))

        self._nodes_table = _table(["ID", "X", "Y"])
        nl.addWidget(self._nodes_table, stretch=1)

        ni = QHBoxLayout()
        self._node_id_e = QLineEdit()
        self._node_id_e.setPlaceholderText("ID")
        self._node_x_e = QLineEdit()
        self._node_x_e.setPlaceholderText("X")
        self._node_y_e = QLineEdit()
        self._node_y_e.setPlaceholderText("Y")
        for w in (self._node_id_e, self._node_x_e, self._node_y_e):
            ni.addWidget(w)
        nl.addLayout(ni)

        nb = QHBoxLayout()
        add_node_btn = _btn("Add Node")
        rm_node_btn = _btn("Remove Selected", danger=True)
        add_node_btn.clicked.connect(self._add_node)
        rm_node_btn.clicked.connect(lambda: self._remove_rows(self._nodes_table))
        nb.addWidget(add_node_btn)
        nb.addWidget(rm_node_btn)
        nl.addLayout(nb)

        nl.addWidget(_sep())
        nl.addWidget(_header("Origin / Destinations"))

        og = QGridLayout()
        og.setColumnStretch(1, 1)
        og.addWidget(QLabel("Origin:"), 0, 0)
        self._origin_e = QLineEdit()
        og.addWidget(self._origin_e, 0, 1)
        og.addWidget(QLabel("Destinations:"), 1, 0)
        self._dest_e = QLineEdit()
        self._dest_e.setPlaceholderText("e.g.  4 ; 5")
        og.addWidget(self._dest_e, 1, 1)
        sub = QLabel("semicolon-separated")
        sub.setStyleSheet(f"color: {SUBTEXT}; font-size: 8pt;")
        og.addWidget(sub, 2, 1)
        nl.addLayout(og)

        self._origin_e.textChanged.connect(lambda _: self._update_preview())
        self._dest_e.textChanged.connect(lambda _: self._update_preview())

        root.addWidget(nodes_panel, stretch=1)

        # ── Edges panel ───────────────────────────────────────────────
        edges_panel = _panel()
        el = QVBoxLayout(edges_panel)
        el.setContentsMargins(12, 12, 12, 12)
        el.setSpacing(6)

        el.addWidget(_header("Edges"))

        self._edges_table = _table(["Start", "End", "Cost"])
        el.addWidget(self._edges_table, stretch=1)

        ei = QHBoxLayout()
        self._edge_start_e = QLineEdit()
        self._edge_start_e.setPlaceholderText("Start")
        self._edge_end_e = QLineEdit()
        self._edge_end_e.setPlaceholderText("End")
        self._edge_cost_e = QLineEdit()
        self._edge_cost_e.setPlaceholderText("Cost")
        for w in (self._edge_start_e, self._edge_end_e, self._edge_cost_e):
            ei.addWidget(w)
        el.addLayout(ei)

        eb = QHBoxLayout()
        add_edge_btn = _btn("Add Edge")
        rm_edge_btn = _btn("Remove Selected", danger=True)
        add_edge_btn.clicked.connect(self._add_edge)
        rm_edge_btn.clicked.connect(lambda: self._remove_rows(self._edges_table))
        eb.addWidget(add_edge_btn)
        eb.addWidget(rm_edge_btn)
        el.addLayout(eb)

        el.addWidget(_sep())
        el.addWidget(_header("Save"))

        sg = QGridLayout()
        sg.setColumnStretch(1, 1)
        sg.addWidget(QLabel("Filename:"), 0, 0)
        self._fname_e = QLineEdit("my_problem.txt")
        sg.addWidget(self._fname_e, 0, 1)
        save_btn = _btn("Save File")
        save_btn.clicked.connect(self._save_file)
        sg.addWidget(save_btn, 1, 0, 1, 2)
        el.addLayout(sg)

        root.addWidget(edges_panel, stretch=1)

        # ── Live preview panel ────────────────────────────────────────
        preview_panel = _panel()
        pl = QVBoxLayout(preview_panel)
        pl.setContentsMargins(12, 12, 12, 12)
        pl.setSpacing(6)

        pl.addWidget(_header("Live Preview"))

        self._prev_fig = Figure(facecolor=BG)
        self._prev_ax = self._prev_fig.add_subplot(111, facecolor=BG)
        self._prev_ax.axis("off")
        self._prev_ax.text(
            0.5,
            0.5,
            "Add nodes to see a preview",
            transform=self._prev_ax.transAxes,
            ha="center",
            va="center",
            color=SURFACE,
            fontsize=13,
        )

        self._prev_canvas = FigureCanvas(self._prev_fig)
        self._prev_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        pl.addWidget(self._prev_canvas, stretch=1)

        root.addWidget(preview_panel, stretch=2)

    # ── table helpers ─────────────────────────────────────────────────
    def _add_node(self) -> None:
        try:
            nid = int(self._node_id_e.text())
            x = int(self._node_x_e.text())
            y = int(self._node_y_e.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid input", "ID, X and Y must be integers.")
            return
        for row in range(self._nodes_table.rowCount()):
            if int(self._nodes_table.item(row, 0).text()) == nid:
                QMessageBox.warning(self, "Duplicate", f"Node ID {nid} already exists.")
                return
        r = self._nodes_table.rowCount()
        self._nodes_table.insertRow(r)
        for col, val in enumerate([nid, x, y]):
            item = QTableWidgetItem(str(val))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._nodes_table.setItem(r, col, item)
        self._node_id_e.clear()
        self._node_x_e.clear()
        self._node_y_e.clear()
        self._update_preview()

    def _add_edge(self) -> None:
        try:
            start = int(self._edge_start_e.text())
            end = int(self._edge_end_e.text())
            cost = float(self._edge_cost_e.text())
        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid input",
                "Start/End must be integers; Cost must be a number.",
            )
            return
        display = int(cost) if cost == int(cost) else cost
        r = self._edges_table.rowCount()
        self._edges_table.insertRow(r)
        for col, val in enumerate([start, end, display]):
            item = QTableWidgetItem(str(val))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._edges_table.setItem(r, col, item)
        self._edge_start_e.clear()
        self._edge_end_e.clear()
        self._edge_cost_e.clear()
        self._update_preview()

    def _remove_rows(self, table: QTableWidget) -> None:
        rows = sorted({i.row() for i in table.selectedItems()}, reverse=True)
        for r in rows:
            table.removeRow(r)
        self._update_preview()

    # ── live preview ──────────────────────────────────────────────────
    def _update_preview(self) -> None:
        self._prev_ax.clear()
        self._prev_ax.set_facecolor(BG)
        self._prev_fig.set_facecolor(BG)

        node_rows = self._table_rows(self._nodes_table)
        if not node_rows:
            self._prev_ax.axis("off")
            self._prev_ax.text(
                0.5,
                0.5,
                "Add nodes to see a preview",
                transform=self._prev_ax.transAxes,
                ha="center",
                va="center",
                color=SURFACE,
                fontsize=13,
            )
            self._prev_canvas.draw()
            return

        nodes_dict: dict[int, Node] = {}
        for row in node_rows:
            nid, x, y = int(row[0]), int(row[1]), int(row[2])
            nodes_dict[nid] = Node(id=nid, coordinates=(x, y))

        for row in self._table_rows(self._edges_table):
            s, e, c = int(row[0]), int(row[1]), float(row[2])
            if s in nodes_dict and e in nodes_dict:
                nodes_dict[s].edges.append(Edge(start_node_id=s, end_node_id=e, cost=c))

        all_ids = list(nodes_dict.keys())

        try:
            origin = int(self._origin_e.text().strip())
            if origin not in nodes_dict:
                origin = all_ids[0]
        except ValueError:
            origin = all_ids[0]

        try:
            dests = [
                int(d.strip()) for d in self._dest_e.text().split(";") if d.strip()
            ]
            dests = [d for d in dests if d in nodes_dict]
        except ValueError:
            dests = []
        if not dests:
            dests = [all_ids[-1]] if len(all_ids) > 1 else [all_ids[0]]

        problem = Problem(
            nodes=list(nodes_dict.values()), origin=origin, destinations=dests
        )
        try:
            visualise(problem, None, ax=self._prev_ax, title="Live Preview")
        except Exception:
            self._prev_ax.axis("off")
            self._prev_ax.text(
                0.5,
                0.5,
                "Preview unavailable",
                transform=self._prev_ax.transAxes,
                ha="center",
                va="center",
                color=RED,
                fontsize=12,
            )

        self._prev_fig.tight_layout()
        self._prev_canvas.draw()

    # ── save ──────────────────────────────────────────────────────────
    def _save_file(self) -> None:
        nodes = self._table_rows(self._nodes_table)
        edges = self._table_rows(self._edges_table)
        origin_str = self._origin_e.text().strip()
        dest_str = self._dest_e.text().strip()
        fname = self._fname_e.text().strip()

        if not nodes:
            QMessageBox.warning(
                self, "No nodes", "Add at least one node before saving."
            )
            return
        if not origin_str:
            QMessageBox.warning(self, "No origin", "Enter an origin node ID.")
            return
        if not dest_str:
            QMessageBox.warning(
                self, "No destinations", "Enter at least one destination."
            )
            return
        if not fname:
            QMessageBox.warning(self, "No filename", "Enter a filename.")
            return
        if not fname.endswith(".txt"):
            fname += ".txt"
        try:
            int(origin_str)
        except ValueError:
            QMessageBox.warning(self, "Invalid origin", "Origin must be an integer.")
            return

        lines: list[str] = ["Nodes:"]
        for nid, x, y in nodes:
            lines.append(f"{nid}: ({x},{y})")
        lines += ["", "Edges:"]
        for start, end, cost in edges:
            lines.append(f"({start},{end}): {cost}")
        lines += ["", "Origin:", origin_str, "", "Destinations:", dest_str]
        content = "\n".join(lines) + "\n"

        init = str(TEST_DATA_DIR) if TEST_DATA_DIR.exists() else str(_HERE)
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save problem file",
            str(Path(init) / fname),
            "Text files (*.txt);;All files (*)",
        )
        if not save_path:
            return

        with open(save_path, "w", encoding="utf-8") as fh:
            fh.write(content)

        QMessageBox.information(self, "Saved", f"File saved:\n{save_path}")
        self._search_tab.refresh()

    # ── util ──────────────────────────────────────────────────────────
    @staticmethod
    def _table_rows(table: QTableWidget) -> list[list[str]]:
        rows = []
        for r in range(table.rowCount()):
            rows.append([table.item(r, c).text() for c in range(table.columnCount())])
        return rows


# ──────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI Search Visualiser")
        self.resize(1350, 840)
        self.setMinimumSize(960, 640)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)

        search_tab = SearchTab()
        create_tab = CreateTab(search_tab)

        tabs.addTab(search_tab, "  Run Search  ")
        tabs.addTab(create_tab, "  Create File  ")

        self.setCentralWidget(tabs)


# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
