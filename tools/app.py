import json
import math
import os
import traceback
from collections import Counter

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from agent.graph import build_graph

try:
    from agent.agentic import RouteRecorder, Router, build_agentic_graph, DEFAULT_MODEL
    AGENT_IMPORT_ERROR = None
except Exception as _exc:  # keep Standard mode working if agent mode can't load
    AGENT_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"
    DEFAULT_MODEL = "llama-3.3-70b-versatile"


# =========================================================
# 1. PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PrepLab | Intelligent Preprocessing",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 PrepLab")
st.caption(
    "Ask in plain language • an LLM routes the LangGraph nodes • "
    "see the exact route it took"
)


# =========================================================
# 2. SESSION STATE
# =========================================================

DEFAULTS = {
    "original_df": None,
    "processed_df": None,
    "processing_result": None,
    "processing_error": None,
    "processing_traceback": None,
    "run_notice": None,
    "run_done": False,
    "run_mode": "agent",
    "user_request": "",
    "running_node": None,
    "failed_node": None,
    "node_status": {},
    "node_details": {},
    "execution_trace": [],
    "route_log": [],
    "api_key": "",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# 3. NODE METADATA (real LangGraph nodes)
# =========================================================

NODE_INFO = {
    "problem_definition": {
        "title": "Problem Definition",
        "description": "Identify the ML task and relevant preprocessing context.",
        "tool": "problem_definition.py",
    },
    "inspect_data": {
        "title": "Inspect Data",
        "description": "Inspect dataset shape, data types, missing values, duplicates and cardinality.",
        "tool": "inspect_data.py",
    },
    "clean_columns": {
        "title": "Clean Columns",
        "description": "Standardize column names and handle columns requiring review.",
        "tool": "clean_columns.py",
    },
    "duplicate_handling": {
        "title": "Duplicate Handling",
        "description": "Find duplicate rows and remove them according to the configured policy.",
        "tool": "data_quality/duplicate_handling.py",
    },
    "type_correction": {
        "title": "Type Correction",
        "description": "Apply configured data type conversions.",
        "tool": "data_quality/type_correction.py",
    },
    "decision": {
        "title": "Decision Node",
        "description": "Choose the next action using the current workflow state.",
        "tool": "agent/nodes.py",
    },
    "check_missing_values": {
        "title": "Check Missing Values",
        "description": "Identify columns with missing values and classify them as categorical or numerical.",
        "tool": "agent/nodes.py",
    },
    "mode_imputation": {
        "title": "Mode Imputation",
        "description": "Fill categorical missing values using the mode.",
        "tool": "imputation/mode_imputation.py",
    },
    "check_bell_shaped": {
        "title": "Check Bell-Shaped Distribution",
        "description": "Determine which numerical columns are considered bell-shaped.",
        "tool": "outliers/check_bell_shaped.py",
    },
    "detect_outliers": {
        "title": "Detect Outliers",
        "description": "Use Z-score for bell-shaped columns and IQR for other columns.",
        "tool": "outliers/detect_outliers.py",
    },
    "median_imputation": {
        "title": "Median Imputation",
        "description": "Fill selected numerical columns with their median.",
        "tool": "imputation/median_imputation.py",
    },
    "outlier_capping": {
        "title": "Outlier Capping",
        "description": "Cap extreme values in selected columns using the configured capping tool.",
        "tool": "outliers/outlier_capping.py",
    },
    "check_skewness": {
        "title": "Check Skewness",
        "description": "Measure skewness in numerical columns to choose an imputation strategy.",
        "tool": "imputation/check_skewness.py",
    },
    "check_symmetry": {
        "title": "Check Symmetry",
        "description": "Evaluate symmetry. This node exists in the graph but is not currently selected by decision_node.",
        "tool": "imputation/check_symmetry.py",
    },
    "mean_imputation": {
        "title": "Mean Imputation",
        "description": "Fill selected numerical columns with their mean.",
        "tool": "imputation/mean_imputation.py",
    },
    "encoding_plan_ready": {
        "title": "Encoding Plan",
        "description": "Build the ordinal, frequency and one-hot encoding plan.",
        "tool": "agent/nodes.py",
    },
    "ordinal_encoding": {
        "title": "Ordinal Encoding",
        "description": "Apply explicitly supplied ordinal mappings.",
        "tool": "encoding/ordinal_encoding.py",
    },
    "frequency_encoding": {
        "title": "Frequency Encoding",
        "description": "Encode high-cardinality categorical columns.",
        "tool": "encoding/frequency_encoding.py",
    },
    "one_hot_encoding": {
        "title": "One-Hot Encoding",
        "description": "Encode remaining low-cardinality categorical columns.",
        "tool": "encoding/one_hot_encoding.py",
    },
    "feature_scaling": {
        "title": "Feature Scaling",
        "description": "Optionally scale numerical feature columns while excluding the target.",
        "tool": "agent/nodes.py",
    },
    "finish": {
        "title": "Finish",
        "description": "Mark the workflow complete and record the final dataset shape.",
        "tool": "agent/nodes.py",
    },
}

VALID_STATUS = {"pending", "running", "success", "failed", "skipped"}
STATUS_ALIASES = {
    "ok": "success",
    "done": "success",
    "completed": "success",
    "complete": "success",
    "error": "failed",
    "failure": "failed",
    "skip": "skipped",
}

NW, NH = 260, 76          # node width / height (shared by both canvases)


def cnode(cid, x, y, label, kind="real", refs=None, desc="", note=""):
    return {
        "id": cid, "x": x, "y": y, "label": label, "kind": kind,
        "refs": refs or [cid], "desc": desc, "note": note,
    }


# =========================================================
# 4. STANDARD MODE LAYOUT (the fixed branching logic)
#    Real nodes map 1:1 to LangGraph nodes. "alias" and "logic"
#    boxes exist only to keep the diagram readable.
# =========================================================

CANVAS_W, CANVAS_H = 2200, 2700

ALIAS_NOTE = (
    "Same LangGraph node as the other 'Decision Node' box. It is drawn twice "
    "so arrows stay readable; both boxes share the same recorded details."
)

CANVAS = [
    # Stage 1 - setup
    cnode("problem_definition", 190, 80, "Problem Definition"),
    cnode("inspect_data", 570, 80, "Inspect Data"),
    cnode("clean_columns", 950, 80, "Clean Columns"),
    cnode("duplicate_handling", 1330, 80, "Duplicate Handling"),
    cnode("type_correction", 1710, 80, "Type Correction"),
    # Stage 2 - missing values
    cnode("check_missing_values", 950, 270, "Check Missing Values"),
    cnode("decision_a", 500, 460, "Decision Node", "alias", ["decision"],
          note=ALIAS_NOTE),
    cnode("identify", 1400, 460, "Identify Missing Columns", "logic",
          ["check_missing_values"],
          desc="Collects the columns that contain missing values and splits them "
               "into categorical and numerical. Performed inside Check Missing Values.",
          note="Logic step drawn for clarity. It is not a separate LangGraph node; "
               "its status follows Check Missing Values."),
    cnode("mode_imputation", 1050, 650, "Mode Imputation"),
    cnode("check_bell_shaped", 1750, 650, "Choose Detection Method"),
    # Stage 3 - outliers & imputation
    cnode("iqr", 1450, 840, "IQR  ·  skewed / unknown", "logic",
          ["detect_outliers", "check_bell_shaped"],
          desc="IQR method: used for numerical columns that are not considered "
               "bell-shaped (skewed or unknown distribution).",
          note="Method branch drawn for clarity. Detect Outliers picks the method "
               "per column, so both IQR and Z-score can run in the same execution."),
    cnode("zscore", 1750, 840, "Z-score  ·  bell-shaped", "logic",
          ["detect_outliers", "check_bell_shaped"],
          desc="Z-score method: used for numerical columns considered bell-shaped.",
          note="Method branch drawn for clarity. Detect Outliers picks the method "
               "per column, so both IQR and Z-score can run in the same execution."),
    cnode("detect_outliers", 1600, 1030, "Detect Outliers"),
    cnode("median_a", 1100, 1220, "Median Imputation", "alias",
          ["median_imputation"],
          note="Median Imputation is one LangGraph node used by two branches. "
               "This box is the 'outliers found' branch (followed by Outlier Capping)."),
    cnode("check_skewness", 1650, 1220, "Check Skewness"),
    cnode("check_symmetry", 2000, 1220, "Check Symmetry (unused)"),
    cnode("outlier_capping", 1100, 1410, "Outlier Capping"),
    cnode("median_b", 1500, 1410, "Median Imputation", "alias",
          ["median_imputation"],
          note="Median Imputation is one LangGraph node used by two branches. "
               "This box is the 'skewed' branch."),
    cnode("mean_imputation", 1900, 1410, "Mean Imputation"),
    cnode("decision_b", 1000, 1600, "Decision Node", "alias", ["decision"],
          note=ALIAS_NOTE),
    # Stage 4 - encoding & scaling
    cnode("encoding_plan_ready", 1000, 1790, "Encoding Plan"),
    cnode("ordinal_encoding", 640, 1980, "Ordinal Encoding"),
    cnode("frequency_encoding", 1000, 1980, "Frequency Encoding"),
    cnode("one_hot_encoding", 1360, 1980, "One-Hot Encoding"),
    cnode("feature_scaling", 1000, 2170, "Feature Scaling (optional)"),
    cnode("finish", 1000, 2360, "Finish"),
    cnode("end", 1000, 2550, "END", "logic", ["finish"],
          desc="Workflow terminates after Finish.",
          note="Terminal marker drawn for clarity; not a separate LangGraph node."),
]

STAGES = [
    {"label": "STAGE 1  ·  SETUP", "y0": 24, "y1": 136},
    {"label": "STAGE 2  ·  MISSING VALUES", "y0": 196, "y1": 720},
    {"label": "STAGE 3  ·  OUTLIERS & IMPUTATION", "y0": 764, "y1": 1664},
    {"label": "STAGE 4  ·  ENCODING & SCALING", "y0": 1708, "y1": 2620},
]


def E(src, dst, label="", **kw):
    return {"src": src, "dst": dst, "label": label, **kw}


EDGES = [
    E("problem_definition", "inspect_data"),
    E("inspect_data", "clean_columns"),
    E("clean_columns", "duplicate_handling"),
    E("duplicate_handling", "type_correction"),
    E("type_correction", "check_missing_values", "via Decision Node"),
    E("check_missing_values", "decision_a", "No missing values"),
    E("check_missing_values", "identify", "Missing values found"),
    E("identify", "mode_imputation", "Categorical columns"),
    E("identify", "check_bell_shaped", "Numerical columns"),
    E("mode_imputation", "decision_a", dashed=True,
      pts=[(920, 650), (500, 650), (500, 498)]),
    E("check_bell_shaped", "iqr", "Skewed / unknown"),
    E("check_bell_shaped", "zscore", "Bell-shaped"),
    E("iqr", "detect_outliers"),
    E("zscore", "detect_outliers"),
    E("detect_outliers", "median_a", "Outliers found"),
    E("detect_outliers", "check_skewness", "No outliers", label_at=(1712, 1156)),
    E("median_a", "outlier_capping"),
    E("check_skewness", "median_b", "Skewed"),
    E("check_skewness", "mean_imputation", "Symmetric"),
    E("outlier_capping", "decision_b", dashed=True),
    E("median_b", "decision_b", dashed=True),
    E("mean_imputation", "decision_b", dashed=True),
    E("decision_a", "encoding_plan_ready", "Skip to encoding",
      pts=[(370, 460), (300, 460), (300, 1790), (870, 1790)]),
    E("decision_b", "encoding_plan_ready"),
    E("encoding_plan_ready", "ordinal_encoding"),
    E("ordinal_encoding", "frequency_encoding"),
    E("frequency_encoding", "one_hot_encoding"),
    E("one_hot_encoding", "feature_scaling"),
    E("feature_scaling", "finish"),
    E("finish", "end"),
]

POS = {c["id"]: (c["x"], c["y"]) for c in CANVAS}


def _dedupe(pts):
    out = []
    for p in pts:
        p = (float(p[0]), float(p[1]))
        if not out or out[-1] != p:
            out.append(p)
    return out


def _edge_geometry(e):
    sx, sy = POS[e["src"]]
    dx, dy = POS[e["dst"]]

    if e.get("pts"):
        pts = e["pts"]
    elif sy == dy:
        if dx > sx:
            pts = [(sx + NW / 2, sy), (dx - NW / 2, dy)]
        else:
            pts = [(sx - NW / 2, sy), (dx + NW / 2, dy)]
    elif sx == dx:
        pts = [(sx, sy + NH / 2), (dx, dy - NH / 2)]
    else:
        my = e.get("mid", (sy + dy) / 2)
        pts = [(sx, sy + NH / 2), (sx, my), (dx, my), (dx, dy - NH / 2)]

    pts = _dedupe(pts)

    if e.get("label_at"):
        lx, ly = e["label_at"]
    else:
        best, lx, ly = -1, pts[0][0], pts[0][1]
        for a, b in zip(pts, pts[1:]):
            length = math.hypot(b[0] - a[0], b[1] - a[1])
            if length > best:
                best = length
                lx, ly = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    return pts, lx, ly


EDGE_GEOM = {}
for _e in EDGES:
    _e["id"] = f'{_e["src"]}->{_e["dst"]}'
    EDGE_GEOM[_e["id"]] = _edge_geometry(_e)


# =========================================================
# 5. HELPERS: SAFE SERIALISATION & DETAIL CLASSIFICATION
# =========================================================

def safe(obj, depth=0):
    """Turn arbitrary node output into small, JSON-safe data."""
    if depth > 5:
        return "…"
    if isinstance(obj, pd.DataFrame):
        return {
            "type": "DataFrame",
            "shape": list(obj.shape),
            "columns": [str(c) for c in obj.columns][:40],
        }
    if isinstance(obj, pd.Series):
        return {"type": "Series", "name": str(obj.name), "length": int(len(obj))}
    if isinstance(obj, dict):
        return {str(k): safe(v, depth + 1) for k, v in list(obj.items())[:60]}
    if isinstance(obj, (list, tuple, set)):
        seq = list(obj)
        items = [safe(v, depth + 1) for v in seq[:60]]
        if len(seq) > 60:
            items.append(f"… +{len(seq) - 60} more")
        return items
    if obj is None or isinstance(obj, (bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, str):
        return obj if len(obj) <= 600 else obj[:600] + "…"
    if hasattr(obj, "tolist"):
        try:
            return safe(obj.tolist(), depth + 1)
        except Exception:
            pass
    if hasattr(obj, "item"):
        try:
            return safe(obj.item(), depth + 1)
        except Exception:
            pass
    return str(obj)[:300]


IGNORED_KEYS = {
    "completed_steps", "processing_history", "current_df", "original_df",
    "file_path", "step", "status", "timestamp", "history",
}

# First matching section wins. Only keys the node actually recorded are shown.
SECTION_RULES = [
    ("issues", ("error", "warning", "fail", "exception")),
    ("decision", ("method", "strategy", "decision", "next_action", "selected",
                  "chosen", "policy", "route")),
    ("reason", ("reason", "why", "rationale", "explanation", "skew", "bell",
                "threshold", "statistic", "score", "observed", "p_value")),
    ("output", ("output", "result", "outlier", "imput", "cap", "fill", "remov",
                "encod", "plan", "scal", "dropped", "converted", "mapping")),
    ("input", ("input", "column", "candidate", "target", "shape", "numeric",
               "numerical", "categorical", "missing", "feature")),
]


def classify(details):
    sections = {}
    for key, value in details.items():
        base = key.split(".")[-1].lower()
        if base in IGNORED_KEYS:
            continue
        for name, tokens in SECTION_RULES:
            if any(t in base for t in tokens):
                sections.setdefault(name, {})[key] = value
                break
    return sections


def norm_status(value):
    value = str(value or "success").lower()
    value = STATUS_ALIASES.get(value, value)
    return value if value in VALID_STATUS else "success"


def infer_running(node_name, update):
    """LangGraph streams completed nodes; infer which one starts next."""
    if node_name == "finish":
        return None
    if node_name == "decision":
        nxt = (update or {}).get("next_action")
        return nxt if nxt in NODE_INFO and nxt != "decision" else None
    return "decision"


def node_sections_and_raw(merged):
    return classify(merged), merged


# =========================================================
# 6A. STANDARD MODE PAYLOAD (fixed layout, live status)
# =========================================================

def build_payload(live=False):
    ss = st.session_state
    trace = ss.execution_trace
    running = ss.running_node if live else None

    counts = Counter(t["node"] for t in trace)
    order = {}
    for i, t in enumerate(trace, 1):
        order.setdefault(t["node"], i)

    status = {n: ss.node_status.get(n, "pending") for n in NODE_INFO}

    seen = Counter(counts)
    if running:
        seen[running] += 1

    def ex(n):
        return seen.get(n, 0) > 0

    cap = ex("outlier_capping")
    med = seen.get("median_imputation", 0)
    num_branch = ex("check_bell_shaped")
    missing_branch = ex("mode_imputation") or num_branch
    nomiss = ex("check_missing_values") and not missing_branch

    rules = {
        "identify": missing_branch,
        "iqr": ex("detect_outliers"),
        "zscore": ex("detect_outliers"),
        "median_a": cap,
        "median_b": med > (1 if cap else 0),
        "decision_a": ex("decision") and (nomiss or ex("mode_imputation")),
        "decision_b": ex("decision") and num_branch,
        "end": True,
    }

    def canvas_status(c):
        base = status[c["refs"][0]]
        if c["kind"] == "real":
            return base
        rule = rules.get(c["id"], True)
        if base == "running":
            if c["id"] == "decision_a":
                return "running" if not num_branch else "pending"
            if c["id"] == "decision_b":
                return "running" if num_branch else "pending"
            return "pending"
        if base == "failed":
            return "failed"
        if base == "success":
            if rule:
                return "success"
            return "skipped" if ss.run_done else "pending"
        return base

    cstat = {c["id"]: canvas_status(c) for c in CANVAS}

    special = {
        "check_missing_values->decision_a": nomiss,
        "decision_a->encoding_plan_ready": (
            cstat["decision_a"] == "success" and ex("encoding_plan_ready")
            and not num_branch
        ),
        "decision_b->encoding_plan_ready": (
            cstat["decision_b"] == "success" and ex("encoding_plan_ready")
        ),
    }

    edges_out = []
    for e in EDGES:
        pts, lx, ly = EDGE_GEOM[e["id"]]
        if e["id"] in special:
            done = bool(special[e["id"]])
        else:
            done = cstat[e["src"]] == "success" and cstat[e["dst"]] == "success"
        if done:
            state = "done"
        elif cstat[e["dst"]] == "running" and cstat[e["src"]] == "success":
            state = "active"
        else:
            state = "idle"
        edges_out.append({
            "id": e["id"], "from": e["src"], "to": e["dst"],
            "label": e["label"], "pts": pts, "lx": lx, "ly": ly,
            "dashed": bool(e.get("dashed")), "state": state,
        })

    nodes_out = []
    for c in CANVAS:
        refs = c["refs"]
        merged = {}
        for r in refs:
            for k, v in ss.node_details.get(r, {}).items():
                merged[k if len(refs) == 1 else f"{r}.{k}"] = v

        info = NODE_INFO.get(refs[0], {})
        if c["kind"] == "real":
            title = info.get("title", c["label"])
            desc = info.get("description", "")
            tool = info.get("tool", "")
            note = f'LangGraph node · runs {info.get("tool", "")}'
            sub = c["id"]
        else:
            title = c["label"]
            desc = c["desc"] or info.get("description", "")
            tool = info.get("tool", "") if c["kind"] == "alias" else ""
            note = c["note"]
            sub = "decision (hub)" if c["kind"] == "alias" and refs[0] == "decision" \
                else (refs[0] if c["kind"] == "alias" else "logic step")

        sections, raw = node_sections_and_raw(merged)
        nodes_out.append({
            "id": c["id"], "x": c["x"], "y": c["y"], "label": c["label"],
            "sub": sub, "kind": c["kind"], "status": cstat[c["id"]],
            "title": title, "desc": desc, "tool": tool, "note": note,
            "runs": counts.get(refs[0], 0) if c["kind"] != "logic" else None,
            "order": order.get(refs[0]) if c["kind"] == "real" else None,
            "sections": sections, "raw": raw,
        })

    def to_canvas(name):
        if not name:
            return None
        if name == "decision":
            return "decision_b" if num_branch else "decision_a"
        return name

    last = trace[-1]["node"] if trace else None
    return {
        "W": CANVAS_W, "H": CANVAS_H, "NW": NW, "NH": NH,
        "title": "Pipeline map", "subtitle": "",
        "key": "preplab_flow_v2",
        "footnote": ("This illustrates the requested branching logic, not a verified rendering "
                     "of the current executable graph. Colours and details come from the actual run."),
        "stages": STAGES, "nodes": nodes_out, "edges": edges_out,
        "live": bool(live),
        "running": to_canvas(running),
        "last": to_canvas(last),
        "counts": {s: sum(1 for v in status.values() if v == s) for s in VALID_STATUS},
    }


# =========================================================
# 6B. AGENT MODE PAYLOAD (the route the router actually took)
#     Every executed node in order. Each router decision is a circle on the
#     connector between two nodes. Unused tools sit below.
# =========================================================

AGENT_COLS = 5
AGENT_PX, AGENT_PY = 420, 215
AGENT_X0, AGENT_Y0 = 210, 170
AGENT_W = AGENT_X0 * 2 + (AGENT_COLS - 1) * AGENT_PX
CIRCLE_R = 25
PALETTE_PY = 112


def _slot(i):
    """Serpentine layout: row 0 left->right, row 1 right->left, ..."""
    r, c = divmod(i, AGENT_COLS)
    cc = c if r % 2 == 0 else AGENT_COLS - 1 - c
    return AGENT_X0 + cc * AGENT_PX, AGENT_Y0 + r * AGENT_PY


def _connector(i, j):
    (xa, ya), (xb, yb) = _slot(i), _slot(j)
    if ya == yb:
        d = 1 if xb > xa else -1
        return (xa + d * NW / 2, ya), (xb - d * NW / 2, yb)
    return (xa, ya + NH / 2), (xb, yb - NH / 2)


def _agent_edges(eid, src, dst, p, q, circle=None, circle_id=None,
                 dst_status="pending", circle_status="success", dashed=False):
    """Return edge dicts for src -> [circle] -> dst."""
    def seg_state(status):
        if status == "running":
            return "active"
        if status == "success":
            return "done"
        return "idle"

    out = []
    if circle is None:
        pts = _dedupe([p, q])
        out.append({"id": eid, "from": src, "to": dst, "label": "", "pts": pts,
                    "lx": 0, "ly": 0, "dashed": dashed, "state": seg_state(dst_status)})
        return out

    mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
    length = math.hypot(q[0] - p[0], q[1] - p[1]) or 1
    ux, uy = (q[0] - p[0]) / length, (q[1] - p[1]) / length
    a_end = (mx - ux * CIRCLE_R, my - uy * CIRCLE_R)
    b_start = (mx + ux * CIRCLE_R, my + uy * CIRCLE_R)
    out.append({"id": eid + "a", "from": src, "to": circle_id, "label": "",
                "pts": _dedupe([p, a_end]), "lx": 0, "ly": 0, "dashed": False,
                "state": seg_state(circle_status)})
    out.append({"id": eid + "b", "from": circle_id, "to": dst, "label": "",
                "pts": _dedupe([b_start, q]), "lx": 0, "ly": 0, "dashed": dashed,
                "state": seg_state(dst_status)})
    return out


def build_agent_payload(live=False):
    ss = st.session_state
    trace = ss.execution_trace
    log = ss.route_log
    running = ss.running_node if live else None
    failed = ss.failed_node

    status = {n: ss.node_status.get(n, "pending") for n in NODE_INFO}
    counts = Counter(t["node"] for t in trace if t["node"] != "decision")

    # ---- 1. steps = executed worker nodes (decisions become connector circles)
    steps = []
    pending_via = None
    dec_i = 0
    for t in trace:
        if t["node"] == "decision":
            pending_via = dec_i
            dec_i += 1
            continue
        steps.append({"node": t["node"], "via": pending_via, "state": "done",
                      "update": t.get("update", {})})
        pending_via = None

    if running and running != "decision":
        steps.append({"node": running, "via": pending_via, "state": "running", "update": {}})
    elif failed:
        steps.append({"node": failed, "via": pending_via, "state": "failed", "update": {}})

    ghost = (running == "decision") or (
        not steps and not ss.run_done and not failed
    )

    last_idx = {}
    for i, s in enumerate(steps):
        last_idx[s["node"]] = i

    # ---- 2. nodes + edges along the route
    nodes_out, edges_out = [], []
    ids = []

    for i, s in enumerate(steps):
        node = s["node"]
        sid = f"s{i + 1}"
        ids.append(sid)
        x, y = _slot(i)

        if s["state"] in ("running", "failed", "pending"):
            nstatus = s["state"]
        else:
            nstatus = status[node] if status[node] in ("failed", "skipped") else "success"

        merged = dict(s["update"])
        if last_idx[node] == i:
            merged = {**ss.node_details.get(node, {}), **merged}

        info = NODE_INFO.get(node, {})
        sections, raw = node_sections_and_raw(merged)
        nodes_out.append({
            "id": sid, "x": x, "y": y, "label": info.get("title", node),
            "sub": node, "kind": "real", "status": nstatus,
            "title": info.get("title", node),
            "desc": info.get("description", ""), "tool": info.get("tool", ""),
            "note": f"Step #{i + 1} of this run · LangGraph node · runs {info.get('tool', '')}",
            "runs": counts.get(node, 0),
            "order": i + 1 if s["state"] != "pending" else None,
            "sections": sections, "raw": raw,
        })

    step_status = {n["id"]: n["status"] for n in nodes_out}

    def router_node(k, x, y, pending=False, running_now=False):
        entry = log[k] if k < len(log) else None
        if entry is None:
            st_ = "running" if running_now else "pending"
            return {
                "id": f"r{k + 1}", "x": x, "y": y, "label": f"R{k + 1}" if not pending else "?",
                "sub": "router", "kind": "router", "status": st_, "source": "llm",
                "title": f"Router decision #{k + 1}",
                "desc": "The router is deciding which node to run next." if running_now
                        else "The router will choose the next node here.",
                "tool": "", "note": "", "runs": None, "order": None,
                "sections": {}, "raw": {},
            }
        chosen = entry.get("chosen", "?")
        chosen_title = NODE_INFO.get(chosen, {}).get("title", chosen)
        sections = {
            "input": {
                "user_request": ss.user_request or "(none)",
                "previous_node": entry.get("from_node"),
                "dataset_at_this_step": entry.get("dataset", {}),
                "nodes_it_could_choose": entry.get("allowed", []),
            },
            "decision": {
                "chosen_node": chosen,
                "decided_by": "LLM",
                **({"model": entry["model"]} if entry.get("model") else {}),
            },
            "reason": {"stated_reason": entry.get("reason", "")},
            "output": {"next_node": chosen, "latency_ms": entry.get("latency_ms")},
        }
        if entry.get("notes"):
            sections["issues"] = {"validation_notes": entry["notes"]}
        return {
            "id": f"r{k + 1}", "x": x, "y": y, "label": f"R{k + 1}",
            "sub": "router", "kind": "router", "status": "success",
            "source": entry.get("source", "llm"),
            "title": f"Router decision #{k + 1} → {chosen_title}",
            "desc": "The router (replacing the fixed Decision Node) chose which node runs next.",
            "tool": "",
            "note": "The reason is the model's own stated rationale for the choice; it is "
                    "recorded, not independently verified.",
            "runs": None, "order": None, "sections": sections, "raw": entry,
        }

    for i in range(1, len(steps)):
        p, q = _connector(i - 1, i)
        via = steps[i]["via"]
        sid_a, sid_b = ids[i - 1], ids[i]
        if via is not None:
            mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
            rn = router_node(via, mx, my)
            nodes_out.append(rn)
            edges_out += _agent_edges(
                f"e{i}", sid_a, sid_b, p, q, circle=True, circle_id=rn["id"],
                dst_status=step_status[sid_b], circle_status=rn["status"])
        else:
            edges_out += _agent_edges(
                f"e{i}", sid_a, sid_b, p, q, dst_status=step_status[sid_b])

    # ---- 3. ghost "next step" (router deciding / not started yet)
    running_id = None
    if ghost and ids:
        gi = len(steps)
        p, q = _connector(gi - 1, gi)
        gx, gy = _slot(gi)
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        rn = router_node(dec_i, mx, my, pending=(running != "decision"),
                         running_now=(running == "decision"))
        nodes_out.append(rn)
        nodes_out.append({
            "id": "ghost", "x": gx, "y": gy, "label": "Next step",
            "sub": "chosen by the router", "kind": "ghost", "status": "pending",
            "title": "Next step",
            "desc": "Nodes appear here as the router chooses them.",
            "tool": "", "note": "", "runs": None, "order": None,
            "sections": {}, "raw": {},
        })
        edges_out += _agent_edges(
            "eg", ids[-1], "ghost", p, q, circle=True, circle_id=rn["id"],
            dst_status="pending", circle_status=rn["status"], dashed=True)
        if running == "decision":
            running_id = rn["id"]
    elif ghost:
        gx, gy = _slot(0)
        nodes_out.append({
            "id": "ghost", "x": gx, "y": gy, "label": "Next step",
            "sub": "chosen by the router", "kind": "ghost", "status": "pending",
            "title": "Next step",
            "desc": "The LLM will choose a tool based on your request.",
            "tool": "", "note": "", "runs": None, "order": None,
            "sections": {}, "raw": {},
        })

    for n in nodes_out:
        if n["status"] == "running" and n["kind"] == "real":
            running_id = n["id"]

    # ---- 4. bands, palette of unused tools
    total_slots = len(steps) + (1 if ghost else 0)
    rows = max(1, math.ceil(total_slots / AGENT_COLS))
    band1_y0 = 30
    band1_y1 = AGENT_Y0 + (rows - 1) * AGENT_PY + NH / 2 + 60
    n_exec = sum(1 for s in steps if s["state"] != "pending")

    stages = [{
        "label": f"ROUTE TAKEN  ·  {n_exec} NODE RUN{'S' if n_exec != 1 else ''}"
                 f"  ·  {len(log)} ROUTER DECISION{'S' if len(log) != 1 else ''}",
        "y0": band1_y0, "y1": band1_y1,
    }]

    executed = {s["node"] for s in steps}
    unused = [n for n in NODE_INFO if n not in executed and n != "decision"]
    H = band1_y1 + 110

    if unused:
        band2_y0 = band1_y1 + 44
        rows2 = math.ceil(len(unused) / AGENT_COLS)
        first_y = band2_y0 + 95
        for k, name in enumerate(unused):
            r, c = divmod(k, AGENT_COLS)
            info = NODE_INFO[name]
            merged = dict(ss.node_details.get(name, {}))
            sections, raw = node_sections_and_raw(merged)
            nodes_out.append({
                "id": f"t:{name}", "x": AGENT_X0 + c * AGENT_PX, "y": first_y + r * PALETTE_PY,
                "label": info["title"], "sub": name, "kind": "real",
                "status": "skipped" if ss.run_done else status[name],
                "title": info["title"], "desc": info["description"], "tool": info["tool"],
                "note": ("Not used in this run." if ss.run_done
                         else "Available to the router; not used yet."),
                "runs": 0, "order": None, "sections": sections, "raw": raw,
            })
        band2_y1 = first_y + (rows2 - 1) * PALETTE_PY + NH / 2 + 40
        stages.append({"label": "AVAILABLE NODES  ·  NOT USED IN THIS RUN"
                       if ss.run_done else "AVAILABLE NODES  ·  THE ROUTER MAY CHOOSE FROM THESE",
                       "y0": band2_y0, "y1": band2_y1})
        H = band2_y1 + 110

    last_id = None
    for i in range(len(steps) - 1, -1, -1):
        if steps[i]["state"] == "done":
            last_id = ids[i]
            break

    req = (ss.user_request or "").strip().replace("\n", " ")
    return {
        "W": AGENT_W, "H": H, "NW": NW, "NH": NH,
        "title": "Route map",
        "subtitle": (req[:90] + "…") if len(req) > 90 else req,
        "key": "preplab_route_v1",
        "footnote": ("This is the route recorded from the actual run. Circles are LLM decisions. "
                     "Reasons are the model's stated rationale."),
        "stages": stages, "nodes": nodes_out, "edges": edges_out,
        "live": bool(live), "running": running_id, "last": last_id,
        "counts": {s: sum(1 for v in status.values() if v == s) for s in VALID_STATUS},
    }


# =========================================================
# 7. FLOWCHART COMPONENT (SVG canvas: zoom / pan / click / live)
# =========================================================

FLOW_HTML = r"""<!doctype html>
<html data-theme="dark"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{
  --bg:#0a0f1f;--dot:#1c2650;--band:rgba(255,255,255,.028);--card:#121a35;--card2:#0e152c;
  --ink:#e8edff;--muted:#8f9bc7;--line:#2c3a6b;--edge:#4c5d99;--panel:rgba(14,21,44,.97);
  --pending:#5b9bff;--running:#fbbf24;--success:#34d399;--failed:#f87171;--skipped:#8a96b8;--accent:#8b9aff;
  --shadow:0 10px 34px rgba(0,0,0,.5);
}
html[data-theme=light]{
  --bg:#f3f5fc;--dot:#d3d9ee;--band:rgba(40,60,140,.045);--card:#ffffff;--card2:#f1f4fd;
  --ink:#141b34;--muted:#5b6690;--line:#cbd3ee;--edge:#9aa6d1;--panel:rgba(255,255,255,.98);
  --pending:#2563eb;--running:#d97706;--success:#059669;--failed:#dc2626;--skipped:#7a86a8;--accent:#4f46e5;
  --shadow:0 10px 34px rgba(20,30,70,.18);
}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);font:14px Inter,system-ui,-apple-system,Segoe UI,sans-serif;overflow:hidden}
#wrap{position:relative;height:100vh;border:1px solid var(--line);border-radius:16px;overflow:hidden;background:var(--bg)}
svg{display:block;width:100%;height:100%;cursor:grab;touch-action:none;user-select:none}
svg.grabbing{cursor:grabbing}
#vp{transform-origin:0 0}
#vp.smooth{transition:transform .38s cubic-bezier(.2,.8,.2,1)}

#toolbar{position:absolute;left:12px;right:12px;top:12px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;z-index:5;pointer-events:none}
#toolbar>*{pointer-events:auto}
.pill,.tools{background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow);backdrop-filter:blur(6px)}
.pill{display:flex;align-items:center;gap:8px;padding:7px 12px;font-weight:600}
.pill small{font-weight:500;color:var(--muted);max-width:340px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.live{display:inline-flex;align-items:center;gap:5px;font-size:11px;letter-spacing:.08em;color:var(--running);font-weight:700}
.live:before{content:"";width:8px;height:8px;border-radius:50%;background:var(--running);animation:blink 1s infinite}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:12px;color:var(--muted);padding:7px 12px}
.legend b{color:var(--ink);font-weight:600}
.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:baseline}
.tools{display:flex;align-items:center;gap:4px;padding:4px;margin-left:auto}
.tools button{background:transparent;border:0;color:var(--ink);font:600 12.5px inherit;padding:7px 10px;border-radius:8px;cursor:pointer}
.tools button:hover{background:var(--card2)}
.tools button.toggle.on{background:var(--accent);color:#fff}
#zlbl{min-width:44px;text-align:center;font-size:12px;color:var(--muted)}
#hint{position:absolute;left:14px;bottom:12px;font-size:12px;color:var(--muted);background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:6px 10px;z-index:4}

/* nodes */
.node{cursor:pointer;--c:var(--pending)}
.node .box{fill:var(--card);stroke:var(--line);stroke-width:1.8;filter:drop-shadow(0 5px 10px rgba(0,0,0,.22))}
.node .accent{fill:var(--c)}
.node.n-running{--c:var(--running)}
.node.n-success{--c:var(--success)}
.node.n-failed{--c:var(--failed)}
.node.n-skipped{--c:var(--skipped);opacity:.5}
.node.n-success .box{stroke:var(--success)}
.node.n-failed .box{stroke:var(--failed);stroke-width:2.6}
.node.n-running .box{stroke:var(--running);stroke-width:2.8;animation:pulse 1.3s ease-in-out infinite}
.node.n-skipped .box{stroke-dasharray:6 5}
.node.k-logic .box{fill:var(--card2);stroke-dasharray:3 5}
.node.k-logic.n-success .box{stroke-dasharray:3 5}
.node.k-ghost{opacity:.6}
.node.k-ghost .box{fill:transparent;stroke-dasharray:5 6;filter:none}
.node:hover .box{stroke:var(--accent)}
.node.sel .box{stroke:var(--ink);stroke-width:3.2;stroke-dasharray:none}
.label{fill:var(--ink);font:600 15px Inter,system-ui,sans-serif}
.sub{fill:var(--muted);font:500 11px ui-monospace,SFMono-Regular,Menlo,monospace}
.badge circle.bg{fill:var(--c)}
.badge.hollow circle.bg{fill:none;stroke:var(--c);stroke-width:2}
.badge text{fill:#fff;font:800 13px Inter,sans-serif;text-anchor:middle}
.badge .spin{fill:none;stroke:#fff;stroke-width:2.4;stroke-linecap:round;stroke-dasharray:14 40;transform-box:fill-box;transform-origin:center;animation:spin .9s linear infinite}
.meta{fill:var(--muted);font:600 10.5px Inter,sans-serif;text-anchor:end}

/* router decision circles */
.node.k-router{--rcol:#8b5cf6}
.node.k-router .rc{fill:var(--rcol);stroke:var(--bg);stroke-width:3;filter:drop-shadow(0 4px 8px rgba(0,0,0,.35))}
.node.k-router .rt{fill:#fff;font:800 13px Inter,sans-serif;text-anchor:middle}
.node.k-router.n-pending{opacity:.55}
.node.k-router.n-running .rc{animation:pulse 1.2s ease-in-out infinite}
.node.k-router:hover .rc{stroke:var(--accent)}
.node.k-router.sel .rc{stroke:var(--ink)}
.node.k-router .spin2{fill:none;stroke:#fff;stroke-width:3;stroke-linecap:round;stroke-dasharray:20 60;transform-box:fill-box;transform-origin:center;animation:spin .9s linear infinite}

/* bands */
.band{fill:var(--band);stroke:var(--line);stroke-dasharray:2 7}
.band-label{fill:var(--muted);font:700 13px Inter,sans-serif;letter-spacing:.14em}
.foot{fill:var(--muted);font:500 15px Inter,sans-serif;text-anchor:middle}

/* edges */
.edge{fill:none;stroke:var(--edge);stroke-width:2.2;stroke-linejoin:round}
.edge.dashed{stroke-dasharray:7 6}
.edge.done{stroke:var(--success);stroke-width:3.4}
.edge.active{stroke:var(--running);stroke-width:3.4;stroke-dasharray:10 7;animation:dash .7s linear infinite}
.elabel rect{fill:var(--bg);stroke:var(--line)}
.elabel text{fill:var(--muted);font:600 11.5px Inter,sans-serif}
.elabel.done rect{stroke:var(--success)}
.elabel.done text{fill:var(--success)}
.elabel.active rect{stroke:var(--running)}
#arr-idle path{fill:var(--edge)}#arr-done path{fill:var(--success)}#arr-active path{fill:var(--running)}

/* panel */
#panel{position:absolute;top:64px;right:12px;bottom:12px;width:392px;max-width:calc(100% - 24px);overflow:auto;z-index:6;
  background:var(--panel);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow);padding:16px 16px 20px;
  transform:translateX(calc(100% + 24px));transition:transform .28s cubic-bezier(.2,.8,.2,1);backdrop-filter:blur(8px)}
#panel.open{transform:none}
#panel h3{margin:0;font-size:18px;line-height:1.25}
#panel h4{margin:18px 0 8px;font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
#panel p{margin:8px 0;line-height:1.5}
#panel .head{display:flex;gap:10px;align-items:flex-start;justify-content:space-between}
#pclose{background:var(--card2);border:1px solid var(--line);color:var(--ink);border-radius:9px;width:30px;height:30px;cursor:pointer;font-size:15px;flex:none}
.st{display:inline-block;padding:3px 10px;border-radius:99px;font:700 11px Inter,sans-serif;letter-spacing:.07em;color:#fff;margin-top:8px}
.st.pending{background:var(--pending)}.st.running{background:var(--running);color:#1b1300}.st.success{background:var(--success);color:#04261a}
.st.failed{background:var(--failed)}.st.skipped{background:var(--skipped)}
.note{color:var(--muted);font-size:12.5px;background:var(--card2);border:1px solid var(--line);border-radius:10px;padding:8px 10px;margin-top:10px;line-height:1.45}
.empty{color:var(--muted);font-style:italic;font-size:13px}
.kv{display:grid;grid-template-columns:112px 1fr;gap:8px;padding:6px 0;border-bottom:1px dashed var(--line)}
.kv:last-child{border-bottom:0}
.kv .k{color:var(--muted);font:600 12px ui-monospace,Menlo,monospace;word-break:break-word}
.kv .v{min-width:0;word-break:break-word}
.tag{display:inline-block;background:var(--card2);border:1px solid var(--line);border-radius:7px;padding:1px 7px;margin:0 4px 4px 0;font:500 12px ui-monospace,Menlo,monospace}
pre{margin:0;max-height:200px;overflow:auto;background:var(--card2);border:1px solid var(--line);border-radius:9px;padding:8px;font:11.5px/1.45 ui-monospace,Menlo,monospace;white-space:pre-wrap;word-break:break-word}
code{font:12px ui-monospace,Menlo,monospace;background:var(--card2);border:1px solid var(--line);border-radius:6px;padding:1px 6px}
.chip{display:inline-flex;gap:6px;align-items:center;margin:0 6px 6px 0;background:var(--card2);color:var(--ink);border:1px solid var(--line);border-radius:99px;padding:5px 11px;font:600 12px Inter,sans-serif;cursor:pointer}
.chip:hover{border-color:var(--accent)}
.chip i{color:var(--muted);font-style:normal;font-weight:500}
details summary{cursor:pointer;color:var(--muted);font-weight:600;margin-top:16px}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes dash{to{stroke-dashoffset:-34}}
@keyframes blink{50%{opacity:.25}}
@keyframes pulse{0%,100%{filter:drop-shadow(0 0 0 rgba(251,191,36,0))}50%{filter:drop-shadow(0 0 14px rgba(251,191,36,.85))}}
</style></head>
<body>
<div id="wrap">
  <div id="toolbar">
    <div class="pill"><span id="title">Pipeline map</span> <span id="live" class="live" hidden>LIVE</span> <small id="subtitle"></small></div>
    <div class="pill legend" id="legend"></div>
    <div class="tools">
      <button id="zo" title="Zoom out (-)">&minus;</button>
      <span id="zlbl">100%</span>
      <button id="zi" title="Zoom in (+)">+</button>
      <button id="fit" title="Fit the whole chart (0)">Fit</button>
      <button id="fitw" title="Fit to width">Width</button>
      <button id="goto" title="Jump to the running / latest node">Active</button>
      <button id="follow" class="toggle" title="Keep the running node in view">Follow</button>
      <button id="wheel" class="toggle" title="Mouse wheel zooms (off = Ctrl/Cmd + wheel)">Wheel zoom</button>
      <button id="theme" title="Toggle light / dark">&#9680;</button>
    </div>
  </div>
  <svg id="svg">
    <defs>
      <pattern id="dots" width="30" height="30" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1.3" fill="var(--dot)"/></pattern>
      <marker id="arr-idle" markerWidth="11" markerHeight="11" refX="9" refY="5.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L11,5.5 L0,11 z"/></marker>
      <marker id="arr-done" markerWidth="13" markerHeight="13" refX="10" refY="6.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L13,6.5 L0,13 z"/></marker>
      <marker id="arr-active" markerWidth="13" markerHeight="13" refX="10" refY="6.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L13,6.5 L0,13 z"/></marker>
    </defs>
    <rect id="grid" width="100%" height="100%" fill="url(#dots)"/>
    <g id="vp"></g>
  </svg>
  <div id="hint">Scroll = zoom &middot; drag = pan &middot; click any node or circle for details</div>
  <aside id="panel"></aside>
</div>
<script>
(function(){
const D=/*__DATA__*/null;
const NS='http://www.w3.org/2000/svg';
const $=s=>document.querySelector(s);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function el(tag,attrs,parent){const e=document.createElementNS(NS,tag);for(const k in (attrs||{}))e.setAttribute(k,attrs[k]);if(parent)parent.appendChild(e);return e;}

const KEY=D.key||'preplab_flow_v2';
const store={
  get(){try{return JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){return {}}},
  set(p){try{localStorage.setItem(KEY,JSON.stringify(Object.assign(store.get(),p)))}catch(e){}}
};
const saved=store.get();
document.documentElement.dataset.theme=saved.theme||(matchMedia('(prefers-color-scheme: light)').matches?'light':'dark');
let follow=saved.follow!==false, wheelZoom=saved.wheel!==false, sel=null;

$('#title').textContent=D.title||'Pipeline map';
$('#subtitle').textContent=D.subtitle||'';
$('#subtitle').title=D.subtitle||'';

const svg=$('#svg'), vp=$('#vp'), dots=$('#dots');
const bandsL=el('g',{},vp), edgesL=el('g',{},vp), labelsL=el('g',{},vp), nodesL=el('g',{},vp);
const byId={}; D.nodes.forEach(n=>byId[n.id]=n);
const NW=D.NW, NH=D.NH;

/* ---------- drawing ---------- */
function pathFrom(pts,r){
  r=r||16; let d='M'+pts[0][0]+','+pts[0][1];
  for(let i=1;i<pts.length-1;i++){
    const a=pts[i-1],b=pts[i],c=pts[i+1];
    const l1=Math.hypot(b[0]-a[0],b[1]-a[1]), l2=Math.hypot(c[0]-b[0],c[1]-b[1]);
    const rr=Math.min(r,l1/2,l2/2);
    d+=' L'+(b[0]-(b[0]-a[0])/l1*rr)+','+(b[1]-(b[1]-a[1])/l1*rr)+' Q'+b[0]+','+b[1]+' '+(b[0]+(c[0]-b[0])/l2*rr)+','+(b[1]+(c[1]-b[1])/l2*rr);
  }
  const z=pts[pts.length-1]; return d+' L'+z[0]+','+z[1];
}
D.stages.forEach(s=>{
  el('rect',{class:'band',x:20,y:s.y0,width:D.W-40,height:s.y1-s.y0,rx:22},bandsL);
  el('text',{class:'band-label',x:44,y:s.y0+34},bandsL).textContent=s.label;
});
el('text',{class:'foot',x:D.W/2,y:D.H-28},vp).textContent=D.footnote||'';

D.edges.forEach(e=>{
  if(e.pts.length<2)return;
  el('path',{class:'edge '+e.state+(e.dashed?' dashed':''),d:pathFrom(e.pts),'marker-end':'url(#arr-'+e.state+')'},edgesL);
  if(e.label){
    const w=e.label.length*6.9+20;
    const g=el('g',{class:'elabel '+e.state,transform:'translate('+e.lx+','+e.ly+')'},labelsL);
    el('rect',{x:-w/2,y:-12,width:w,height:24,rx:12},g);
    el('text',{y:4,'text-anchor':'middle'},g).textContent=e.label;
  }
});

const GLYPH={success:'✓',failed:'✕',skipped:'–'};
function drawRouter(n){
  const g=el('g',{class:'node k-router n-'+n.status+' src-'+(n.source||'llm'),transform:'translate('+n.x+','+n.y+')','data-id':n.id,tabindex:0},nodesL);
  el('circle',{class:'rc',r:25},g);
  if(n.status==='running') el('circle',{class:'spin2',r:11},g);
  else el('text',{class:'rt',y:5},g).textContent=n.label;
  n.g=g;
}
D.nodes.forEach(n=>{
  if(n.kind==='router'){drawRouter(n);return;}
  const g=el('g',{class:'node n-'+n.status+' k-'+n.kind,transform:'translate('+(n.x-NW/2)+','+(n.y-NH/2)+')','data-id':n.id,tabindex:0},nodesL);
  el('rect',{class:'box',width:NW,height:NH,rx:14},g);
  el('rect',{class:'accent',x:11,y:15,width:4,height:NH-30,rx:2},g);
  el('text',{class:'label',x:28,y:34},g).textContent=n.label;
  el('text',{class:'sub',x:28,y:55},g).textContent=n.kind==='ghost'?n.sub:(n.sub+'  ·  '+n.status.toUpperCase());
  if(n.kind!=='ghost'){
    const b=el('g',{class:'badge'+(n.status==='pending'?' hollow':''),transform:'translate('+(NW-24)+',24)'},g);
    el('circle',{class:'bg',r:12},b);
    if(n.status==='running') el('circle',{class:'spin',r:6},b);
    else if(GLYPH[n.status]) el('text',{y:5},b).textContent=GLYPH[n.status];
    const meta=[]; if(n.order) meta.push('#'+n.order); if(n.runs>1) meta.push('×'+n.runs);
    if(meta.length) el('text',{class:'meta',x:NW-14,y:NH-12},g).textContent=meta.join('  ');
  }
  n.g=g;
});

/* ---------- view: zoom / pan ---------- */
let view={x:0,y:0,k:1}, pt;
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const box=()=>svg.getBoundingClientRect();
function apply(smooth){
  vp.classList.toggle('smooth',!!smooth);
  vp.style.transform='translate('+view.x+'px,'+view.y+'px) scale('+view.k+')';
  dots.setAttribute('patternTransform','translate('+view.x+' '+view.y+') scale('+view.k+')');
  $('#zlbl').textContent=Math.round(view.k*100)+'%';
  clearTimeout(pt); pt=setTimeout(()=>store.set({view}),150);
  if(smooth) setTimeout(()=>vp.classList.remove('smooth'),420);
}
function zoomAt(cx,cy,f,smooth){
  const k=clamp(view.k*f,0.12,2.6), r=k/view.k;
  view={x:cx-(cx-view.x)*r,y:cy-(cy-view.y)*r,k}; apply(smooth);
}
function fitAll(){const r=box();const k=Math.min(r.width/(D.W+60),r.height/(D.H+60));view={k,x:(r.width-D.W*k)/2,y:(r.height-D.H*k)/2};apply(true);}
function fitWidth(smooth){const r=box();const k=clamp(r.width/(D.W+40),0.25,1.2);view={k,x:(r.width-D.W*k)/2,y:70};apply(smooth);}
function focus(id,k){
  const n=byId[id]; if(!n)return; const r=box();
  const kk=k||Math.max(view.k,0.6), off=sel?190:0;
  view={k:kk,x:r.width/2-off-n.x*kk,y:r.height/2-n.y*kk}; apply(true);
}

svg.addEventListener('wheel',e=>{
  if(!wheelZoom&&!(e.ctrlKey||e.metaKey))return;
  e.preventDefault(); const r=box();
  zoomAt(e.clientX-r.left,e.clientY-r.top,Math.exp(-e.deltaY*(e.ctrlKey?0.01:0.0015)));
},{passive:false});

let drag=null,moved=0;
svg.addEventListener('pointerdown',e=>{
  if(e.button!==0)return;
  drag={x:e.clientX,y:e.clientY,vx:view.x,vy:view.y}; moved=0; svg.classList.add('grabbing');
});
window.addEventListener('pointermove',e=>{
  if(!drag)return; const dx=e.clientX-drag.x, dy=e.clientY-drag.y;
  moved=Math.max(moved,Math.hypot(dx,dy));
  if(moved>4){view.x=drag.vx+dx;view.y=drag.vy+dy;apply();}
});
window.addEventListener('pointerup',e=>{
  if(!drag)return; drag=null; svg.classList.remove('grabbing');
  if(moved<5){ const g=e.target.closest&&e.target.closest('.node'); select(g?g.dataset.id:null); }
});
svg.addEventListener('keydown',e=>{ if(e.key==='Enter'){const g=e.target.closest('.node'); if(g)select(g.dataset.id);} });

/* ---------- details panel ---------- */
function fmt(v){
  if(v===null||v===undefined)return '<span class="empty">—</span>';
  if(Array.isArray(v)&&v.every(x=>x===null||typeof x!=='object'))
    return v.length?v.map(x=>'<span class="tag">'+esc(x)+'</span>').join(''):'<span class="empty">none</span>';
  if(typeof v==='object')return '<pre>'+esc(JSON.stringify(v,null,2))+'</pre>';
  return '<span>'+esc(v)+'</span>';
}
function renderPanel(){
  const P=$('#panel');
  if(!sel){P.classList.remove('open');return;}
  const n=byId[sel];
  const ins=D.edges.filter(e=>e.to===n.id), outs=D.edges.filter(e=>e.from===n.id);
  const chip=(id,lab)=>'<button class="chip" data-go="'+id+'">'+esc(byId[id].kind==='router'?(byId[id].title||byId[id].label):byId[id].label)+(lab?' <i>'+esc(lab)+'</i>':'')+'</button>';
  const empty=n.status==='pending'?'Not run yet — this fills in from the real execution.'
    :(n.status==='skipped'?'Skipped in this run.':'Nothing was recorded for this section.');
  const sec=(title,key)=>{
    const o=n.sections[key];
    const rows=o?Object.entries(o).map(([k,v])=>'<div class="kv"><div class="k">'+esc(k)+'</div><div class="v">'+fmt(v)+'</div></div>').join('')
      :'<p class="empty">'+empty+'</p>';
    return '<h4>'+title+'</h4>'+rows;
  };
  const rawKeys=Object.keys(n.raw||{});
  P.innerHTML=
   '<div class="head"><div><h3>'+esc(n.title)+'</h3><span class="st '+n.status+'">'+n.status.toUpperCase()+'</span></div><button id="pclose" title="Close (Esc)">✕</button></div>'
   +(n.desc?'<p>'+esc(n.desc)+'</p>':'')
   +(n.note?'<div class="note">'+esc(n.note)+'</div>':'')
   +'<h4>Execution</h4>'
   +'<div class="kv"><div class="k">status</div><div class="v">'+n.status+'</div></div>'
   +(n.tool?'<div class="kv"><div class="k">source</div><div class="v"><code>'+esc(n.tool)+'</code></div></div>':'')
   +((n.kind==='real'||n.kind==='alias')&&n.runs!=null?'<div class="kv"><div class="k">runs</div><div class="v">'+n.runs+'</div></div>':'')
   +(n.order?'<div class="kv"><div class="k">position</div><div class="v">#'+n.order+'</div></div>':'')
   +sec('Input','input')+sec('Decision','decision')+sec('Reason','reason')+sec('Output','output')+sec('Errors / warnings','issues')
   +'<h4>Connections</h4>'
   +'<p style="margin:0 0 6px;color:var(--muted);font-size:12px">Comes from</p>'+(ins.length?ins.map(e=>chip(e.from,e.label)).join(''):'<p class="empty">Entry point</p>')
   +'<p style="margin:8px 0 6px;color:var(--muted);font-size:12px">Leads to</p>'+(outs.length?outs.map(e=>chip(e.to,e.label)).join(''):'<p class="empty">End of the workflow</p>')
   +(rawKeys.length?'<details><summary>Everything recorded</summary><div style="margin-top:8px"><pre>'+esc(JSON.stringify(n.raw,null,2))+'</pre></div></details>':'');
  P.classList.add('open'); P.scrollTop=0;
}
function select(id){
  sel=id&&byId[id]?id:null; store.set({sel});
  D.nodes.forEach(n=>n.g.classList.toggle('sel',n.id===sel));
  renderPanel();
}
$('#panel').addEventListener('click',e=>{
  if(e.target.closest('#pclose')){select(null);return;}
  const b=e.target.closest('[data-go]'); if(b){select(b.dataset.go);focus(b.dataset.go);}
});

/* ---------- toolbar ---------- */
const center=()=>{const r=box();return [r.width/2,r.height/2];};
$('#zi').onclick=()=>{const c=center();zoomAt(c[0],c[1],1.3,true)};
$('#zo').onclick=()=>{const c=center();zoomAt(c[0],c[1],1/1.3,true)};
$('#fit').onclick=fitAll;
$('#fitw').onclick=()=>fitWidth(true);
$('#goto').onclick=()=>{const id=D.running||D.last; if(id){focus(id,Math.max(view.k,0.8));}};
const setToggle=(id,on)=>$(id).classList.toggle('on',on);
setToggle('#follow',follow); setToggle('#wheel',wheelZoom);
$('#follow').onclick=()=>{follow=!follow;store.set({follow});setToggle('#follow',follow);};
$('#wheel').onclick=()=>{wheelZoom=!wheelZoom;store.set({wheel:wheelZoom});setToggle('#wheel',wheelZoom);};
$('#theme').onclick=()=>{const t=document.documentElement.dataset.theme==='dark'?'light':'dark';document.documentElement.dataset.theme=t;store.set({theme:t});};
window.addEventListener('keydown',e=>{
  if(e.key==='Escape')select(null);
  else if(e.key==='+'||e.key==='='){const c=center();zoomAt(c[0],c[1],1.2,true);}
  else if(e.key==='-'){const c=center();zoomAt(c[0],c[1],1/1.2,true);}
  else if(e.key==='0')fitAll();
});

const C=D.counts, LG=[['success','Success'],['running','Running'],['pending','Pending'],['failed','Failed'],['skipped','Skipped']];
$('#legend').innerHTML=LG.map(s=>'<span><i style="background:var(--'+s[0]+')"></i>'+s[1]+' <b>'+(C[s[0]]||0)+'</b></span>').join('');
if(D.live)$('#live').hidden=false;

/* ---------- initial view ---------- */
if(saved.view&&isFinite(saved.view.k)){view=saved.view;apply();}else{fitWidth();}
if(saved.sel&&byId[saved.sel])select(saved.sel);
if(D.live&&follow&&D.running)focus(D.running);
})();
</script>
</body></html>
"""


def safe_json(payload):
    return json.dumps(payload, default=str).replace("</", "<\\/")


def render_flow(slot, live=False):
    if st.session_state.run_mode == "agent":
        payload = build_agent_payload(live)
    else:
        payload = build_payload(live)
    html = FLOW_HTML.replace("/*__DATA__*/null", safe_json(payload))
    with slot.container():
        components.html(html, height=int(flow_height), scrolling=False)


# =========================================================
# 8. SIDEBAR
# =========================================================

with st.sidebar:
    st.title("⚙️ PrepLab Controls")

    st.subheader("Groq API Configuration")

    api_key_input = st.text_input(
        "Groq API Key",
        type="password",
        value=st.session_state.api_key,
        placeholder="Paste your Groq API key",
        help="The key is kept in Streamlit session state and is not included in reports.",
    )

    st.session_state.api_key = api_key_input.strip()

    if st.session_state.api_key:
        st.success("API key supplied for this session.")
    else:
        st.info(
            "Agent mode requires a Groq key (here or GROQ_API_KEY in .env). "
            "It will not run tools without LLM routing."
        )

    st.caption(
        "Do not commit API keys to GitHub or include them in downloaded reports."
    )

    st.divider()

    st.subheader("Agent settings")
    llm_model = st.text_input(
        "Groq model", value=DEFAULT_MODEL,
        help="A Groq chat model that supports function calling.",
    )
    max_steps = st.slider(
        "Max routing steps", min_value=5, max_value=40, value=25,
        help="Safety limit: the run reports an error if the LLM has not selected finish by this many decisions.",
    )

    st.divider()

    st.subheader("Flowchart")
    flow_height = st.slider(
        "Canvas height (px)", min_value=600, max_value=1400, value=860, step=20,
        help="Make the map taller or shorter.",
    )

    st.markdown(
        """
        🟦 **Pending** — not executed yet

        🟨 **Running** — currently executing

        🟩 **Success** — executed successfully

        🟥 **Failed** — execution failed

        ⬜ **Skipped** — not selected or no work needed

        🟣 **Router circle** — LLM decision
        """
    )

    st.divider()

    st.subheader("Pipeline")
    for i, step in enumerate([
        "Tools selected from your request by the LLM",
        "Only selected tools and required workflow steps are run",
    ], 1):
        st.write(f"{i}. {step}")


# =========================================================
# 9. DATA UPLOAD
# =========================================================

st.subheader("1. Upload Dataset")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to view the dataset and configure preprocessing.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except pd.errors.EmptyDataError:
    st.error("The uploaded CSV file is empty.")
    st.stop()
except UnicodeDecodeError:
    st.error("Unable to decode the CSV. Please use UTF-8 encoding.")
    st.stop()
except Exception as exc:
    st.error(f"Unable to read the CSV: {exc}")
    st.stop()

if df.empty:
    st.error("The CSV contains no data rows.")
    st.stop()

if len(df.columns) == 0:
    st.error("The CSV contains no columns.")
    st.stop()

if df.columns.duplicated().any():
    st.error("Duplicate column names were detected. Rename them before uploading.")
    st.stop()

original_df = df.copy(deep=True)
st.session_state.original_df = original_df.copy(deep=True)

c1, c2, c3 = st.columns(3)
c1.metric("Rows", original_df.shape[0])
c2.metric("Columns", original_df.shape[1])
c3.metric("Missing Values", int(original_df.isna().sum().sum()))

with st.expander("Preview uploaded dataset"):
    st.dataframe(original_df.head(20), use_container_width=True)


# =========================================================
# 10. REQUEST + PREPROCESSING SETTINGS
# =========================================================

st.subheader("2. Ask PrepLab / Configure Preprocessing")

with st.form("pipeline_configuration"):
    st.caption("The LLM selects tools from your request. Only selected tools and required workflow steps run.")

    user_request = st.text_area(
        "What should PrepLab do?",
        placeholder="e.g. Inspect the dataset only. / Convert amount to numeric. / "
                    "Fill missing values and cap numerical outliers.",
        height=100,
        help="In Agent mode, the LLM uses this request to select tools. A request is required.",
    )

    problem_type = st.selectbox(
        "Problem Type", ["Classification", "Regression", "Clustering"]
    )

    target_options = ["None"] + [str(c) for c in original_df.columns]
    target_column = st.selectbox("Target Column", target_options)

    cardinality_threshold = st.number_input(
        "High Cardinality Threshold", min_value=2, max_value=1000, value=10
    )

    run_button = st.form_submit_button(
        "🚀 Run", type="primary", use_container_width=True
    )

run_mode = "agent"

# =========================================================
# 11. FLOWCHART (renders live while the graph runs)
# =========================================================

st.subheader("3. Pipeline Flowchart")
st.caption(
    "Scroll to zoom, drag to pan, click any node or router circle for its status, input, "
    "decision, reason, output, warnings and connections. In Agent mode the chart shows the "
    "route the LLM actually took; unselected tools remain available below the route."
)

notice = st.session_state.run_notice
if notice:
    kind, message = notice
    (st.success if kind == "success" else st.error)(message)
    if kind == "error" and st.session_state.processing_traceback:
        with st.expander("Technical error details"):
            st.code(st.session_state.processing_traceback)

flow_slot = st.empty()
if not run_button:
    render_flow(flow_slot)


# =========================================================
# 12. EXECUTE GRAPH WITH LIVE UPDATES
# =========================================================

if run_button:
    if run_mode == "agent" and not (user_request or "").strip():
        st.error("Enter a request so the LLM can select which tools to run.")
        st.stop()

    ss = st.session_state
    ss.processed_df = None
    ss.processing_result = None
    ss.processing_error = None
    ss.processing_traceback = None
    ss.run_notice = None
    ss.run_done = False
    ss.failed_node = None
    ss.run_mode = run_mode
    ss.user_request = (user_request or "").strip() if run_mode == "agent" else ""
    ss.node_status = {name: "pending" for name in NODE_INFO}
    ss.node_details = {}
    ss.execution_trace = []
    ss.route_log = []

    ss.running_node = "decision"
    ss.node_status["decision"] = "running"

    # Modules that read GROQ_API_KEY from the environment. Never written to disk or reports.
    if ss.api_key:
        os.environ["GROQ_API_KEY"] = ss.api_key

    initial_state = {
        "current_df": original_df.copy(deep=True),
        "original_df": original_df.copy(deep=True),
        "file_path": uploaded_file.name,
        "problem_type": problem_type.lower(),
        "target_column": None if target_column == "None" else target_column,
        "original_shape": original_df.shape,
        "final_shape": original_df.shape,
        "completed_steps": [],
        "processing_history": [],
        "errors": [],
        "warnings": [],
        "ordinal_mappings": {},
        "type_conversions": {},
        "high_cardinality_threshold": int(cardinality_threshold),
        "scaling_method": None,
        "finished": False,
    }

    render_flow(flow_slot, live=True)

    recorder = None
    try:
        if run_mode == "agent":
            if AGENT_IMPORT_ERROR:
                raise RuntimeError(
                    f"Agent mode could not be loaded ({AGENT_IMPORT_ERROR}). "
                    "Check agent/agentic.py and that 'langgraph' and 'groq' are installed."
                )
            recorder = RouteRecorder(request=ss.user_request)
            router = Router(
                ss.user_request, recorder, api_key=ss.api_key,
                model=llm_model, max_steps=int(max_steps),
            )
            graph = build_agentic_graph(router)
            recursion_limit = 20 + 3 * int(max_steps)
        else:
            graph = build_graph()
            recursion_limit = 100

        final_state = None

        # One single execution: "updates" drives the live chart,
        # "values" gives us the final state (no second graph.invoke()).
        for mode, chunk in graph.stream(
            initial_state,
            config={"recursion_limit": recursion_limit},
            stream_mode=["updates", "values"],
        ):
            if mode == "values":
                final_state = chunk
                continue

            for node_name, update in chunk.items():
                if node_name not in NODE_INFO:
                    continue

                update = update if isinstance(update, dict) else {}

                ss.node_status[node_name] = "success"
                ss.node_details[node_name] = safe(update)
                ss.execution_trace.append({"node": node_name, "update": safe(update)})

                if recorder is not None:
                    ss.route_log = safe(recorder.decisions)

                nxt = infer_running(node_name, update)
                ss.running_node = nxt
                if nxt:
                    ss.node_status[nxt] = "running"

                render_flow(flow_slot, live=True)

        if not isinstance(final_state, dict):
            raise TypeError("The graph did not return a state dictionary.")

        processed_df = final_state.get("current_df")
        if not isinstance(processed_df, pd.DataFrame):
            raise ValueError("No valid processed DataFrame was returned.")

        ss.processing_result = final_state
        ss.processed_df = processed_df.copy(deep=True)
        ss.running_node = None
        if recorder is not None:
            ss.route_log = safe(recorder.decisions)

        # Reconcile with the recorded processing history.
        for entry in final_state.get("processing_history", []):
            name = entry.get("step")
            if name in NODE_INFO:
                ss.node_status[name] = norm_status(entry.get("status"))
                details = ss.node_details.setdefault(name, {})
                for k, v in safe(entry).items():
                    if k != "step":
                        details[k] = v

        ss.node_status["finish"] = "success" if final_state.get("finished") else "pending"

        # Nodes that never ran in this execution were not selected.
        for name in NODE_INFO:
            if ss.node_status.get(name) in ("pending", "running") and not (
                name == "finish" and not final_state.get("finished")
            ):
                ss.node_status[name] = "skipped"

        ss.run_done = True
        ss.run_notice = ("success", "Preprocessing completed.")

    except Exception as exc:
        ss.processing_error = str(exc)
        ss.processing_traceback = traceback.format_exc()
        if recorder is not None:
            ss.route_log = safe(recorder.decisions)

        failed_node = ss.running_node
        if failed_node and failed_node != "decision":
            ss.node_status[failed_node] = "failed"
            ss.node_details.setdefault(failed_node, {})["error"] = str(exc)
            ss.failed_node = failed_node
        elif failed_node == "decision":
            ss.node_status["decision"] = "failed"
            ss.node_details.setdefault("decision", {})["error"] = str(exc)
        ss.running_node = None

        ss.run_notice = ("error", f"Preprocessing failed: {exc}")

    st.rerun()


# =========================================================
# 13. ROUTING LOG + NODE INSPECTOR (table views)
# =========================================================

if st.session_state.route_log:
    with st.expander("Routing decisions (table view)", expanded=False):
        st.caption(
            "One row per router decision. 'reason' is the model's stated rationale; "
            "the LLM selects every route."
        )
        route_df = pd.DataFrame([
            {
                "step": d.get("step"),
                "after": d.get("from_node"),
                "chosen": d.get("chosen"),
                "source": d.get("source"),
                "reason": d.get("reason"),
                "notes": "; ".join(d.get("notes", [])),
                "latency_ms": d.get("latency_ms"),
            }
            for d in st.session_state.route_log
        ])
        st.dataframe(route_df, use_container_width=True)

with st.expander("Node inspector (table view)"):
    selected_node = st.selectbox(
        "Choose a graph node",
        list(NODE_INFO.keys()),
        format_func=lambda name: NODE_INFO[name]["title"],
    )

    info = NODE_INFO[selected_node]
    node_state = st.session_state.node_status.get(selected_node, "pending")
    details = st.session_state.node_details.get(selected_node, {})

    a, b = st.columns(2)
    with a:
        st.write("**Execution status**")
        st.write(node_state.upper())
        st.write("**Tool / source file**")
        st.code(info["tool"])
    with b:
        st.write("**What this node does**")
        st.write(info["description"])

    st.write("**Recorded internal details**")
    if details:
        st.json(details, expanded=False)
    else:
        st.info("This node has not recorded execution details yet.")

    result_for_history = st.session_state.processing_result
    if result_for_history:
        matching = [
            entry for entry in result_for_history.get("processing_history", [])
            if entry.get("step") == selected_node
        ]
        if matching:
            st.write("**Execution history for this node**")
            st.dataframe(pd.DataFrame(matching), use_container_width=True)


# =========================================================
# 14. RESULTS
# =========================================================

result = st.session_state.processing_result
processed_df = st.session_state.processed_df

if result is not None and processed_df is not None:
    st.subheader("4. Preprocessing Results")

    original_shape = original_df.shape
    final_shape = processed_df.shape

    c1, c2, c3 = st.columns(3)
    c1.metric("Original Rows", original_shape[0])
    c2.metric("Processed Rows", final_shape[0])
    c3.metric("Processed Columns", final_shape[1])

    st.write("**Processed dataset**")
    st.dataframe(processed_df, use_container_width=True)

    missing_counts = processed_df.isna().sum()
    missing_total = int(missing_counts.sum())

    st.metric("Remaining Missing Values", missing_total)

    if missing_total:
        st.dataframe(
            missing_counts[missing_counts > 0].rename("Missing Values").to_frame(),
            use_container_width=True,
        )
    else:
        st.success("No missing values remain.")

    st.download_button(
        "⬇️ Download Processed CSV",
        data=processed_df.to_csv(index=False).encode("utf-8"),
        file_name="preplab_processed.csv",
        mime="text/csv",
        use_container_width=True,
    )

    report = {
        "file_name": uploaded_file.name,
        "mode": st.session_state.run_mode,
        "user_request": st.session_state.user_request,
        "problem_type": problem_type,
        "target_column": None if target_column == "None" else target_column,
        "original_shape": list(original_shape),
        "final_shape": list(final_shape),
        "completed_steps": result.get("completed_steps", []),
        "processing_history": result.get("processing_history", []),
        "routing_log": st.session_state.route_log,
        "errors": result.get("errors", []),
        "warnings": result.get("warnings", []),
        "scaling_method": result.get("scaling_method_used"),
        "high_cardinality_threshold": int(cardinality_threshold),
    }

    st.download_button(
        "⬇️ Download Processing Report",
        data=json.dumps(report, indent=2, default=str),
        file_name="preplab_report.json",
        mime="application/json",
        use_container_width=True,
    )


st.divider()
st.caption("PrepLab | Python • Streamlit • LangGraph • Groq")