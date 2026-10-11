"""
agent/agentic.py - LLM-routed LangGraph for PrepLab ("agent mode").

How it works
------------
* Every processing node, including the former setup nodes, is an LLM-selectable tool.
* The graph starts at the LLM router (class `Router`).
  On every turn it reads the user's request, a profile of the *current*
  DataFrame and the steps already taken, then chooses ONE next node.
* Every worker node returns to `decision`; `finish` ends the graph.
* The LLM can only choose registered nodes (TOOLS). Choices are validated
  (prerequisites, repeat limits, step cap). Missing LLM configuration or
  invalid LLM routes are reported as errors; there is no rule-based fallback.
* Every routing decision is stored in a `RouteRecorder` (step, chosen node,
  the model's stated reason, source, validation notes, latency). The Streamlit
  app draws this log as the route flowchart.

The worker nodes are NOT re-implemented. They are taken from the graph that
`agent.graph.build_graph()` already builds, so your existing tools are reused.

Requirements: `pip install langgraph groq`
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agent.graph import build_graph
from agent.nodes import build_encoding_plan

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# If introspection of build_graph() cannot find your state schema, set it here:
#     from agent.state import AgentState
#     STATE_SCHEMA = AgentState
STATE_SCHEMA: Any = None

# If a worker cannot be taken from build_graph(), map it here explicitly:
#     from agent.nodes import some_function
#     WORKER_OVERRIDES = {"check_missing_values": some_function}
WORKER_OVERRIDES: dict[str, Callable] = {}


# =========================================================
# 1. TOOL REGISTRY  (what the LLM may choose from)
#    `requires` mirrors the order of the existing flow; edit it to match
#    what your nodes really need.
# =========================================================

@dataclass(frozen=True)
class Tool:
    name: str
    summary: str
    requires: tuple = ()
    hint: str = ""


TOOLS: list[Tool] = [
    Tool("problem_definition",
         "Analyze the dataset to describe the likely ML task, target and data-quality concerns.",
         (), "Run only when the user asks for problem definition or dataset recommendations."),
    Tool("inspect_data",
         "Inspect dataset shape, data types, missing values, duplicates and cardinality.",
         (), "Run only when the user asks to inspect or summarize the dataset."),
    Tool("clean_columns",
         "Clean column names and assess columns for removal or review.",
         (), "Run only when the user asks to clean or select columns."),
    Tool("duplicate_handling",
         "Find and remove duplicate rows.",
         (), "Run only when the user asks to handle duplicates."),
    Tool("type_correction",
         "Convert explicitly requested columns to specified data types.",
         (), "Use only conversions specified by the user's request."),
    Tool("check_missing_values",
         "Find columns with missing values and classify them as categorical or numerical.",
         (), "Run first for missing-value checks or imputation; skip for outlier-only requests."),
    Tool("mode_imputation",
         "Fill categorical missing values with the mode.",
         ("check_missing_values",), "Only useful if categorical columns have missing values."),
    Tool("check_bell_shaped",
         "Decide which numerical columns are bell-shaped (selects Z-score vs IQR).",
         (), "First step when the user requests outlier detection or numerical missing-value handling."),
    Tool("detect_outliers",
         "Detect outliers: Z-score for bell-shaped columns, IQR for the others.",
         ("check_bell_shaped",), ""),
    Tool("outlier_capping",
         "Cap extreme values in the columns where outliers were found.",
         ("detect_outliers",),
         "Run only if outliers were found; impute missing values first when requested."),
    Tool("median_imputation",
         "Fill numerical missing values with the median.",
         ("check_missing_values",), "For skewed columns, or after outlier capping."),
    Tool("check_skewness",
         "Measure skewness of numerical columns to choose median vs mean imputation.",
         ("detect_outliers",), "Use for numerical columns with no detected outliers."),
    Tool("mean_imputation",
         "Fill numerical missing values with the mean.",
         ("check_skewness",), "For symmetric columns."),
    Tool("encoding_plan_ready",
         "Build the encoding plan (ordinal / frequency / one-hot).",
         (), "Run before any encoding tool."),
    Tool("ordinal_encoding",
         "Apply explicitly supplied ordinal mappings.",
         ("encoding_plan_ready",), ""),
    Tool("frequency_encoding",
         "Encode high-cardinality categorical columns.",
         ("encoding_plan_ready",), ""),
    Tool("one_hot_encoding",
         "Encode the remaining low-cardinality categorical columns.",
         ("encoding_plan_ready",), ""),
    Tool("feature_scaling",
         "Scale numerical feature columns (target excluded) with the configured method.",
         (), "Only if the user explicitly asks for scaling."),
]

FINISH = Tool("finish", "End the workflow and record the final dataset.", (),
              "Choose as soon as the request is fully satisfied.")

TOOL_BY_NAME = {t.name: t for t in TOOLS}
TOOL_NAMES = [t.name for t in TOOLS]
MAX_REPEAT = 1
MAX_MEDIAN_IMPUTATION_RUNS = 2
WORKFLOWS = {
    "encoding",
    "missing_check",
    "missing_imputation",
    "outlier_detection",
    "outlier_capping",
    "other",
}


class MissingRouteToolCallError(RuntimeError):
    """The model returned a response without calling the route-selection tool."""


# =========================================================
# 2. ROUTE RECORDER
# =========================================================

@dataclass
class RouteRecorder:
    """Collects every routing decision made during one run."""

    request: str = ""
    decisions: list = field(default_factory=list)

    def add(self, **entry):
        self.decisions.append(entry)


# =========================================================
# 3. DATASET PROFILE (what the LLM sees)
# =========================================================

def _profile(state: dict) -> dict:
    df = state.get("current_df")
    out: dict[str, Any] = {}
    if not isinstance(df, pd.DataFrame):
        return out

    num_cols = list(df.select_dtypes(include="number").columns)
    cat_cols = [c for c in df.columns if c not in num_cols]
    missing = {str(c): int(n) for c, n in df.isna().sum().items() if n}
    bell_results = state.get("bell_shaped_results", {}).get(
        "results", state.get("bell_shaped_results", {})
    )
    outlier_report = state.get("outlier_report", {})
    outlier_results = outlier_report.get("results", {})
    skewness_results = state.get("skewness_results", {})
    encoding_plan = state.get("encoding_plan", {})
    target_column = state.get("target_column")
    encoding_candidates = [
        column for column in df.select_dtypes(
            include=["object", "category", "string"]
        ).columns
        if column != target_column
    ]

    out.update({
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "target_column": str(target_column) if target_column is not None else None,
        "column_names": [str(c) for c in df.columns[:50]],
        "column_dtypes": {str(c): str(df[c].dtype) for c in df.columns[:50]},
        "encoding_candidates": {
            str(column): {
                "unique_count": int(df[column].nunique(dropna=True)),
                "sample_values": [
                    str(value) for value in df[column].dropna().unique()[:50]
                ],
            }
            for column in encoding_candidates[:50]
        },
        "high_cardinality_threshold": int(
            state.get("high_cardinality_threshold", 10)
        ),
        "numeric_columns": [str(c) for c in num_cols[:30]],
        "categorical_columns_with_unique_counts": {
            str(c): int(df[c].nunique(dropna=True)) for c in cat_cols[:30]
        },
        "categorical_column_values": {
            str(column): [str(value) for value in df[column].dropna().unique()[:30]]
            for column in cat_cols[:30]
        },
        "columns_with_missing_values": dict(list(missing.items())[:30]),
        "total_missing": int(sum(missing.values())),
        "categorical_missing_columns": [
            str(column) for column in df.columns
            if int(df[column].isna().sum()) and column in cat_cols
        ],
        "numerical_missing_columns": [
            str(column) for column in df.columns
            if int(df[column].isna().sum()) and column in num_cols
        ],
        "categorical_imputation_done": state.get(
            "categorical_imputation_done", False
        ),
        "bell_shaped_results": {
            str(column): {
                "is_bell_shaped": details.get("is_bell_shaped"),
                "reason": details.get("reason"),
            }
            for column, details in list(bell_results.items())[:30]
            if isinstance(details, dict)
        },
        "outlier_report": {
            "results": {
                str(column): {"outlier_count": details.get("outlier_count", 0)}
                for column, details in list(outlier_results.items())[:30]
            }
        },
        "skewness_results": {
            str(column): {
                "classification": details.get("classification"),
                "skewness": details.get("skewness"),
            }
            for column, details in list(skewness_results.items())[:30]
            if isinstance(details, dict)
        },
        "encoding_plan": {
            strategy: [str(column) for column in columns[:30]]
            for strategy, columns in encoding_plan.items()
        },
        "feature_scaling_done": state.get("feature_scaling_done", False),
    })
    return out


def _outlier_columns(state: dict) -> list[str]:
    results = state.get("outlier_report", {}).get("results", {})
    return [
        column for column, details in results.items()
        if details.get("outlier_count", 0) > 0
    ]


def _no_outlier_columns(state: dict) -> list[str]:
    detected = set(_outlier_columns(state))
    return [
        column for column in state.get("numerical_missing_columns", [])
        if column not in detected
    ]


def _skewed_columns(state: dict) -> list[str]:
    results = state.get("skewness_results", {})
    columns = []
    for column in _no_outlier_columns(state):
        details = results.get(column, {})
        if isinstance(details, dict):
            if "is_skewed" in details:
                skewed = bool(details["is_skewed"])
            elif details.get("classification") in {"skewed", "undetermined"}:
                skewed = True
            else:
                value = details.get("skewness")
                skewed = value is not None and abs(float(value)) > 0.5
        elif isinstance(details, (int, float)):
            skewed = abs(float(details)) > 0.5
        else:
            skewed = False
        if skewed:
            columns.append(column)
    return columns


def _missing_imputation_complete(state: dict, history: list[str]) -> bool:
    if "check_missing_values" not in history:
        return False

    categorical = set(state.get("categorical_missing_columns", []))
    if categorical and "mode_imputation" not in history:
        return False

    numerical = set(state.get("numerical_missing_columns", []))
    if not numerical:
        return True
    if "check_bell_shaped" not in history or "detect_outliers" not in history:
        return False

    outliers = set(_outlier_columns(state))
    outlier_missing = outliers.intersection(numerical)
    median_done = set(state.get("outlier_median_done_columns", []))
    capped = set(state.get("capped_columns", []))
    if not outlier_missing.issubset(median_done) or not outlier_missing.issubset(capped):
        return False

    no_outliers = numerical - outliers
    if no_outliers:
        if "check_skewness" not in history:
            return False
        skewed = set(_skewed_columns(state)).intersection(no_outliers)
        symmetric = no_outliers - skewed
        if not skewed.issubset(state.get("skewness_median_done_columns", [])):
            return False
        if not symmetric.issubset(state.get("mean_done_columns", [])):
            return False

    return True


def _encoding_complete(state: dict, history: list[str]) -> bool:
    if "encoding_plan_ready" not in state.get("completed_steps", []):
        return False

    plan = state.get("encoding_plan", {})
    completed = {
        "ordinal": set(state.get("ordinal_encoded_columns", [])),
        "frequency": set(state.get("frequency_encoded_columns", [])),
        "one_hot": set(state.get("one_hot_encoded_columns", [])),
    }
    return all(
        set(plan.get(strategy, [])).issubset(completed[strategy])
        for strategy in completed
    )


# =========================================================
# 4. THE ROUTER (replaces the fixed decision node)
# =========================================================

SYSTEM_PROMPT = (
    "You are the routing brain of PrepLab, a data-preprocessing agent built on LangGraph. "
    "You, the LLM, decide which tool is needed from the user's request and current data. "
    "On every turn call select_next_node exactly once to choose ONE next node to run; "
    "re-evaluate after each tool result. The workflow rules below are guidance for "
    "selecting relevant tools and required branches, not instructions to run every "
    "node in a fixed pipeline. The application validates choices and blocks invalid "
    "routes, but does not select a substitute tool for you. Do not return prose.\n"
    "Routing rules:\n"
    "- next_node MUST be one of allowed_nodes. Never invent names.\n"
    "- Do only what the user's request explicitly asks. Do not run setup, inspection, "
    "cleanup, or other tools automatically. Execute any prerequisite nodes required "
    "by the requested flow, and all separately requested tasks, then stop. For a "
    "request containing multiple tasks, keep routing until each requested task is "
    "complete; do not finish just because one selected tool has run.\n"
    "- For 'fill/impute missing values', follow this exact flow: check_missing_values; "
    "if categorical missing columns exist, mode_imputation; if numerical missing "
    "columns exist, check_bell_shaped -> detect_outliers. For detected outliers, "
    "median_imputation -> outlier_capping. For numerical columns without outliers, "
    "check_skewness -> median_imputation if skewed or mean_imputation if symmetric. "
    "Process every applicable column and branch before finish. Skip empty branches. "
    "Never choose finish immediately after check_missing_values for a fill/impute request.\n"
    "- Set workflow to missing_imputation for fill/impute requests, missing_check for "
    "check/report-only requests, outlier_detection for detect-only requests, "
    "outlier_capping when capping is requested, and other for other requests. Keep "
    "the same workflow value for every routing decision in the run.\n"
    "- For a request to only check/report missing values, run check_missing_values "
    "and finish without imputation.\n"
    "- For an outlier-only request, use check_bell_shaped -> detect_outliers; cap "
    "only if the user asks to cap, and impute first only when that request includes "
    "filling missing values.\n"
    "- For an encoding request, set workflow to encoding. Run encoding_plan_ready, "
    "then apply every encoder listed in the resulting plan before finish. YOU must "
    "choose the method for each requested categorical feature and provide it in "
    "requested_encoding_plan when selecting encoding_plan_ready; do not rely on a "
    "fixed default. Use dataset_profile.encoding_candidates (candidate columns, "
    "unique counts, and sample values), high_cardinality_threshold, and the user "
    "query: use ordinal when category values have a clear, unambiguous natural "
    "order (for example low/medium/high, education levels, or ordered ratings), "
    "and provide the corresponding ordinal_mappings from lowest to highest. Do not "
    "treat arbitrary labels or IDs as ordered; do not infer an order when it is "
    "ambiguous. For remaining nominal features use frequency when unique_count meets "
    "or exceeds the profile threshold, and one-hot for low-cardinality features. "
    "An explicitly requested method in the user's query takes precedence when valid. "
    "If the user names specific columns, include exactly those "
    "eligible columns in requested_encoding_columns; otherwise include all eligible "
    "categorical feature columns. Exclude the target. Put every selected column in "
    "exactly one of ordinal, frequency, or one_hot. Do not put every column into "
    "one_hot by default.\n"
    "- For standalone requests such as inspection, problem definition, column "
    "cleaning, duplicate handling, type correction, or scaling, select the matching "
    "tool only. For other or compound requests, use the user request, tool summaries, "
    "dataset profile, and completed results to select each necessary next tool. Follow "
    "only the relevant tools and their declared workflow prerequisites. Never run "
    "unrelated setup, encoding, or scaling nodes.\n"
    "- Do not repeat a completed node unless clearly needed.\n"
    "- Choose 'finish' as soon as the request is fully satisfied.\n"
    "- For type_correction, include type_conversions only for column/type conversions "
    "explicitly specified by the user. Supported types: numeric, integer, float, "
    "string, category, datetime, boolean.\n"
    "- For every column placed in the ordinal plan, include its complete "
    "ordinal_mappings in the same tool call. These may come from an explicit user "
    "mapping or an unambiguous natural ordering in the observed category labels. "
    "Never fabricate ordering for nominal labels. "
    "When the user requests scaling, include scaling_method: standard, minmax, "
    "or robust, matching the requested method.\n"
    "- Before finish, ensure every applicable branch of the requested flow has run. "
    "Never continue into encoding or scaling unless the user explicitly asks for it.\n"
    "- The reason must reference the user's request or the dataset facts you were given."
)


class Router:
    """LLM-driven decision node with validation and no non-LLM routing."""

    def __init__(self, user_request: str, recorder: RouteRecorder, api_key: str = "",
                 model: str = DEFAULT_MODEL, max_steps: int = 25):
        self.user_request = (user_request or "").strip()
        self.recorder = recorder
        self.api_key = (api_key or "").strip()
        self.model = model or DEFAULT_MODEL
        self.max_steps = int(max_steps)
        self.history: list[str] = []
        self.requested_workflow: Optional[str] = None

    # ---------- validation ----------
    def _allowed(self, state: dict) -> dict[str, Tool]:
        done = set(self.history)
        allowed: dict[str, Tool] = {}
        for tool in TOOLS:
            repeat_limit = (
                MAX_MEDIAN_IMPUTATION_RUNS
                if tool.name == "median_imputation"
                else MAX_REPEAT
            )
            if self.history.count(tool.name) >= repeat_limit:
                continue
            if not set(tool.requires) <= done:
                continue
            if tool.name == "mode_imputation" and not state.get(
                "categorical_missing_columns"
            ):
                continue
            if tool.name == "check_bell_shaped" and not state.get(
                "numerical_missing_columns"
            ) and not _profile(state).get("numeric_columns"):
                continue
            if tool.name == "check_bell_shaped" and state.get(
                "missing_values_checked"
            ) and not state.get(
                "numerical_missing_columns"
            ):
                continue
            if tool.name == "detect_outliers" and not state.get(
                "numerical_missing_columns"
            ) and not _profile(state).get("numeric_columns"):
                continue
            if tool.name == "detect_outliers" and state.get(
                "missing_values_checked"
            ) and not state.get(
                "numerical_missing_columns"
            ):
                continue
            if tool.name == "outlier_capping":
                outliers = set(_outlier_columns(state))
                median_done = set(state.get("outlier_median_done_columns", []))
                missing_outliers = outliers.intersection(
                    state.get("numerical_missing_columns", [])
                )
                capped = set(state.get("capped_columns", []))
                if (
                    not outliers
                    or not missing_outliers.issubset(median_done)
                    or outliers.issubset(capped)
                ):
                    continue
            if tool.name == "check_skewness":
                skewness_checked = "check_skewness" in done
                if not _no_outlier_columns(state) or skewness_checked:
                    continue
            if tool.name == "median_imputation":
                outliers_pending = set(_outlier_columns(state)).intersection(
                    state.get("numerical_missing_columns", [])
                ) - set(
                    state.get("outlier_median_done_columns", [])
                )
                skewed_pending = set(_skewed_columns(state)) - set(
                    state.get("skewness_median_done_columns", [])
                )
                if not outliers_pending and not skewed_pending:
                    continue
            if tool.name == "mean_imputation":
                symmetric = set(_no_outlier_columns(state)) - set(
                    _skewed_columns(state)
                )
                if not symmetric or symmetric.issubset(
                    set(state.get("mean_done_columns", []))
                ):
                    continue
            if tool.name == "ordinal_encoding" and not state.get(
                "encoding_plan", {}
            ).get("ordinal"):
                continue
            if tool.name == "frequency_encoding" and not state.get(
                "encoding_plan", {}
            ).get("frequency"):
                continue
            if tool.name == "one_hot_encoding" and not state.get(
                "encoding_plan", {}
            ).get("one_hot"):
                continue
            if tool.name == "feature_scaling" and state.get(
                "feature_scaling_done"
            ):
                continue
            if tool.name == "encoding_plan_ready" and (
                self.requested_workflow not in {None, "encoding"}
                or "encoding_plan_ready" in done
            ):
                continue
            if tool.name in {
                "ordinal_encoding",
                "frequency_encoding",
                "one_hot_encoding",
            } and self.requested_workflow != "encoding":
                continue
            allowed[tool.name] = tool
        workflow_complete = {
            "missing_check": "check_missing_values" in done,
            "missing_imputation": _missing_imputation_complete(state, self.history),
            "encoding": _encoding_complete(state, self.history),
            "outlier_detection": (
                "detect_outliers" in done
                or not _profile(state).get("numeric_columns")
            ),
            "outlier_capping": (
                not _profile(state).get("numeric_columns")
                or (
                    "detect_outliers" in done
                    and set(_outlier_columns(state)).issubset(
                        set(state.get("capped_columns", []))
                    )
                )
            ),
            "other": bool(self.history),
        }
        if workflow_complete.get(self.requested_workflow, bool(self.history)):
            allowed[FINISH.name] = FINISH
        return allowed

    def _why_not(self, name: str) -> str:
        if name not in TOOL_BY_NAME and name != FINISH.name:
            return "not a registered node"
        if name == FINISH.name:
            return "the requested workflow has not completed"
        repeat_limit = (
            MAX_MEDIAN_IMPUTATION_RUNS if name == "median_imputation" else MAX_REPEAT
        )
        if self.history.count(name) >= repeat_limit:
            return f"already run {self.history.count(name)} times"
        missing = [r for r in TOOL_BY_NAME[name].requires if r not in self.history]
        return f"prerequisites not run yet: {', '.join(missing)}" if missing else "not allowed now"

    # ---------- LLM ----------
    def _llm_ready(self) -> None:
        if not (self.api_key or os.environ.get("GROQ_API_KEY")):
            raise RuntimeError(
                "Agent mode requires a Groq API key. Configure it in the sidebar "
                "or set GROQ_API_KEY in the environment."
            )
        try:
            import groq  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Agent mode requires the 'groq' package. Install it with pip install groq."
            ) from exc

    def _ask(self, state: dict, profile: dict, allowed: dict, feedback: Optional[str]) -> str:
        from groq import Groq

        client = Groq(api_key=self.api_key or os.environ.get("GROQ_API_KEY"))

        payload = {
            "user_request": self.user_request,
            "workflow_selected_so_far": self.requested_workflow,
            "dataset_profile": profile,
            "steps_so_far": [
                {"node": d["chosen"], "why": d.get("reason", "")}
                for d in self.recorder.decisions
            ],
            "allowed_nodes": {
                name: {"summary": t.summary, "hint": t.hint} for name, t in allowed.items()
            },
        }
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, default=str)},
        ]
        if feedback:
            messages.append({"role": "user", "content": feedback})

        route_tool = {
            "type": "function",
            "function": {
                "name": "select_next_node",
                "description": (
                    "Select the next workflow node. For encoding_plan_ready, choose "
                    "a method for each requested column and include ordinal mappings "
                    "for clearly ordered category labels."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "next_node": {
                            "type": "string",
                            "enum": TOOL_NAMES + [FINISH.name],
                        },
                        "reason": {"type": "string"},
                        "workflow": {
                            "type": "string",
                            "enum": sorted(WORKFLOWS),
                        },
                        "type_conversions": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                        "ordinal_mappings": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "object",
                                "additionalProperties": {"type": "number"},
                            },
                        },
                        "requested_encoding_columns": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "requested_encoding_plan": {
                            "type": "object",
                            "properties": {
                                "ordinal": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "frequency": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "one_hot": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["ordinal", "frequency", "one_hot"],
                            "additionalProperties": False,
                        },
                        "scaling_method": {
                            "type": "string",
                            "enum": ["standard", "minmax", "robust"],
                        },
                    },
                    "required": ["next_node", "reason", "workflow"],
                    "additionalProperties": False,
                },
            },
        }
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=1024,
            tools=[route_tool],
            tool_choice="required",
            parallel_tool_calls=False,
        )
        message = response.choices[0].message
        if not message.tool_calls:
            raise MissingRouteToolCallError(
                f"The LLM did not return the required select_next_node call "
                f"(finish_reason={response.choices[0].finish_reason!r})."
            )
        if len(message.tool_calls) != 1:
            raise RuntimeError(
                f"The LLM returned {len(message.tool_calls)} function calls; "
                "exactly one select_next_node call is required."
            )
        call = message.tool_calls[0]
        if call.function.name != "select_next_node":
            raise RuntimeError(
                f"The LLM returned unexpected function {call.function.name!r}; "
                "expected 'select_next_node'."
            )
        return call.function.arguments or ""

    # ---------- LangGraph node ----------
    def node(self, state: dict) -> dict:
        t0 = time.time()
        step = len(self.recorder.decisions) + 1
        allowed = self._allowed(state)
        profile = _profile(state)
        notes: list[str] = []
        chosen: Optional[str] = None
        reason = ""
        source = "llm"
        update: dict = {}

        if step > self.max_steps:
            raise RuntimeError(
                f"Agent router reached its {self.max_steps}-step safety limit "
                "without the LLM selecting finish."
            )

        self._llm_ready()
        feedback = None
        for attempt in (1, 2):
            try:
                raw = self._ask(state, profile, allowed, feedback)
            except MissingRouteToolCallError as exc:
                notes.append(f"Attempt {attempt}: {exc}")
                feedback = (
                    "Your previous response did not call select_next_node. "
                    "Call select_next_node now with one allowed next_node and a workflow."
                )
                continue
            try:
                data = json.loads(raw)
                cand = str(data.get("next_node", "")).strip()
                why = str(data.get("reason", "")).strip()
            except (json.JSONDecodeError, AttributeError) as exc:
                notes.append(
                    f"Attempt {attempt}: invalid select_next_node arguments "
                    f"({type(exc).__name__}: {exc})."
                )
                feedback = (
                    "Call select_next_node again with valid arguments matching its schema."
                )
                continue

            workflow = data.get("workflow")
            if workflow not in WORKFLOWS:
                notes.append(f"Attempt {attempt}: invalid workflow {workflow!r}.")
                feedback = f"workflow must be one of: {', '.join(sorted(WORKFLOWS))}."
                continue
            if (
                self.requested_workflow is not None
                and workflow != self.requested_workflow
            ):
                notes.append(
                    f"Attempt {attempt}: workflow changed from "
                    f"{self.requested_workflow!r} to {workflow!r}."
                )
                feedback = (
                    f"Keep workflow set to {self.requested_workflow!r} for this run."
                )
                continue
            self.requested_workflow = workflow

            if cand in allowed:
                chosen, reason = cand, why or "(no reason given)"
                if chosen == "encoding_plan_ready" and workflow != "encoding":
                    notes.append(
                        f"Attempt {attempt}: encoding_plan_ready requires workflow='encoding'."
                    )
                    chosen = None
                    feedback = (
                        "Choose encoding_plan_ready only with workflow set to encoding."
                    )
                    continue
                if chosen == "encoding_plan_ready":
                    requested_columns = data.get("requested_encoding_columns")
                    plan = data.get("requested_encoding_plan")
                    if not isinstance(requested_columns, list) or not all(
                        isinstance(column, str) for column in requested_columns
                    ):
                        notes.append(
                            f"Attempt {attempt}: requested_encoding_columns must be a list of column names."
                        )
                        chosen = None
                        feedback = (
                            "When selecting encoding_plan_ready, provide "
                            "requested_encoding_columns as an array (empty means all "
                            "eligible categorical features)."
                        )
                        continue
                    if not isinstance(plan, dict) or set(plan) != {
                        "ordinal",
                        "frequency",
                        "one_hot",
                    } or not all(
                        isinstance(columns, list)
                        and all(isinstance(column, str) for column in columns)
                        for columns in plan.values()
                    ):
                        notes.append(
                            f"Attempt {attempt}: requested_encoding_plan must contain ordinal, frequency, and one_hot column lists."
                        )
                        chosen = None
                        feedback = (
                            "When selecting encoding_plan_ready, choose the method "
                            "per feature and provide requested_encoding_plan with "
                            "ordinal, frequency, and one_hot arrays."
                        )
                        continue
                    update["requested_encoding_columns"] = requested_columns
                    update["requested_encoding_plan"] = plan

                conversions = data.get("type_conversions")
                if chosen == "type_correction" and conversions is not None:
                    if not isinstance(conversions, dict) or not conversions or not all(
                        isinstance(column, str) and isinstance(value, str)
                        for column, value in conversions.items()
                    ):
                        notes.append(
                            f"Attempt {attempt}: type_conversions must map column names to type names."
                        )
                        chosen = None
                        feedback = (
                            "When selecting type_correction, type_conversions must be a JSON "
                            "object mapping column names to supported type names."
                        )
                        continue
                    update["type_conversions"] = conversions
                elif chosen == "type_correction":
                    notes.append(
                        f"Attempt {attempt}: type_conversions are required for type correction."
                    )
                    chosen = None
                    feedback = (
                        "When selecting type_correction, provide a non-empty "
                        "type_conversions object mapping each requested column to its type."
                    )
                    continue
                mappings = data.get("ordinal_mappings")
                if mappings is not None:
                    if (
                        workflow == "encoding"
                        and "encoding_plan_ready" in state.get("completed_steps", [])
                    ):
                        mappings = None
                if mappings is not None:
                    if not isinstance(mappings, dict) or not all(
                        isinstance(column, str)
                        and isinstance(mapping, dict)
                        and all(
                            isinstance(value, (int, float)) and not isinstance(value, bool)
                            for value in mapping.values()
                        )
                        for column, mapping in mappings.items()
                    ):
                        notes.append(
                            f"Attempt {attempt}: ordinal_mappings must map column names to value mappings."
                        )
                        chosen = None
                        feedback = (
                            "ordinal_mappings must be an object mapping column names "
                            "to objects of category-to-number mappings."
                        )
                        continue
                    update["ordinal_mappings"] = mappings
                if chosen == "encoding_plan_ready":
                    candidate_state = dict(state)
                    candidate_state.update(update)
                    try:
                        build_encoding_plan(candidate_state)
                    except (TypeError, ValueError) as exc:
                        notes.append(
                            f"Attempt {attempt}: invalid encoding plan ({exc})."
                        )
                        update.pop("requested_encoding_columns", None)
                        update.pop("requested_encoding_plan", None)
                        if "ordinal_mappings" not in state:
                            update.pop("ordinal_mappings", None)
                        chosen = None
                        feedback = (
                            "Revise the per-column encoding plan so it covers exactly "
                            "the selected categorical features once each. Ordinal "
                            "columns require explicit mappings that cover observed "
                            "categories. Use the cardinality threshold and workflow "
                            "rules when choosing methods."
                        )
                        continue
                scaling_method = data.get("scaling_method")
                if chosen == "feature_scaling":
                    if scaling_method not in {"standard", "minmax", "robust"}:
                        notes.append(
                            f"Attempt {attempt}: scaling_method is required when selecting feature_scaling."
                        )
                        chosen = None
                        feedback = (
                            "When selecting feature_scaling, provide scaling_method "
                            "as standard, minmax, or robust."
                        )
                        continue
                    update["scaling_method"] = scaling_method
                break

            notes.append(f"Attempt {attempt}: '{cand}' rejected - {self._why_not(cand)}.")
            feedback = (
                f"'{cand}' is not allowed ({self._why_not(cand)}). "
                f"Choose one of: {', '.join(allowed)}."
            )

        if chosen is None:
            raise RuntimeError(
                "The LLM did not select a valid route after two attempts. "
                + " ".join(notes)
            )

        self.recorder.add(
            step=step,
            from_node=self.history[-1] if self.history else "start",
            chosen=chosen,
            workflow=self.requested_workflow,
            reason=reason,
            source=source,
            model=self.model,
            notes=notes,
            latency_ms=int((time.time() - t0) * 1000),
            allowed=list(allowed),
            dataset={k: profile.get(k) for k in ("rows", "columns", "total_missing") if k in profile},
        )
        self.history.append(chosen)

        update["next_action"] = chosen
        update["requested_workflow"] = self.requested_workflow
        return update


# =========================================================
# 5. BUILD THE AGENTIC GRAPH FROM YOUR EXISTING NODES
# =========================================================

def _introspect_existing_graph():
    """Return the existing node runnables and state schema."""
    compiled = build_graph()
    builder = getattr(compiled, "builder", None)

    nodes: dict[str, Any] = {}
    if builder is not None and getattr(builder, "nodes", None):
        for name, spec in builder.nodes.items():
            runnable = getattr(spec, "runnable", None) or getattr(spec, "action", None)
            if runnable is not None:
                nodes[name] = runnable
    if not nodes:
        for name, pnode in getattr(compiled, "nodes", {}).items():
            bound = getattr(pnode, "bound", None)
            if bound is not None:
                nodes[name] = bound

    schema = STATE_SCHEMA
    if schema is None and builder is not None:
        schema = getattr(builder, "state_schema", None) or getattr(builder, "schema", None)

    nodes.update(WORKER_OVERRIDES)
    return nodes, schema


def _worker_with_context(name: str, runnable: Any):
    def run(state: dict) -> dict:
        worker_state = dict(state)
        if name in {"check_bell_shaped", "detect_outliers"}:
            if state.get("missing_values_checked"):
                columns = state.get("numerical_missing_columns", [])
            else:
                df = state["current_df"]
                columns = [
                    column for column in df.select_dtypes(include="number").columns
                    if not pd.api.types.is_bool_dtype(df[column])
                ]
            worker_state["numerical_missing_columns"] = columns
        elif name == "check_skewness":
            worker_state["imputation_columns"] = _no_outlier_columns(state)
        elif name == "median_imputation":
            pending_outliers = set(_outlier_columns(state)).intersection(
                state.get("numerical_missing_columns", [])
            ) - set(
                state.get("outlier_median_done_columns", [])
            )
            if pending_outliers:
                worker_state["imputation_columns"] = sorted(pending_outliers)
                worker_state["imputation_reason"] = "outliers"
            else:
                pending_skewed = set(_skewed_columns(state)) - set(
                    state.get("skewness_median_done_columns", [])
                )
                worker_state["imputation_columns"] = sorted(pending_skewed)
                worker_state["imputation_reason"] = "skewed"
        elif name == "outlier_capping":
            worker_state["imputation_columns"] = sorted(
                set(_outlier_columns(state)) - set(state.get("capped_columns", []))
            )
        elif name == "mean_imputation":
            worker_state["imputation_columns"] = sorted(
                set(_no_outlier_columns(state)) - set(_skewed_columns(state))
                - set(state.get("mean_done_columns", []))
            )
        elif name in {"ordinal_encoding", "frequency_encoding", "one_hot_encoding"}:
            encoding = {
                "ordinal_encoding": "ordinal",
                "frequency_encoding": "frequency",
                "one_hot_encoding": "one_hot",
            }[name]
            worker_state["encoding_columns"] = state.get(
                "encoding_plan", {}
            ).get(encoding, [])

        invoke = getattr(runnable, "invoke", None)
        return invoke(worker_state) if invoke else runnable(worker_state)

    return run


def build_agentic_graph(router: Router):
    """Compile the LLM-routed graph. `router` records every decision it makes."""
    nodes, schema = _introspect_existing_graph()

    needed = TOOL_NAMES + [FINISH.name]
    missing = [n for n in needed if n not in nodes]
    if missing:
        raise RuntimeError(
            "These nodes were not found in agent.graph.build_graph(): "
            f"{', '.join(missing)}. Map them in WORKER_OVERRIDES (agent/agentic.py)."
        )
    if schema is None:
        raise RuntimeError(
            "Could not read the graph's state schema. Set STATE_SCHEMA in agent/agentic.py "
            "to the state class used by build_graph()."
        )

    graph = StateGraph(schema)
    for name in needed:
        graph.add_node(
            name,
            _worker_with_context(name, nodes[name]) if name in TOOL_NAMES else nodes[name],
        )
    graph.add_node("decision", router.node)

    graph.add_edge(START, "decision")

    for name in TOOL_NAMES:
        graph.add_edge(name, "decision")

    routes = {name: name for name in TOOL_NAMES + [FINISH.name]}
    graph.add_conditional_edges("decision", lambda s: s["next_action"], routes)
    graph.add_edge(FINISH.name, END)

    return graph.compile()