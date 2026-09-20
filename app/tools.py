"""Task execution tools — ChatGPT-style "tool use" for runnable tasks.

Each tool detects its own input patterns and executes deterministically:
- calc   : arithmetic expressions (+ - * / % ^ parentheses), guarded eval
- time   : current date/time questions
- convert: unit conversion (km↔miles, kg↔lb, °C↔°F, …)
"""
from __future__ import annotations

import ast
import datetime as dt
import operator as op
import re
from typing import Optional, Tuple

# --------------------------------------------------------------------------- #
# calculator — safe arithmetic evaluator (never eval()s raw text)
# --------------------------------------------------------------------------- #
_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Mod: op.mod, ast.Pow: op.pow, ast.FloorDiv: op.floordiv}
_UNARY = {ast.USub: op.neg, ast.UAdd: op.pos}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


def safe_calc(expr: str) -> float:
    """Evaluate an arithmetic expression safely (no names/calls allowed)."""
    expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/")
    tree = ast.parse(expr, mode="eval")
    result = _eval_node(tree)
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return result


# --------------------------------------------------------------------------- #
# detection patterns
# --------------------------------------------------------------------------- #
_CALC_RE = re.compile(
    r"^[\s\d+\-*/%^().×÷]+$"
    r"|^(?:what(?:'s| is)|compute|calculate|solve|eval(?:uate)?)\s+(.+?)[?！?]*$",
    re.I)

_TIME_RE = re.compile(
    r"\b(what(?:'s| is)?\s+(?:the\s+)?(?:time|date|day)|time\s+now|today[''s]?\s+date)\b"
    r"|දැන්\s*(වෙලාව|වේලාව|දිනය|දිනය)\s*මොකක්ද"
    r"|අද\s*(දිනය|දවස|දින|දිනය)\s*මොකක්ද"
    r"|¿?qué\s+hora|quelle\s+heure|wie\s+spät|который\s+час|saat\s+kaç|كم\s+الساعة|समय\s+क्या|几点|今何時|몇\s+시",
    re.I)

_UNIT_RE = re.compile(
    r"(-?\d+(?:[.,]\d+)?)\s*"
    r"(km|kilometers?|kilometres?|mi|mile|miles|m|meter|meters?|metres?|cm|mm|"
    r"kg|kilos?|kilograms?|lb|lbs|pounds?|g|grams?|"
    r"c|celsius|f|fahrenheit|°c|°f)\s*"
    r"(?:to|in|as|=|→|->|ට|දක්වා)\s*"
    r"(km|kilometers?|kilometres?|mi|mile|miles|m|meter|meters?|metres?|cm|mm|"
    r"kg|kilos?|kilograms?|lb|lbs|pounds?|g|grams?|"
    r"c|celsius|f|fahrenheit|°c|°f)\b",
    re.I)

_FACTORS = {
    ("km", "m"): 1000.0, ("m", "km"): 0.001, ("km", "mi"): 0.621371,
    ("mi", "km"): 1.609344, ("m", "mi"): 0.000621371, ("cm", "m"): 0.01,
    ("m", "cm"): 100.0, ("mm", "cm"): 0.1, ("cm", "mm"): 10.0,
    ("kg", "lb"): 2.2046226, ("lb", "kg"): 0.45359237,
    ("kg", "g"): 1000.0, ("g", "kg"): 0.001, ("lb", "g"): 453.59237,
}


def _norm_unit(u: str) -> str:
    u = u.lower().strip("°")
    return {"kilometer": "km", "kilometers": "km", "kilometres": "km",
            "mile": "mi", "miles": "mi", "meter": "m", "meters": "m",
            "metre": "m", "metres": "m", "kilo": "kg", "kilos": "kg",
            "kilogram": "kg", "kilograms": "kg", "pound": "lb", "pounds": "lb",
            "gram": "g", "grams": "g", "celsius": "c", "fahrenheit": "f",
            "farenheit": "f"}[u] if u in ("kilometer", "kilometers", "kilometres",
            "mile", "miles", "meter", "meters", "metre", "metres", "kilo",
            "kilos", "kilogram", "kilograms", "pound", "pounds", "gram",
            "grams", "celsius", "fahrenheit", "farenheit") else u


def convert(value: float, src: str, dst: str) -> Optional[float]:
    src, dst = _norm_unit(src), _norm_unit(dst)
    if src == dst:
        return value
    if src == "c" and dst == "f":
        return value * 9 / 5 + 32
    if src == "f" and dst == "c":
        return (value - 32) * 5 / 9
    if (src, dst) in _FACTORS:
        return value * _FACTORS[(src, dst)]
    return None


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def detect(text: str) -> Optional[Tuple[str, str]]:
    """Detect a runnable task. Returns (tool, argument) or None."""
    t = text.strip()

    m = _UNIT_RE.search(t)
    if m:
        return "convert", f"{m.group(1)} {m.group(2)} to {m.group(3)}"

    m = _CALC_RE.match(t)
    if m:
        expr = m.group(1) if m.lastindex else t
        if re.search(r"\d", expr) and re.search(r"[+\-*/%^×÷]", expr):
            return "calc", expr

    if _TIME_RE.search(t) and len(t) < 60:
        return "time", t

    return None


def _fmt(x) -> str:
    """Format numbers with thousand separators (works for int and float)."""
    if isinstance(x, float):
        s = f"{x:,.4f}".rstrip("0").rstrip(".")
        return s if s not in ("", "-") else "0"
    return f"{x:,}"


def run(tool: str, arg: str) -> str:
    """Execute a detected task and return a human-readable result."""
    try:
        if tool == "calc":
            result = safe_calc(arg)
            return f"{arg.strip()} = {_fmt(result)}"
        if tool == "time":
            now = dt.datetime.now().astimezone()
            off = int(now.utcoffset().total_seconds() // 3600)
            return (f"📅 {now.strftime('%A, %d %B %Y — %H:%M:%S')} "
                    f"(UTC{off:+d})")
        if tool == "convert":
            m = _UNIT_RE.search(arg)
            if not m:
                return "I couldn't parse that conversion."
            value, src, dst = float(m.group(1).replace(",", ".")), m.group(2), m.group(3)
            out = convert(value, src, dst)
            if out is None:
                return f"unsupported conversion: {_norm_unit(src)} → {_norm_unit(dst)}"
            return f"{_fmt(value)} {_norm_unit(src)} = {_fmt(out)} {_norm_unit(dst)}"
    except Exception:
        return "I couldn't compute that — try a simpler expression."
    return "unknown tool"
