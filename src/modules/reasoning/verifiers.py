from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


class JSONStreamVerifier:
    """
    Lightweight streaming JSON prefix verifier (approximate).
    Checks bracket balance and basic string/escape structure.
    Returns is_valid_prefix and whether a complete JSON object/array has been closed.
    """
    def __init__(self) -> None:
        pass

    def check_prefix(self, s: str) -> Tuple[bool, bool]:
        stack = []
        in_string = False
        escape = False
        complete = False
        for ch in s:
            if in_string:
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch in "{[":
                    stack.append(ch)
                elif ch in "}]":
                    if not stack:
                        return False, False
                    top = stack.pop()
                    if (top == "{" and ch != "}") or (top == "[" and ch != "]"):
                        return False, False
        if in_string:
            # Unclosed string is okay as prefix, not complete
            return True, False
        complete = len(stack) == 0 and any(ch in "}]" for ch in s)
        return True, complete


class SafeMathVerifier:
    """
    Verifies simple arithmetic expressions produce finite results.
    """
    def __init__(self) -> None:
        self._allowed_nodes = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.Pow, ast.Mod,
            ast.Load, ast.Call, ast.Name,
        )
        self._allowed_names = {
            "sqrt": math.sqrt, "abs": abs, "round": round,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log, "exp": math.exp,
        }

    def _safe_eval(self, expr: str) -> Optional[float]:
        try:
            node = ast.parse(expr, mode="eval")
            for n in ast.walk(node):
                if not isinstance(n, self._allowed_nodes):
                    return None
                if isinstance(n, ast.Call):
                    if not isinstance(n.func, ast.Name):
                        return None
                    if n.func.id not in self._allowed_names:
                        return None
                if isinstance(n, ast.Name) and n.id not in self._allowed_names:
                    return None
            return float(eval(compile(node, "<ast>", "eval"), {"__builtins__": {}}, self._allowed_names))
        except Exception:
            return None

    def check(self, expr: str) -> bool:
        v = self._safe_eval(expr)
        if v is None:
            return False
        if not math.isfinite(v):
            return False
        return True


