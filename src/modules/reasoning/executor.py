from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from .verifiers import SafeMathVerifier


CALC_OPEN = "[calc]"
CALC_CLOSE = "[/calc]"


class ReasoningExecutor:
    """
    Minimal executor for inline action blocks.
    Supports [calc] ... [/calc] arithmetic evaluation.
    """
    def __init__(self) -> None:
        self.math = SafeMathVerifier()

    def execute_blocks(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Finds [calc]...[/calc] blocks, evaluates, and replaces with result.
        Returns (new_text, exec_log)
        """
        exec_log: Dict[str, Any] = {"calc": []}
        out = text
        pattern = re.compile(re.escape(CALC_OPEN) + r"(.*?)" + re.escape(CALC_CLOSE), re.DOTALL)
        def _repl(m: re.Match[str]) -> str:
            expr = m.group(1).strip()
            ok = self.math.check(expr)
            result = None
            if ok:
                # SafeMathVerifier.check already evaluated once; evaluate again for value
                # We re-use internal method via a tiny hack (not exposed)
                from .verifiers import SafeMathVerifier as _SMV  # type: ignore
                smv = _SMV()
                result = smv._safe_eval(expr)  # type: ignore[attr-defined]
            exec_log["calc"].append({"expr": expr, "ok": ok, "result": result})
            return str(result) if (ok and result is not None) else "[calc_error]"
        out = pattern.sub(_repl, out)
        return out, exec_log


