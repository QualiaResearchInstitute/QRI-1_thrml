from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from modules.guidance.poe_decoding import guided_generate
from .executor import ReasoningExecutor
from .verifiers import JSONStreamVerifier, SafeMathVerifier


class MicroSearch:
    """
    Tiny beam search orchestrator that tries a couple of branches:
    - Plain guided generation
    - Generation that inserts a [calc]...[/calc] block when math is detected
    Scores branches by a simple combination of NLL and verifier passes.
    """
    def __init__(self) -> None:
        self.executor = ReasoningExecutor()
        self.json_verifier = JSONStreamVerifier()
        self.math_verifier = SafeMathVerifier()

    def _nll_score(self, logits: torch.Tensor, ids: torch.Tensor) -> float:
        # Rough negative log-likelihood over generated tail (single batch assumed)
        if logits.dim() == 3:
            l = logits[0, -ids.shape[1]:, :]
        else:
            l = logits[-ids.shape[1]:, :]
        probs = F.log_softmax(l, dim=-1)
        take = probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        return float(-take.mean().item())

    def _verifier_score(self, text: str) -> float:
        # Reward JSON validity and valid calc expressions
        json_ok, json_complete = self.json_verifier.check_prefix(text)
        calc_bonus = 0.0
        if "[calc]" in text and "[/calc]" in text:
            new_text, log = self.executor.execute_blocks(text)
            # Count successful calcs
            calc_ok = sum(1 for e in log.get("calc", []) if e.get("ok"))
            calc_bonus = 0.1 * calc_ok
            # Replace text for downstream (but scoring only uses bonus)
            text = new_text
        base = 0.0
        if json_ok:
            base += 0.2
        if json_complete:
            base += 0.2
        return base + calc_bonus

    def run(
        self,
        model,
        tokenizer,
        input_ids: torch.Tensor,
        critics,
        lambdas,
        projectors,
        schedule_cfg,
        max_new_tokens: int = 128,
    ) -> Tuple[str, Dict[str, Any]]:
        # Branch 1: plain
        out_plain = guided_generate(
            model, tokenizer, input_ids.clone(), critics, lambdas, projectors, schedule_cfg, max_new_tokens=max_new_tokens
        )
        text_plain = tokenizer.decode(out_plain[0][input_ids.shape[1]:], skip_special_tokens=True)
        score_plain = self._verifier_score(text_plain)

        # Branch 2: encourage calc if numbers present
        prompt_ids = input_ids.clone()
        prompt_text = tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
        if any(ch.isdigit() for ch in prompt_text):
            # Light nudge: append a calc tag hint
            hint = " Solve step by step. Use [calc]expr[/calc] for arithmetic."
            hint_ids = tokenizer(hint, return_tensors="pt")["input_ids"].to(prompt_ids.device)
            prompt_ids = torch.cat([prompt_ids, hint_ids], dim=1)
        out_calc = guided_generate(
            model, tokenizer, prompt_ids, critics, lambdas, projectors, schedule_cfg, max_new_tokens=max_new_tokens
        )
        text_calc = tokenizer.decode(out_calc[0][prompt_ids.shape[1]:], skip_special_tokens=True)
        score_calc = self._verifier_score(text_calc)

        if score_calc > score_plain:
            return text_calc, {"branch": "calc", "score": score_calc}
        else:
            return text_plain, {"branch": "plain", "score": score_plain}


