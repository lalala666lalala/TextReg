"""
RuleBank: minimal rule frequency tracker.

Maintains a {canonical_description -> mention_count} mapping that records
which behavioral rules have appeared during optimization and how often.
Downstream gradient purification and semantic edit regularization read
RuleBank as a historical prior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    rule_id: str
    canonical_description: str
    mention_count: int = 0


class RuleBank:
    def __init__(self) -> None:
        self.rules: dict[str, Rule] = {}
        self.next_id: int = 1

    # -- write --

    def insert(self, canonical_description: str, count: int = 1) -> str:
        """Insert a new rule and return its rule_id."""
        rule_id = f"R{self.next_id}"
        self.next_id += 1
        self.rules[rule_id] = Rule(
            rule_id=rule_id,
            canonical_description=canonical_description,
            mention_count=count,
        )
        return rule_id

    def increment(self, rule_id: str, value: int = 1) -> None:
        """Increment mention_count for an existing rule; silently skip if not found."""
        if rule_id in self.rules:
            self.rules[rule_id].mention_count += value

    # -- read --

    def get_summary(self, max_rules: int = 30) -> str:
        """Return a text summary sorted by mention_count descending (for LLM prompts)."""
        if not self.rules:
            return "(empty)"
        sorted_rules = sorted(
            self.rules.values(),
            key=lambda r: r.mention_count,
            reverse=True,
        )[:max_rules]
        lines = [
            f"{r.rule_id} (count={r.mention_count}): {r.canonical_description}"
            for r in sorted_rules
        ]
        return "\n".join(lines)

    # -- batch --

    def apply_operations(self, operations: list[dict]) -> None:
        """Batch execute insert / increment operations.

        Expected format::

            [
                {"type": "increment", "rule_id": "R3", "value": 1},
                {"type": "insert", "canonical_description": "...", "value": 1}
            ]
        """
        for op in operations:
            op_type = op.get("type", "")
            if op_type == "increment":
                rule_id = op.get("rule_id", "")
                value = int(op.get("value", 1))
                self.increment(rule_id, value)
            elif op_type == "insert":
                desc = op.get("canonical_description", "")
                value = int(op.get("value", 1))
                if desc:
                    self.insert(desc, value)


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def _extract_first_json_object(s: str) -> str:
    """Extract the first complete JSON object from a string (bracket-matching)."""
    start = s.find("{")
    if start == -1:
        return ""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return ""


# ---------------------------------------------------------------------------
# LLM-driven RuleBank update
# ---------------------------------------------------------------------------

RULE_EXTRACTION_PROMPT = """You are a rule canonicalization and matching engine.

Given a raw textual gradient (feedback on how to improve a system prompt) and the current RuleBank, perform two tasks:

1. Extract mid-level canonical behavioral rules from the raw gradient.
   - Remove references to specific entities, exact numbers, or particular examples.
   - Preserve structural reasoning patterns.
   - Keep rules at mid-level abstraction (not too specific, not too vague).

2. For each extracted rule, compare it with the existing RuleBank:
   - If semantically equivalent to an existing rule (same structural pattern, not just similar wording), output an INCREMENT operation with that rule's ID.
   - If no match exists, output an INSERT operation with the canonical description.

[CURRENT RULEBANK]
{rulebank_summary}

[RAW GRADIENT]
{raw_gradient}

Output STRICTLY valid JSON matching this schema, nothing else:
{{
    "operations": [
        {{"type": "increment", "rule_id": "R3", "value": 1}},
        {{"type": "insert", "canonical_description": "Always verify intermediate computation steps before producing a final answer", "value": 1}}
    ]
}}"""


def update_rulebank_from_gradient(
    rulebank: RuleBank,
    raw_gradient_text: str,
    engine: Any,
) -> None:
    """Call LLM for rule extraction + matching on a single raw gradient,
    then update RuleBank with deterministic Python logic.
    Silently skips on LLM call or parse failure.
    """
    if not (raw_gradient_text or "").strip():
        return

    prompt = RULE_EXTRACTION_PROMPT.format(
        rulebank_summary=rulebank.get_summary(),
        raw_gradient=raw_gradient_text,
    )
    try:
        reply = engine(prompt)
        if hasattr(reply, "value"):
            reply = str(reply.value).strip()
        else:
            reply = str(reply).strip()
    except Exception:
        return

    for candidate in [reply, _extract_first_json_object(reply)]:
        if not candidate:
            continue
        try:
            out = json.loads(candidate)
            if isinstance(out, dict) and "operations" in out:
                rulebank.apply_operations(out["operations"])
                return
        except json.JSONDecodeError:
            continue


def update_rulebank_from_gradients(
    rulebank: RuleBank,
    gradients: set,
    engine: Any,
) -> None:
    """Call update_rulebank_from_gradient for each gradient in system_prompt.gradients."""
    for g in list(gradients):
        raw_text = getattr(g, "value", "") or ""
        update_rulebank_from_gradient(rulebank, raw_text, engine)
