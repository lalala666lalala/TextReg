"""
Gradient Purification Gate

After backward() and before optimizer.step(), this module classifies each
textual gradient in system_prompt.gradients into one of three categories:
- generalizable logic fix -> emit purified text
- narrow edge-case patch  -> emit empty string (reject)
- pure style              -> emit empty string (reject)

The RuleBank's mention_count history is used as a prior: frequently-seen
rules are more likely to be accepted. Parse / API failures fall back to
keeping the raw gradient; if purification would empty all gradients, the
filter is bypassed to avoid breaking the optimizer.
"""

import json
from typing import Any, List, Optional, Set


GRADIENT_GATEKEEPER_SYSTEM = """You are the "Gradient Purifier". Your job is to decide whether a proposed feedback (gradient) contains genuinely generalizable improvements, and if so, synthesize them into a concise principle. You output either a purified summary or an empty string.

### INPUT DATA:

<CURRENT_SYSTEM_PROMPT>
{current_prompt}
</CURRENT_SYSTEM_PROMPT>

<EXECUTION_CONTEXT> (The specific user input and model output that triggered this gradient)
{gradient_context}
</EXECUTION_CONTEXT>

<PROPOSED_GRADIENT>
{gradient_text}
</PROPOSED_GRADIENT>

<RULEBANK_SUMMARY> (Historical record of previously accepted generalizable rules and their frequency)
{rulebank_summary}
</RULEBANK_SUMMARY>

### YOUR TASK:

Internally classify the gradient into one of three categories, then act accordingly:

**Category 1: Generalizable Logic Fix -> OUTPUT PURIFIED TEXT**
The feedback identifies a reasoning flaw, missing constraint, or logical gap that applies broadly across many inputs, not just the specific case in <EXECUTION_CONTEXT>.
- HISTORICAL SIGNAL: If the RuleBank contains a similar rule with high mention_count, this fix has been triggered by many different samples across previous steps -> strong evidence it is generalizable. Lean toward accepting.
- Synthesize into a concise, general principle. Strip specific entities, numbers, and scenario details from the execution context. Merge overlapping points. Each sentence in your output must describe a distinct, actionable behavioral rule -- remove any sentence that merely restates or elaborates on another.
- Substantive rules about output format, reasoning scope, verification steps, and counting methods are generalizable -- do NOT reject them.

**Category 2: Narrow Edge-Case Patch -> OUTPUT EMPTY STRING ""**
The feedback proposes a fix that is specific to the rare or unusual scenario shown in <EXECUTION_CONTEXT>. Adopting it would add a narrow rule that helps very few future inputs.
- HISTORICAL SIGNAL: If no similar rule exists in the RuleBank (or mention_count is very low), and the fix is clearly tailored to the specific case in <EXECUTION_CONTEXT>, it is likely a narrow patch. Lean toward rejecting.
- Examples: "When items are listed in reverse order, count backwards", "If the answer involves a fraction, round down" (when only this one case had fractions).

**Category 3: Pure Style / Formatting -> OUTPUT EMPTY STRING ""**
The feedback only concerns tone, formatting, verbosity, or presentation with zero impact on task correctness.
- Examples: "Use bullet points", "Sound more confident", "Use passive voice".

### OUTPUT FORMAT:
Respond with valid JSON only:
{{
    "purified_gradient": "Your synthesized concise principle (if Category 1), or empty string (if Category 2 or 3)."
}}"""


def _get_context_str(gradients_context: dict, g: Any) -> str:
    """Extract the execution context string from gradients_context[g] for the Gatekeeper."""
    ctx = gradients_context.get(g) if gradients_context is not None else None
    if ctx is None:
        return ""
    c = ctx.get("context")
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(str(part) for part in c)
    return ""


def _call_gatekeeper(
    gradient_text: str,
    gradient_context: str,
    current_prompt: str,
    engine: Any,
    rulebank_summary: str = "",
    verbose: bool = False,
) -> Optional[dict]:
    """Call the purification LLM and return parsed JSON with purified_gradient; None on failure."""
    prompt = GRADIENT_GATEKEEPER_SYSTEM.format(
        current_prompt=current_prompt or "",
        gradient_context=gradient_context or "",
        gradient_text=gradient_text or "",
        rulebank_summary=rulebank_summary or "No historical data available.",
    )
    if verbose:
        print("=" * 60)
        print("[Gatekeeper] FULL PROMPT SENT TO LLM:")
        print("=" * 60)
        print(prompt)
        print("=" * 60)
    try:
        reply = engine(prompt)
        if hasattr(reply, "value"):
            reply = str(reply.value).strip()
        else:
            reply = str(reply).strip()
    except Exception:
        return None

    for candidate in [reply, _extract_first_json_object(reply)]:
        if not candidate:
            continue
        try:
            out = json.loads(candidate)
            if isinstance(out, dict) and "purified_gradient" in out:
                return out
        except json.JSONDecodeError:
            continue
    return None


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


def apply_gradient_purification_gate(
    system_prompt: Any,
    engine: Any,
    *,
    rulebank: Any = None,
    verbose: bool = False,
) -> dict:
    """
    Classify each gradient in system_prompt.gradients via LLM:
    - Non-empty purified text -> overwrite g.value, keep (accept)
    - Empty string -> discard (reject, including edge case and stylistic)
    - Parse failure -> keep original gradient

    Returns:
    {
        "accepted_texts": list[str],   # purified gradient texts that were accepted
        "n_before": int,               # gradient count before purification
        "n_after": int,                # gradient count after purification
        "n_rejected": int,             # number rejected
        "n_parse_error": int,          # number of parse failures
    }
    """
    current_prompt = getattr(system_prompt, "value", "") or ""
    gradients_context = getattr(system_prompt, "gradients_context", None) or {}

    # Prepare RuleBank summary
    rulebank_summary = ""
    if rulebank is not None and hasattr(rulebank, "get_summary"):
        rulebank_summary = rulebank.get_summary()
    if not rulebank_summary:
        rulebank_summary = "No historical data available."

    n_before = len(system_prompt.gradients)
    accepted_texts: List[str] = []
    n_parse_error = 0
    to_remove: Set[Any] = set()

    for g in list(system_prompt.gradients):
        gradient_text = getattr(g, "value", "") or ""
        if not gradient_text.strip():
            to_remove.add(g)
            continue

        gradient_context = _get_context_str(gradients_context, g)
        result = _call_gatekeeper(
            gradient_text, gradient_context, current_prompt, engine,
            rulebank_summary=rulebank_summary,
            verbose=verbose,
        )

        if result is None:
            n_parse_error += 1
            if verbose:
                print("[PurificationGate] Parse/API failed -> keep original gradient")
            continue

        purified = (result.get("purified_gradient") or "").strip()
        if purified:
            g.value = purified
            accepted_texts.append(purified)
            if verbose:
                print(f"[PurificationGate] purified ({len(purified)} chars): {purified[:100]}...")
        else:
            if verbose:
                print("[PurificationGate] Rejected (edge-case patch or pure style) -> discard")
            to_remove.add(g)

    # If purification would remove ALL gradients, keep them to avoid breaking the optimizer
    n_keep = len(system_prompt.gradients) - len(to_remove)
    if n_keep == 0 and to_remove:
        if verbose:
            print("[PurificationGate] Would remove all gradients -> keeping all to avoid empty gradients.")
        to_remove = set()
    for g in to_remove:
        system_prompt.gradients.discard(g)

    n_after = len(system_prompt.gradients)
    n_rejected = n_before - n_after

    return {
        "accepted_texts": accepted_texts,
        "n_before": n_before,
        "n_after": n_after,
        "n_rejected": n_rejected,
        "n_parse_error": n_parse_error,
    }
