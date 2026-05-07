"""
Adaptive Sparse Prompt Optimization with Semantic Edit Regularization

Finite-difference monitoring of Representational Inefficiency:
  delta_log_I = log(1 + rho_t)  [length term, exact]
              + scope_term      [specificity direction from M_delta: +1/-1/0]

Pipeline (per step, after backward):
  1. Collect gradient contexts
  2. M_delta semantic delta analysis (1 LLM call) -> rules_changed with types + specificity_direction
  3. Gradient purification (with RuleBank history) -> RuleBank updated inside purification
  4. Generate regularization guidance (1 LLM call, or early return if stable)
  5. Inject synthetic gradient
  6. Return metrics dict

Max 2 LLM calls: M_delta (1) + regularization guidance (1, skippable).
"""

from __future__ import annotations

import json
import math
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# inject helper
# ---------------------------------------------------------------------------

def inject_pre_step_constraints(system_prompt, instruction_text: str) -> None:
    """Append a regularization directive as a synthetic gradient to system_prompt."""
    import textgrad as tg

    var = tg.Variable(
        value=instruction_text,
        requires_grad=False,
        role_description="Semantic edit regularization directive: follow this instruction when revising the prompt to control representational inefficiency.",
    )
    system_prompt.gradients.add(var)
    if not hasattr(system_prompt, "gradients_context") or system_prompt.gradients_context is None:
        system_prompt.gradients_context = {}
    system_prompt.gradients_context[var] = None


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------

def _extract_first_json_object(s: str) -> str:
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
# M_delta Semantic Delta Analyzer (with structured rule types + specificity)
# ---------------------------------------------------------------------------

SEMANTIC_DELTA_SYSTEM = """You are the "Semantic Delta Analyzer". Compare PREVIOUS_PROMPT and CURRENT_PROMPT at the level of behavioral rules, classify each change, and judge the overall specificity shift.

WHAT YOU MUST NOT DO:
1. DO NOT judge whether the task logic or the new rules are correct.
2. DO NOT act as a prompt quality evaluator.
3. DO NOT produce a character-level or word-level diff. Focus on rule-level structural changes.

WHAT YOU MUST DO:
1. Identify each structural/behavioral change between PREVIOUS_PROMPT and CURRENT_PROMPT (rule-level, not wording).
2. Classify each change. IMPORTANT: Default to CASE_PATCH unless there is clear positive evidence for GENERALIZED_RULE.

Classification criteria:

GENERALIZED_RULE -- A broadly applicable, task-agnostic behavioral principle that is likely useful across many inputs. Mark as GENERALIZED_RULE if:
- It is NOT tied to specific entities, exact numbers, named templates, or surface strings, AND
- It is a reusable reasoning/decision primitive (not a rare exception branch), AND
- It has at least ONE strong support signal:
  (a) a semantically similar high-frequency RuleBank entry (high mention_count), OR
  (b) it appears relevant across multiple execution contexts in GRADIENT_CONTEXTS (at least two distinct contexts).

Otherwise, default to CASE_PATCH.

CASE_PATCH -- The DEFAULT for new/modified rules. Typical signs:
- The change is best explained as being triggered by a single specific example in GRADIENT_CONTEXTS, with no clear evidence it would apply beyond that example.
- There is weak historical support: no semantically similar high-frequency rule in RuleBank (low mention_count), and no evidence the rule applies across multiple contexts.
- The change introduces extra constraints/conditions that mainly resolve a localized failure pattern rather than improving broadly reusable behavior.
(Weak signal only: reliance on very specific triggers often correlates with narrow scope.)

STYLE_ONLY -- Pure wording/formatting change with no behavioral impact.

3. After classifying all changes, judge the OVERALL specificity direction:
- "increase": The update adds or strengthens patch-like rules overall (e.g., adds CASE_PATCH rules, introduces new narrow conditions, or narrows a general rule into a localized rule).
- "decrease": The update removes CASE_PATCH rules or replaces them with GENERALIZED_RULEs, AND does NOT add new CASE_PATCH rules.
- "neutral": No meaningful net change (only STYLE_ONLY changes, or mixed changes that roughly offset).

[INPUTS]
<INITIAL_PROMPT>
{initial_prompt}
</INITIAL_PROMPT>

<PREVIOUS_PROMPT>
{previous_prompt}
</PREVIOUS_PROMPT>

<CURRENT_PROMPT>
{current_prompt}
</CURRENT_PROMPT>

<RULEBANK_SUMMARY>
{rulebank_summary}
</RULEBANK_SUMMARY>

<GRADIENT_CONTEXTS> (Execution contexts from the previous iteration showing what inputs/outputs triggered the changes)
{gradient_contexts}
</GRADIENT_CONTEXTS>

[OUTPUT FORMAT]
Return strictly a JSON object matching this schema. Do not output anything else.
{{
    "rules_changed": [
        {{
            "description": "Brief description of what changed",
            "type": "GENERALIZED_RULE | CASE_PATCH | STYLE_ONLY"
        }}
    ],
    "specificity_direction": "increase | decrease | neutral"
}}"""


def collect_gradient_contexts(system_prompt: Any, max_contexts: int = 3) -> str:
    """Extract execution contexts from system_prompt.gradients and gradients_context."""
    contexts = []
    gradients_context = getattr(system_prompt, "gradients_context", None) or {}
    for i, g in enumerate(list(system_prompt.gradients)[:max_contexts]):
        ctx = gradients_context.get(g)
        if ctx is None:
            continue
        c = ctx.get("context") if isinstance(ctx, dict) else ctx
        if c is None:
            continue
        ctx_str = str(c) if not isinstance(c, str) else c
        contexts.append(f"[Context {i+1}] {ctx_str[:500]}")
    return "\n".join(contexts) if contexts else "No gradient contexts available."


def run_semantic_delta_analyzer(
    previous_prompt: str,
    current_prompt: str,
    initial_prompt: str,
    engine: Any,
    rulebank_summary: str = "",
    gradient_contexts: str = "",
) -> Optional[dict]:
    """
    M_delta: LLM compares (P_{t-1}, P_t, P_0) and returns structured
    rules_changed + specificity_direction.
    """
    if not (previous_prompt or "").strip() or not (current_prompt or "").strip():
        return None
    if (previous_prompt or "").strip() == (current_prompt or "").strip():
        return {"rules_changed": [], "specificity_direction": "neutral"}

    prompt = SEMANTIC_DELTA_SYSTEM.format(
        initial_prompt=(initial_prompt or "")[:4000],
        previous_prompt=(previous_prompt or "")[:4000],
        current_prompt=(current_prompt or "")[:4000],
        rulebank_summary=rulebank_summary or "No historical data available.",
        gradient_contexts=gradient_contexts or "No gradient contexts available.",
    )
    try:
        reply = engine(prompt)
        if hasattr(reply, "value"):
            reply = str(reply.value).strip()
        else:
            reply = str(reply).strip()
    except Exception:
        return None

    for raw in [reply, _extract_first_json_object(reply)]:
        if not raw:
            continue
        try:
            out = json.loads(raw)
            if isinstance(out, dict) and "rules_changed" in out:
                return out
        except json.JSONDecodeError:
            continue
    return None


# ---------------------------------------------------------------------------
# Metric extraction functions
# ---------------------------------------------------------------------------

def compute_rho_t(
    current_prompt: str,
    previous_prompt: str,
    tokenizer_fn: Callable[[str], int],
) -> float:
    """Inflation rate rho_t = (len(P_t) - len(P_{t-1})) / len(P_{t-1})."""
    n_cur = tokenizer_fn(current_prompt or "")
    if not (previous_prompt and previous_prompt.strip()):
        return 0.0
    n_prev = tokenizer_fn(previous_prompt.strip())
    if n_prev <= 0:
        return 0.0
    return (n_cur - n_prev) / n_prev


def compute_omega_edit(semantic_delta_result: Optional[dict]) -> int:
    """Count rules_changed entries with type == CASE_PATCH."""
    if not semantic_delta_result or not semantic_delta_result.get("rules_changed"):
        return 0
    return sum(
        1 for r in semantic_delta_result["rules_changed"]
        if isinstance(r, dict) and r.get("type") == "CASE_PATCH"
    )


def extract_specificity_direction(semantic_delta_result: Optional[dict]) -> int:
    """Extract specificity_direction from M_delta output. Returns +1 / -1 / 0."""
    if not semantic_delta_result:
        return 0
    direction = semantic_delta_result.get("specificity_direction", "neutral")
    if isinstance(direction, str):
        direction = direction.strip().lower()
    if direction == "increase":
        return 1
    elif direction == "decrease":
        return -1
    return 0


def compute_log_delta_I(rho_t: float, scope_term: int) -> tuple:
    """
    delta_log_I = log(1 + max(rho_t, 0)) + scope_term
    Returns (length_term, scope_term).
    """
    length_term = math.log(1 + max(rho_t, 0))
    return length_term, scope_term


# ---------------------------------------------------------------------------
# Regularization guidance generation (LLM call or early return)
# ---------------------------------------------------------------------------

def summarize_rules_changed(semantic_delta_result: Optional[dict]) -> str:
    """Convert M_delta's rules_changed list to readable text."""
    if not semantic_delta_result or not semantic_delta_result.get("rules_changed"):
        return "No rule changes detected."
    lines = []
    for r in semantic_delta_result["rules_changed"]:
        if isinstance(r, dict):
            desc = r.get("description", "unknown change")
            rtype = r.get("type", "UNKNOWN")
            lines.append(f"- [{rtype}] {desc}")
        elif isinstance(r, str):
            lines.append(f"- {r}")
    return "\n".join(lines) if lines else "No rule changes detected."


# ---------------------------------------------------------------------------
# Binary state machine: determine regularization mode (deterministic)
# ---------------------------------------------------------------------------

def determine_regularization_mode(
    rho_t: float,
    specificity_direction: int,
    length_threshold: float = 0.1,
) -> str:
    """
    Binary state machine: two signals each indicate whether degradation occurred,
    combined into 4 modes.

    - length_increased: rho_t > threshold
    - spec_increased: specificity_direction == +1

    Returns:
    - "STRONG_REGULARIZATION": length up + specificity up (overfitting bloat)
    - "COMPRESSION_ONLY":      length up + specificity stable (expression bloat)
    - "GENERALIZE_ONLY":       length stable + specificity up (structural narrowing)
    - "NO_REGULARIZATION":     both signals stable
    """
    length_increased = rho_t > length_threshold
    spec_increased = (specificity_direction == 1)

    if length_increased and spec_increased:
        return "STRONG_REGULARIZATION"
    elif length_increased and not spec_increased:
        return "COMPRESSION_ONLY"
    elif not length_increased and spec_increased:
        return "GENERALIZE_ONLY"
    else:
        return "NO_REGULARIZATION"


# ---------------------------------------------------------------------------
# Regularization guidance generation (LLM executes under Python-determined mode)
# ---------------------------------------------------------------------------

REGULARIZATION_GUIDANCE_PROMPT = """You are a structural regularization controller.

You are given a REGULARIZATION_MODE, the current prompt, and a list of recent rule changes. Generate precise structural regularization guidance strictly according to the specified mode.

<REGULARIZATION_MODE>
{mode}
</REGULARIZATION_MODE>

Mode definitions (follow the one that matches):

STRONG_REGULARIZATION:
  Prompt length increased AND new narrow case-specific rules were added.
  This is the most critical situation -- the prompt is both bloating and
  becoming more specialized.
  You MUST give firm directives to BOTH compress AND generalize:
  (1) Merge redundant sentences that express the same behavioral rule into a single shorter statement.
  (2) Tighten verbose phrasing -- convey the same meaning in fewer tokens.
  (3) Identify narrow rules that target specific rare scenarios and rewrite them as broader principles that cover a wider range of inputs.
  (4) Remove case-specific patches that cannot be meaningfully generalized. However, do NOT remove rules that are broadly useful -- only compress their expression while preserving their behavioral intent.
  Tone: firm imperative.

COMPRESSION_ONLY:
  Prompt length increased but rule specificity did not worsen. The prompt is bloating and must be shortened.
  You MUST focus ONLY on reducing prompt length:
  (1) Merge redundant sentences that express the same behavioral rule into a single shorter statement.
  (2) Tighten verbose phrasing -- convey the same meaning in fewer tokens.
  Do NOT remove rules that are broadly useful -- compress their expression while preserving their behavioral intent.
  Tone: moderate directive.

GENERALIZE_ONLY:
  Prompt length is stable but new narrow case-specific rules were added. The prompt is becoming more specialized without growing longer.
  You MUST focus ONLY on generalization:
  (1) Identify narrow rules that target specific rare scenarios.
  (2) Rewrite them as broader principles that cover a wider range of inputs.
  Length compression is not the goal -- focus on broadening scope.
  Tone: moderate directive.

<CURRENT_PROMPT>
{current_prompt}
</CURRENT_PROMPT>

<NEWLY_CHANGED_RULES>(Rule changes detected between the previous and current prompt, each tagged as GENERALIZED_RULE, CASE_PATCH, or STYLE_ONLY)
{rules_changed_summary}
</NEWLY_CHANGED_RULES>

Rules for generating guidance:
- Do NOT alter task semantics or remove rules that are clearly beneficial for task correctness.
- You MUST reference specific sentences or rules from <CURRENT_PROMPT> and <NEWLY_CHANGED_RULES> when suggesting merges, removals, or generalizations. Do NOT give generic advice -- say exactly WHICH rules to change and HOW.
- Be concise. Do not repeat or rephrase the same suggestion. Do not include explanations or justifications -- state what to change and how, nothing more.

Output STRICTLY valid JSON:
{{
    "guidance": "Your concise regularization guidance referencing specific rules."
}}"""


def generate_regularization_guidance(
    rho_t: float,
    specificity_direction: int,
    rules_changed_summary: str,
    current_prompt: str,
    engine: Any,
    length_threshold: float = 0.2,
) -> Optional[str]:
    """
    Determine regularization mode (Python binary state machine), then call LLM
    to generate directives under that mode.
    NO_REGULARIZATION mode does not call LLM and returns None.
    """
    mode = determine_regularization_mode(rho_t, specificity_direction, length_threshold)

    if mode == "NO_REGULARIZATION":
        return None

    prompt = REGULARIZATION_GUIDANCE_PROMPT.format(
        mode=mode,
        current_prompt=(current_prompt or "")[:3000],
        rules_changed_summary=(rules_changed_summary or "No rule changes detected.")[:1500],
    )
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
            if isinstance(out, dict) and "guidance" in out:
                # Normalize: LLM may return guidance as list/dict/int instead of str
                g = out["guidance"]
                if isinstance(g, list):
                    g = "\n".join(str(item) for item in g if item is not None)
                elif not isinstance(g, str):
                    g = str(g)
                guidance = g.strip()
                return guidance if guidance else None
        except json.JSONDecodeError:
            continue

    # JSON parse failed, try using reply text directly
    if reply and len(reply) < 500:
        return reply
    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def apply_adaptive_sparse_pipeline(
    system_prompt: Any,
    current_prompt: str,
    previous_prompt: str,
    initial_prompt: str,
    tokenizer_fn: Callable[[str], int],
    llm_api: Any,
    *,
    rulebank: Any = None,
    length_threshold: float = 0.2,
    verbose: bool = False,
) -> dict:
    """
    Semantic edit regularization pipeline (after backward, before optimizer.step).

    Args:
        llm_api:    Engine for gradient purification, RuleBank distillation,
                    M_delta semantic delta, and regularization guidance.
        rulebank:   RuleBank instance for historical rule frequency tracking.

    Returns a dict of intermediate metrics for logging.
    """
    from textreg.autograd.gradient_purification_gate import apply_gradient_purification_gate

    # 1. Collect gradient contexts (before purification modifies them)
    gradient_contexts_str = collect_gradient_contexts(system_prompt)

    # 2. Prepare RuleBank summary
    rulebank_summary = ""
    if rulebank is not None and hasattr(rulebank, "get_summary"):
        rulebank_summary = rulebank.get_summary()
    if not rulebank_summary:
        rulebank_summary = "(empty)"

    # 3. M_delta semantic delta analysis (LLM call #1)
    semantic_delta_result = run_semantic_delta_analyzer(
        previous_prompt, current_prompt, initial_prompt, llm_api,
        rulebank_summary=rulebank_summary,
        gradient_contexts=gradient_contexts_str,
    )
    if verbose:
        if semantic_delta_result:
            print("[AdaptiveSparse] M_delta rules_changed:", semantic_delta_result.get("rules_changed", []))
            print("[AdaptiveSparse] M_delta specificity_direction:", semantic_delta_result.get("specificity_direction", "N/A"))
        else:
            print("[AdaptiveSparse] M_delta returned None (parse error or identical prompts)")

    # 4. Extract metrics from M_delta
    specificity_direction = extract_specificity_direction(semantic_delta_result)
    omega_edit = compute_omega_edit(semantic_delta_result)

    # 5. Compute rho_t and delta_log_I
    rho_t = compute_rho_t(current_prompt, previous_prompt, tokenizer_fn)
    length_term, scope_term = compute_log_delta_I(rho_t, specificity_direction)
    if verbose:
        print(f"[AdaptiveSparse] rho_t={rho_t:.4f}, omega_edit={omega_edit}, "
              f"length_term={length_term:.4f}, scope_term={scope_term}, "
              f"delta_log_I={length_term + scope_term:.4f}")

    # 6. Gradient purification + RuleBank update
    purification_result = apply_gradient_purification_gate(
        system_prompt, llm_api, rulebank=rulebank, verbose=verbose,
    )
    if rulebank is not None and purification_result["accepted_texts"]:
        from textreg.autograd.rulebank import update_rulebank_from_gradients
        if verbose:
            print(f"[RuleBank] Updating from {len(purification_result['accepted_texts'])} purified gradients...")
        update_rulebank_from_gradients(rulebank, system_prompt.gradients, llm_api)
        if verbose:
            print(f"[RuleBank] Now has {len(rulebank.rules)} rules")

    # 7. Determine regularization mode and generate guidance (LLM call #2 if needed)
    reg_mode = determine_regularization_mode(rho_t, specificity_direction, length_threshold=length_threshold)
    rules_changed_summary = summarize_rules_changed(semantic_delta_result)
    guidance = generate_regularization_guidance(
        rho_t, specificity_direction, rules_changed_summary,
        current_prompt, llm_api,
        length_threshold=length_threshold,
    )
    if verbose:
        print(f"[AdaptiveSparse] Regularization mode: {reg_mode}")
        print(f"[AdaptiveSparse] Regularization guidance: {guidance}")

    # 8. Inject synthetic gradient (if guidance available)
    if guidance:
        inject_pre_step_constraints(system_prompt, guidance)

    # 9. Return metrics
    return {
        "rho_t": rho_t,
        "omega_edit": omega_edit,
        "length_term": length_term,
        "scope_term": scope_term,
        "delta_log_I": length_term + scope_term,
        "specificity_direction": specificity_direction,
        "reg_mode": reg_mode,
        "token_count": tokenizer_fn(current_prompt),
        "purification_result": purification_result,
        "rulebank_size": len(rulebank.rules) if rulebank else 0,
        "guidance": guidance,
    }
