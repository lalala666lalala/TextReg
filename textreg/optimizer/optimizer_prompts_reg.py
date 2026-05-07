"""
TGD prompts for optimizer_reg: feedback is split into
- <REG_FEEDBACK>     : semantic edit regularization guidance
- <CONTEXT>          : task feedback only
so the optimizer can distinguish and apply them separately.
"""
from .optimizer_prompts import (
    GLOSSARY_TEXT,
    OPTIMIZER_SYSTEM_PROMPT,
    TGD_PROMPT_SUFFIX,
    TGD_MULTIPART_PROMPT_INIT,
    TGD_MULTIPART_PROMPT_PREFIX,
    MOMENTUM_PROMPT_ADDITION,
    CONSTRAINT_PROMPT_ADDITION,
    IN_CONTEXT_EXAMPLE_PROMPT_ADDITION,
)

# System prompt for reg: add REG_FEEDBACK to glossary so the optimizer knows the tag
REG_GLOSSARY_ADDITION = (
    "\n# - <REG_FEEDBACK>: Semantic edit regularization feedback; follow its structural directives to control prompt length and specificity."
)
OPTIMIZER_SYSTEM_PROMPT_REG = OPTIMIZER_SYSTEM_PROMPT.replace(GLOSSARY_TEXT.rstrip(), GLOSSARY_TEXT.rstrip() + REG_GLOSSARY_ADDITION)

# TGD update instruction for reg: reg_feedback + task context
# The trailing reference to <REG_FEEDBACK> is only appended when reg_feedback is non-empty,
# to avoid dangling references when no regularization guidance was injected.
TGD_PROMPT_PREFIX_REG_BASE = (
    "Here is the role of the variable you will improve: <ROLE>{variable_desc}</ROLE>.\n\n"
    "The variable is the text within the following span: <VARIABLE> {variable_short} </VARIABLE>\n\n"
    "{reg_section}"
    "Here is the context and task feedback we got for the variable:\n\n"
    "<CONTEXT>{variable_grad}</CONTEXT>\n\n"
)

_TGD_TRAILING_WITH_REG = (
    "Improve the variable ({variable_desc}) using the feedback provided in <FEEDBACK> tags, "
    "following the structural directives in <REG_FEEDBACK> to control prompt length and specificity.\n"
)
_TGD_TRAILING_NO_REG = (
    "Improve the variable ({variable_desc}) using the feedback provided in <FEEDBACK> tags.\n"
)

# Soft-projection variant: structural decomposition of how task feedback and
# regularization feedback should interact, plus an item-level task-dominance
# fallback (per paper Eq. 9 / Eq. 10).
_TGD_TRAILING_WITH_REG_PROJ = (
    "Improve the variable ({variable_desc}) by integrating both the task "
    "feedback in <FEEDBACK> and the regularization feedback in "
    "<REG_FEEDBACK>. Both contain concrete instructions; read them together "
    "and apply them as follows:\n"
    "- When the task feedback and the regularization feedback point to the "
    "same part of the prompt, apply them as one combined edit.\n"
    "- When they target different parts, apply both.\n"
    "- When they conflict (e.g., the task feedback asks you to add a new "
    "rule while the regularization feedback asks you to merge or shorten), "
    "address the task feedback in the form most consistent with what the "
    "regularization feedback specifies — for instance, by folding the task "
    "fix into the change that the regularization feedback proposes, so the "
    "two are realized through a single coordinated edit instead of two "
    "unrelated ones.\n"
    "If a specific task feedback item genuinely cannot be addressed without "
    "violating a regularization feedback item, prioritize that task "
    "feedback item and apply it completely, even if this means setting "
    "aside that specific regularization item. All other regularization "
    "items in <REG_FEEDBACK> still apply normally to the rest of the "
    "prompt.\n"
)

# Section inserted when reg_feedback is non-empty
REG_FEEDBACK_SECTION = (
    "The following is semantic edit regularization feedback. "
    "Follow its structural directives to control prompt length and specificity:\n\n"
    "<REG_FEEDBACK>{reg_feedback}</REG_FEEDBACK>\n\n"
)

REG_FEEDBACK_EMPTY_SECTION = ""


def construct_tgd_prompt_reg(do_momentum: bool = False,
                             do_constrained: bool = False,
                             do_in_context_examples: bool = False,
                             do_gradient_memory: bool = False,
                             do_soft_projection: bool = False,
                             reg_feedback: str = "",
                             **optimizer_kwargs):
    """
    Same as construct_tgd_prompt, but splits feedback into:
    - reg_feedback  → <REG_FEEDBACK>
    - variable_grad → <CONTEXT> (task feedback only)

    Args:
        do_soft_projection: when True and reg_feedback is non-empty, use the
            soft-projection trailing that asks the optimizer to pick edits
            minimizing conflict with <REG_FEEDBACK>. When False, use the
            simpler trailing (baseline behavior).
    """
    has_reg = bool(reg_feedback and reg_feedback.strip())
    reg_section = REG_FEEDBACK_SECTION.format(reg_feedback=reg_feedback) if has_reg else REG_FEEDBACK_EMPTY_SECTION
    optimizer_kwargs["reg_section"] = reg_section

    if has_reg:
        trailing = _TGD_TRAILING_WITH_REG_PROJ if do_soft_projection else _TGD_TRAILING_WITH_REG
    else:
        trailing = _TGD_TRAILING_NO_REG
    prefix_template = TGD_PROMPT_PREFIX_REG_BASE + trailing

    variable_grad = optimizer_kwargs.get("variable_grad")

    if isinstance(variable_grad, str):
        multipart = False
        prompt = prefix_template.format(**optimizer_kwargs)
    else:
        gradient_context = list(variable_grad)
        init_parts = [TGD_MULTIPART_PROMPT_INIT.format(**optimizer_kwargs)]
        if reg_section:
            init_parts.append(reg_section)
        gradient_context = init_parts + gradient_context
        multipart = True
        prompt = TGD_MULTIPART_PROMPT_PREFIX.format(**optimizer_kwargs)

    if do_momentum:
        prompt += MOMENTUM_PROMPT_ADDITION.format(**optimizer_kwargs)

    if do_constrained:
        prompt += CONSTRAINT_PROMPT_ADDITION.format(**optimizer_kwargs)

    if do_in_context_examples:
        prompt += IN_CONTEXT_EXAMPLE_PROMPT_ADDITION.format(**optimizer_kwargs)

    if do_gradient_memory:
        prompt += optimizer_kwargs.get("gradient_memory", "")

    prompt += TGD_PROMPT_SUFFIX.format(**optimizer_kwargs)

    if not multipart:
        return prompt
    return gradient_context + [prompt]
