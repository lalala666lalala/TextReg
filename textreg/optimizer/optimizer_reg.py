"""
optimizer_reg: same as optimizer but splits feedback into <REG_FEEDBACK> (L1/L2 semantic-edit)
and task feedback in the prompt, so the optimizer can distinguish and apply them separately.
"""
from typing import List, Union, Tuple
from collections import defaultdict
from textgrad.variable import Variable
from textgrad import logger
from textgrad.engine import EngineLM
from textgrad.config import validate_engine_or_get_default

from .optimizer import Optimizer
from .optimizer_prompts import GRADIENT_TEMPLATE, GRADIENT_MULTIPART_TEMPLATE
from .optimizer_prompts_reg import construct_tgd_prompt_reg, OPTIMIZER_SYSTEM_PROMPT_REG


def _is_reg_gradient(g) -> bool:
    """True if this gradient is the semantic-edit regularization synthetic feedback."""
    role = (getattr(g, "role_description", "") or "").lower()
    return "l1/l2" in role or "behavioral constraint" in role or "regularization directive" in role


def get_reg_and_task_gradients(variable) -> Tuple[str, Union[str, List]]:
    """
    Splits variable.gradients into two buckets:
    - reg_feedback: semantic-edit regularization guidance (<REG_FEEDBACK>).
    - task_grad: remaining (task) gradients, same format as get_gradient_and_context_text.
    """
    reg_parts = []
    task_grads = []
    for g in variable.gradients:
        if _is_reg_gradient(g):
            reg_parts.append((g.value or "").strip())
        else:
            task_grads.append(g)

    reg_feedback = "\n".join(p for p in reg_parts if p).strip()

    gradient_content = []
    context_dict = getattr(variable, "gradients_context", None) or {}
    for g in task_grads:
        if context_dict.get(g) is None:
            gradient_content.append(g.value)
        else:
            context = context_dict[g]
            if isinstance(context.get("context"), str):
                criticism_and_context = GRADIENT_TEMPLATE.format(feedback=g.value, **context)
                gradient_content.append(criticism_and_context)
            elif isinstance(context.get("context"), list):
                context_prompt = GRADIENT_MULTIPART_TEMPLATE.format(**context, feedback=g.value)
                criticism_and_context = context["context"] + [context_prompt]
                gradient_content.extend(criticism_and_context)
            else:
                raise ValueError("Context must be either a string or a list.")

    if all(isinstance(i, str) for i in gradient_content):
        task_grad = "\n".join(gradient_content) if gradient_content else ""
    else:
        task_grad = gradient_content
    return reg_feedback, task_grad


class TextualGradientDescentReg(Optimizer):
    """
    TextualGradientDescent with separate <REG_FEEDBACK> (L1/L2) and task feedback in the prompt.
    Otherwise identical to TextualGradientDescent.
    """

    def __init__(self,
                 parameters: List[Variable],
                 verbose: int = 0,
                 engine: Union[EngineLM, str] = None,
                 constraints: List[str] = None,
                 new_variable_tags: List[str] = None,
                 optimizer_system_prompt: str = OPTIMIZER_SYSTEM_PROMPT_REG,
                 in_context_examples: List[str] = None,
                 gradient_memory: int = 0,
                 do_soft_projection: bool = False):
        super().__init__(parameters)

        if new_variable_tags is None:
            new_variable_tags = ["<IMPROVED_VARIABLE>", "</IMPROVED_VARIABLE>"]

        self.engine = validate_engine_or_get_default(engine)
        self.verbose = verbose
        self.constraints = constraints if constraints is not None else []
        self.optimizer_system_prompt = optimizer_system_prompt.format(
            new_variable_start_tag=new_variable_tags[0],
            new_variable_end_tag=new_variable_tags[1],
        )
        self.do_constrained = len(self.constraints) > 0
        self.new_variable_tags = new_variable_tags
        self.in_context_examples = in_context_examples if in_context_examples is not None else []
        self.do_in_context_examples = len(self.in_context_examples) > 0
        self.gradient_memory = gradient_memory
        self.gradient_memory_dict = defaultdict(list)
        self.do_gradient_memory = gradient_memory > 0
        self.do_soft_projection = do_soft_projection

    @property
    def constraint_text(self):
        constraints_ordered = [f"Constraint {i + 1}: {c}" for i, c in enumerate(self.constraints)]
        return "\n".join(constraints_ordered)

    def get_gradient_memory_text(self, variable: Variable):
        grad_memory = ""
        variable_grad_memory = self.gradient_memory_dict[variable][-self.gradient_memory:]
        for i, grad_info in enumerate(variable_grad_memory):
            grad_memory += f"\n<FEEDBACK-{i + 1}> {grad_info['value']}</FEEDBACK-{i + 1}>\n"
        return grad_memory

    def update_gradient_memory(self, variable: Variable):
        self.gradient_memory_dict[variable].append({"value": variable.get_gradient_text()})

    def _update_prompt(self, variable: Variable) -> Union[str, List]:
        reg_feedback, task_grad = get_reg_and_task_gradients(variable)
        grad_memory = self.get_gradient_memory_text(variable)
        optimizer_information = {
            "variable_desc": variable.get_role_description(),
            "variable_value": variable.value,
            "variable_grad": task_grad,
            "variable_short": variable.get_short_value(),
            "constraint_text": self.constraint_text,
            "new_variable_start_tag": self.new_variable_tags[0],
            "new_variable_end_tag": self.new_variable_tags[1],
            "in_context_examples": "\n".join(self.in_context_examples),
            "gradient_memory": grad_memory,
        }

        prompt = construct_tgd_prompt_reg(
            do_constrained=self.do_constrained,
            do_in_context_examples=(self.do_in_context_examples and len(self.in_context_examples) > 0),
            do_gradient_memory=(self.do_gradient_memory and (grad_memory != "")),
            do_soft_projection=self.do_soft_projection,
            reg_feedback=reg_feedback,
            **optimizer_information,
        )

        logger.info("TextualGradientDescentReg prompt for update", extra={"prompt": prompt})
        return prompt

    def step(self):
        for parameter in self.parameters:
            prompt_update_parameter = self._update_prompt(parameter)
            new_text = self.engine(prompt_update_parameter, system_prompt=self.optimizer_system_prompt)
            logger.info("TextualGradientDescentReg optimizer response", extra={"optimizer.response": new_text})
            print("--- PROMPT TO OPTIMIZER ---")
            print(prompt_update_parameter)
            print("--- RESPONSE FROM OPTIMIZER ---")
            print(new_text)

            try:
                new_value = new_text.split(self.new_variable_tags[0])[1].split(self.new_variable_tags[1])[0].strip()
            except IndexError:
                logger.error(
                    "TextualGradientDescentReg optimizer response could not be indexed",
                    extra={"optimizer.response": new_text},
                )
                raise IndexError(
                    f"TextualGradientDescentReg optimizer response could not be indexed. Response: {new_text}"
                )
            parameter.set_value(new_value)
            logger.info("TextualGradientDescentReg updated text", extra={"parameter.value": parameter.value})
            if self.verbose:
                print("-----------------------TextualGradientDescentReg------------------------")
                print(parameter.value)

            if self.do_gradient_memory:
                self.update_gradient_memory(parameter)
