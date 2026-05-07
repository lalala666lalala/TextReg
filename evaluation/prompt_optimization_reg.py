"""
Prompt optimization with TextReg's adaptive sparse regularization pipeline.

Usage:
  python evaluation/prompt_optimization_reg.py \
      --task BBH_logical_deduction_three_objects \
      --backbone_engine gpt-4o \
      --model ollama-Qwen/Qwen2.5-7B-Instruct \
      --batch_size 3 --max_epochs 1 --max_steps 12 \
      --seed 42 --num_threads 10 --run_validation \
      --result_dir results/textreg
"""

import os
import argparse
import concurrent
from tqdm import tqdm
import textreg as rv
import textgrad as tg
from textgrad.tasks import load_task
import random
import json

from textreg.autograd.regularization import apply_adaptive_sparse_pipeline
from textreg.autograd.rulebank import RuleBank


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def config():
    parser = argparse.ArgumentParser(description="Optimize a system prompt with TextReg's adaptive sparse pipeline.")
    parser.add_argument("--task", type=str, default="BBH_object_counting", help="The task to evaluate the model on.")
    parser.add_argument("--backbone_engine", type=str, default="gpt-4o", help="The backbone engine for backward, purification, M_delta, reg guidance, and optimizer.")
    parser.add_argument("--model", type=str, default="ollama-meta-llama/Llama-3.1-8B-Instruct", help="The solver model on which the prompt is optimized.")
    parser.add_argument("--batch_size", type=int, default=3, help="The batch size to use for training.")
    parser.add_argument("--max_epochs", type=int, default=1, help="The maximum number of epochs to train for.")
    parser.add_argument("--max_steps", type=int, default=12, help="Maximum optimization steps per epoch (default 12).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_validation", action="store_true", help="Whether to run validation or not.")
    parser.add_argument("--num_threads", type=int, default=3, help="The number of threads to use for evaluation.")
    parser.add_argument("--reg_length_threshold", type=float, default=0.2,
                        help="Length threshold for regularization mode state machine: rho_t > this triggers length_increased (default 0.2).")
    parser.add_argument("--soft_projection", action="store_true", help="Use soft-projection trailing in optimizer_reg (recommended).")
    parser.add_argument("--result_dir", type=str, default="", help="Directory to save result JSON.")
    return parser.parse_args()


def eval_sample(item, eval_fn, model):
    x, y = item
    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y = tg.Variable(str(y), requires_grad=False, role_description="correct answer for the query")
    response = model(x)
    try:
        eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
        return int(eval_output_variable.value)
    except:
        eval_output_variable = eval_fn([x, y, response])
        eval_output_parsed = eval_fn.parse_output(eval_output_variable)
        return int(eval_output_parsed)


def eval_dataset(test_set, eval_fn, model, max_samples: int = None):
    if max_samples is None:
        max_samples = len(test_set)
    accuracy_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for _, sample in enumerate(test_set):
            future = executor.submit(eval_sample, sample, eval_fn, model)
            futures.append(future)
            if len(futures) >= max_samples:
                break
        tqdm_loader = tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)
        for future in tqdm_loader:
            acc_item = future.result()
            accuracy_list.append(acc_item)
            tqdm_loader.set_description(f"Accuracy: {np.mean(accuracy_list)}")
    return accuracy_list


def run_validation_revert(system_prompt: tg.Variable, results, model, eval_fn, val_set):
    """Revert prompt when val accuracy drops."""
    print("Running the current prompt on the validation set...")
    val_performance = np.mean(eval_dataset(val_set, eval_fn, model))
    previous_performance = np.mean(results["validation_acc"][-1])
    print("Val acc: ", val_performance)
    print("Previous acc: ", previous_performance)
    previous_prompt = results["prompt"][-1]

    if val_performance < previous_performance:
        print(f"Rejected prompt: {system_prompt.value}")
        system_prompt.set_value(previous_prompt)
        val_performance = previous_performance

    results["validation_acc"].append(val_performance)


def get_eval_output(x, y, model, eval_fn):
    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y = tg.Variable(str(y), requires_grad=False, role_description="correct answer for the query")
    response = model(x)
    try:
        eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
    except:
        eval_output_variable = eval_fn([x, y, response])
    return eval_output_variable


args = config()
print(vars(args))
set_seed(args.seed)

# Build engines
def _make_engine(engine_name):
    name_l = (engine_name or "").lower()
    if 'llama' in name_l or 'qwen' in name_l or 'ollama' in name_l or 'mistral' in name_l or 'gemma' in name_l:
        return tg.get_engine(engine_name=engine_name, batch_size=args.num_threads)
    return tg.get_engine(engine_name=engine_name)

llm_api = _make_engine(args.backbone_engine)
tg.set_backward_engine(llm_api, override=True)

train_set, val_set, test_set, eval_fn = load_task(args.task, evaluation_api=llm_api)
print("Train/Val/Test Set Lengths: ", len(train_set), len(val_set), len(test_set))

BBH_OTHER_PROMPT2 = "You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: ($VALUE)' where VALUE is the letter of the correct option."
WORD_SORTING_PROMPT = "You will answer a reasoning question. Think step by step. The last line of your response must be in the format: 'Answer: [sorted words]', where [sorted words] are the alphabetically sorted words separated by a single space."
if "word_sorting" in args.task:
    STARTING_SYSTEM_PROMPT = WORD_SORTING_PROMPT
elif ("object_counting" in args.task) or ("GSM8K" in args.task):
    STARTING_SYSTEM_PROMPT = train_set.get_task_description()
else:
    STARTING_SYSTEM_PROMPT = BBH_OTHER_PROMPT2

train_loader = tg.tasks.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
print(f'Starting system prompt: {STARTING_SYSTEM_PROMPT}')

system_prompt = tg.Variable(
    STARTING_SYSTEM_PROMPT,
    requires_grad=True,
    role_description="structured system prompt to a somewhat capable language model that specifies the behavior and strategies for the QA task"
)

if 'llama' in args.model:
    model_api = tg.get_engine(engine_name=args.model, batch_size=args.num_threads)
else:
    model_api = tg.get_engine(engine_name=args.model)

model = tg.BlackboxLLM(model_api, system_prompt)

# TextReg uses the reg-aware optimizer with soft projection
optimizer = rv.TextualGradientDescentReg(
    engine=llm_api, parameters=[system_prompt],
    do_soft_projection=getattr(args, "soft_projection", True),
)

results = {
    "test_acc_mean": [],
    "test_acc": [],
    "prompt": [],
    "validation_acc": [],
    "adaptive_sparse_metrics": [],
}

rulebank = RuleBank()

# Build tokenizer_fn for the adaptive sparse pipeline
tokenizer_fn_pre = None
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    tokenizer_fn_pre = lambda s: len(_enc.encode(s or ""))
except Exception:
    tokenizer_fn_pre = lambda s: len((s or "").split())

# Initial evaluation
print("Evaluating the initial prompt on the test set...")
results["test_acc"].append(eval_dataset(test_set, eval_fn, model))
print("Evaluating the initial prompt on the validation set...")
results["validation_acc"].append(eval_dataset(val_set, eval_fn, model))
results["prompt"].append(system_prompt.get_value())

previous_prompt_for_constraints = STARTING_SYSTEM_PROMPT

for epoch in range(args.max_epochs):
    for steps, (batch_x, batch_y) in enumerate((pbar := tqdm(train_loader, position=0))):
        pbar.set_description(f"Training step {steps}. Epoch {epoch}")
        current_prompt_for_constraints = system_prompt.value
        optimizer.zero_grad()
        losses = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = []
            for (x, y) in zip(batch_x, batch_y):
                future = executor.submit(get_eval_output, x, y, model, eval_fn)
                futures.append(future)
            tqdm_loader = tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)
            for future in tqdm_loader:
                eval_output_variable = future.result()
                losses.append(eval_output_variable)

        total_loss = tg.sum(losses)
        total_loss.backward()

        # TextReg regularization pipeline
        sparse_metrics = apply_adaptive_sparse_pipeline(
            system_prompt,
            current_prompt_for_constraints,
            previous_prompt_for_constraints,
            STARTING_SYSTEM_PROMPT,
            tokenizer_fn_pre,
            llm_api,
            rulebank=rulebank,
            length_threshold=args.reg_length_threshold,
            verbose=True,
        )
        metrics_to_log = {k: v for k, v in sparse_metrics.items() if k != "purification_result"}
        results["adaptive_sparse_metrics"].append(metrics_to_log)

        optimizer.step()
        prompt_after_step = system_prompt.get_value()

        if args.run_validation:
            run_validation_revert(system_prompt, results, model, eval_fn, val_set)
        # Only update previous_prompt when no revert occurred
        if not args.run_validation or system_prompt.get_value() == prompt_after_step:
            previous_prompt_for_constraints = current_prompt_for_constraints
        print("Current sys prompt: ", system_prompt)
        print("Evaluating the current prompt on the test set...")
        test_acc = eval_dataset(test_set, eval_fn, model)
        results["test_acc"].append(test_acc)
        test_acc_mean = np.mean(test_acc)
        results["test_acc_mean"].append(test_acc_mean)
        results["prompt"].append(system_prompt.get_value())
        if steps >= args.max_steps - 1:
            break

model_name = args.backbone_engine.split("/")[-1]
model_name2 = args.model.split("/")[-1]
current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"results_{args.task}_{model_name}_{model_name2}_{current_time_str}.json"

if getattr(args, "result_dir", ""):
    result_dir = args.result_dir
else:
    result_dir = f"./results/textreg/{model_name2}"

os.makedirs(result_dir, exist_ok=True)
filepath = os.path.join(result_dir, filename)
print(f"Results saved to: {filepath}")
with open(filepath, "w") as f:
    json.dump(results, f, indent=4)
