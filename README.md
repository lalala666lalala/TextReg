# TextReg: Mitigating Prompt Distributional Overfitting via Regularized Text-Space Optimization

## Abstract

Large language models (LLMs) are highly sensitive to the prompts used to specify task objectives and behavioral constraints. Many recent prompt optimization methods iteratively rewrite prompts using LLM-generated feedback, but the resulting prompts often become longer, accumulate narrow sample-specific rules, and generalize poorly beyond the training distribution. We study this failure mode as *prompt distributional overfitting* and argue that it reflects a lack of representation control in discrete text-space optimization. We formalize this view through *representational inefficiency*, a dual-factor measure that decomposes prompt inefficiency into capacity cost and scope narrowness, attributing distributional prompt overfitting to their coupled growth during optimization. We propose **TextReg**, a regularization framework that realizes a soft-penalty objective through regularized textual gradients, combining Dual-Evidence Gradient Purification, Semantic Edit Regularization, and Regularization-Guided Prompt Update. Across multiple reasoning benchmarks, **TextReg** substantially improves out-of-distribution (OOD) generalization, with accuracy gains of up to **+11.8%** over TextGrad and **+16.5%** over REVOLVE.


# TextReg

TextReg is a prompt optimization method that adds **adaptive sparse regularization** on top of textual gradient descent (TextGrad). It prevents prompt bloating and overfitting to training samples during optimization, resulting in better out-of-distribution (OOD) generalization.

## Project Structure

```
textreg/                          # Core TextReg package
  autograd/
    regularization.py             # Adaptive sparse pipeline (M_delta + state machine + guidance)
    gradient_purification_gate.py # 3-tier gradient purification gate
    rulebank.py                   # RuleBank: rule frequency tracker
  optimizer/
    optimizer_reg.py              # TextualGradientDescentReg (reg-aware optimizer)
    optimizer_prompts_reg.py      # Optimizer prompt templates with <REG_FEEDBACK>
    optimizer.py                  # Base TextualGradientDescent
  engine/                         # LLM backends (OpenAI, Anthropic, etc.)
  tasks/                          # Benchmark datasets (BBH, GSM8K, MMLU, etc.)

evaluation/
  prompt_optimization_reg.py      # TextReg entry point
  prompt_optimization.py          # TextGrad baseline entry point
```

## Setup

### Requirements

- Python 3.10+
- An OpenAI API key (or compatible endpoint) for the backbone engine

### Install dependencies

```bash
pip install openai tiktoken tqdm numpy python-dotenv
```

### Environment variables

```bash
export OPENAI_API_KEY="your-api-key"

# Optional: custom API endpoint
export OPENAI_API_BASE="https://api.openai.com/v1"
```

If using a local model via Ollama:

```bash
# Start Ollama server first, then:
export OLLAMA_BASE_URL="http://localhost:11434/v1"
```

## Usage

### Run TextReg

```bash
python evaluation/prompt_optimization_reg.py \
    --task BBH_object_counting \
    --backbone_engine gpt-4o \
    --model ollama-Qwen/Qwen2.5-7B-Instruct \
    --soft_projection \
    --batch_size 3 \
    --max_steps 12 \
    --seed 42 \
    --run_validation \
    --result_dir results/textreg
```

### Run TextGrad baseline

```bash
python evaluation/prompt_optimization.py \
    --task BBH_object_counting \
    --backbone_engine gpt-4o \
    --model ollama-Qwen/Qwen2.5-7B-Instruct \
    --batch_size 3 \
    --max_steps 12 \
    --seed 42 \
    --run_validation \
    --result_dir results/baseline
```

### Key arguments

| Argument | Description | Default |
|---|---|---|
| `--task` | Benchmark task name (e.g., `BBH_object_counting`, `BBH_logical_deduction_three_objects`, `GSM8K_DSPy`) | `BBH_object_counting` |
| `--backbone_engine` | LLM for backward pass, purification, M_delta, and optimizer | `gpt-4o` |
| `--model` | Solver model whose prompt is being optimized | `ollama-meta-llama/Llama-3.1-8B-Instruct` |
| `--soft_projection` | Enable soft-projection trailing in optimizer (recommended) | off |
| `--batch_size` | Training minibatch size | `3` |
| `--max_steps` | Maximum optimization steps per epoch | `12` |
| `--max_epochs` | Maximum number of epochs | `1` |
| `--reg_length_threshold` | rho_t threshold for triggering length regularization | `0.2` |
| `--run_validation` | Enable validation-based prompt revert | off |
| `--seed` | Random seed | `42` |
| `--num_threads` | Number of threads for parallel evaluation | `3` |
| `--result_dir` | Directory to save result JSON | auto |

### Supported tasks

- **BBH**: `BBH_object_counting`, `BBH_word_sorting`, `BBH_logical_deduction_three_objects`, `BBH_logical_deduction_five_objects`, `BBH_logical_deduction_seven_objects`, `BBH_tracking_shuffled_objects_five_objects`
- **GSM8K**: `GSM8K_DSPy`

## Output

Results are saved as JSON files in `--result_dir` with the following structure:

```json
{
    "test_acc": [[...], [...]],
    "test_acc_mean": [0.45, 0.52, ...],
    "prompt": ["initial prompt", "step 1 prompt", ...],
    "validation_acc": [[...]],
    "adaptive_sparse_metrics": [
        {"rho_t": 0.15, "reg_mode": "NO_REGULARIZATION", ...},
        {"rho_t": 0.32, "reg_mode": "STRONG_REGULARIZATION", ...}
    ]
}
```

## How TextReg Works

Each optimization step follows this pipeline:

```
Forward: model(system_prompt, x) -> response
Eval:    loss(response, ground_truth) -> score
Backward: textual gradients flow back to system_prompt
    |
    v
[Gradient Purification Gate]
    Classify each gradient as generalizable / narrow patch / pure style
    Reject narrow patches and style-only feedback
    |
    v
[RuleBank Update]
    Extract canonical rules from accepted gradients
    Track rule frequency across steps
    |
    v
[M_delta Semantic Delta Analyzer]
    Compare previous vs current prompt
    Classify changes: GENERALIZED_RULE / CASE_PATCH / STYLE_ONLY
    Output specificity_direction: increase / decrease / neutral
    |
    v
[Binary State Machine]
    rho_t (length change) + specificity_direction -> reg mode
    STRONG_REGULARIZATION / COMPRESSION_ONLY / GENERALIZE_ONLY / NO_REGULARIZATION
    |
    v
[Regularization Guidance Generator]
    LLM generates specific directives based on reg mode
    Injected as synthetic gradient into system_prompt.gradients
    |
    v
[TextualGradientDescentReg.step()]
    Splits gradients into <REG_FEEDBACK> and <CONTEXT>
    Soft projection: coordinate task fixes with reg directives
    LLM rewrites the system prompt
```
