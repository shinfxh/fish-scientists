# AI-Scientist Ablation Tool

A focused implementation of AI-Scientist's tree search capabilities for hyperparameter sweeping and ablation studies. This tool bypasses the ideation and creative research stages to focus specifically on systematic optimization and component analysis.

## Overview

Instead of using the AI-Scientist as a black box for end-to-end research, this tool leverages its powerful tree search and parallel experimentation capabilities for targeted tasks:

1. **Hyperparameter Tuning**: Systematic optimization of model parameters
2. **Ablation Studies**: Component analysis to understand feature importance
3. **Both**: Combined hyperparameter and ablation analysis

## Key Features

- **Focused Workflow**: Bypasses unnecessary stages (ideation, creative research)
- **Tree Search Optimization**: Uses AI-Scientist's sophisticated search algorithms
- **Parallel Execution**: Runs multiple experiments simultaneously
- **Systematic Analysis**: LLM-guided parameter exploration and component analysis
- **Detailed Logging**: Comprehensive experiment tracking and visualization

## How It Works

### Traditional AI-Scientist Pipeline
```
Ideation → Initial Implementation → Baseline Tuning → Creative Research → Ablation Studies → Writeup
```

### Ablation-Focused Pipeline
```
Baseline Code → Hyperparameter Tuning → Ablation Studies → Results Analysis
```

The tool creates a custom `AblationAgentManager` that:
- Skips the ideation and initial implementation stages
- Directly uses your provided baseline code
- Focuses the LLM on systematic parameter optimization
- Conducts structured ablation experiments
- Provides detailed performance analysis

## Installation

Ensure you have the AI-Scientist dependencies installed:

```bash
# Install AI-Scientist requirements
pip install -r requirements.txt

# Make the script executable
chmod +x launch_ablation.py
```

## Usage

### Basic Usage

```bash
python launch_ablation.py \
    --baseline_code example_baseline_cifar10.py \
    --task_description "Optimize CIFAR-10 CNN performance" \
    --mode both
```

### Hyperparameter Tuning Only

```bash
python launch_ablation.py \
    --baseline_code your_model.py \
    --task_description "Find optimal hyperparameters for image classification" \
    --mode hyperparameter \
    --max_hyperparameter_iterations 20
```

### Ablation Studies Only

```bash
python launch_ablation.py \
    --baseline_code your_model.py \
    --task_description "Analyze component importance in my model" \
    --mode ablation \
    --max_ablation_iterations 15
```

### Advanced Configuration

```bash
python launch_ablation.py \
    --baseline_code your_model.py \
    --task_description "Comprehensive model optimization" \
    --mode both \
    --max_hyperparameter_iterations 25 \
    --max_ablation_iterations 15 \
    --num_workers 8 \
    --model_code "gpt-4o-2024-11-20" \
    --model_feedback "gpt-4o-2024-11-20" \
    --workspace_dir "my_experiments" \
    --log_dir "my_logs"
```

## Command Line Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--baseline_code` | str | Yes | - | Path to baseline code file (.py) |
| `--task_description` | str | Yes | - | Description of optimization objective |
| `--mode` | str | No | "both" | Experiment type: "hyperparameter", "ablation", or "both" |
| `--max_hyperparameter_iterations` | int | No | 15 | Max iterations for hyperparameter tuning |
| `--max_ablation_iterations` | int | No | 10 | Max iterations for ablation studies |
| `--num_workers` | int | No | 4 | Number of parallel workers |
| `--workspace_dir` | str | No | "ablation_workspaces" | Workspace directory |
| `--log_dir` | str | No | "ablation_logs" | Log directory |
| `--model_code` | str | No | "gpt-4o-2024-11-20" | Model for code generation |
| `--model_feedback` | str | No | "gpt-4o-2024-11-20" | Model for feedback |
| `--skip_plotting` | bool | No | False | Skip plot generation |
| `--config_template` | str | No | "bfts_config.yaml" | Base config file |

## Baseline Code Requirements

Your baseline code should be a Python script that:

1. **Contains a main training/evaluation loop**
2. **Includes configurable hyperparameters** (as variables or config dict)
3. **Returns a performance metric** (accuracy, F1-score, etc.)
4. **Handles its own data loading and preprocessing**
5. **Saves results** (for the LLM to analyze)

### Example Structure

```python
def main():
    # Configuration (hyperparameters)
    config = {
        'learning_rate': 0.001,
        'batch_size': 128,
        'epochs': 20,
        # ... other hyperparameters
    }
    
    # Model setup
    model = create_model(config)
    
    # Training
    train_results = train_model(model, config)
    
    # Evaluation
    test_accuracy = evaluate_model(model)
    
    # Save results for analysis
    save_results(config, train_results, test_accuracy)
    
    return test_accuracy

if __name__ == "__main__":
    main()
```

## Example: CIFAR-10 Experiments

We provide a complete CIFAR-10 example:

```bash
# Run hyperparameter tuning on the example
python launch_ablation.py \
    --baseline_code example_baseline_cifar10.py \
    --task_description "Optimize CIFAR-10 CNN for maximum accuracy" \
    --mode hyperparameter \
    --max_hyperparameter_iterations 20

# Run ablation studies
python launch_ablation.py \
    --baseline_code example_baseline_cifar10.py \
    --task_description "Analyze which components are most important for CIFAR-10 performance" \
    --mode ablation \
    --max_ablation_iterations 12
```

The example baseline achieves ~72% accuracy and includes:
- Configurable hyperparameters (learning rate, batch size, dropout, etc.)
- Multiple optimizer options (Adam, SGD, AdamW)
- Data augmentation toggles
- Architecture parameters (hidden size, dropout rate)
- Learning rate schedulers

## What the LLM Will Optimize

### Hyperparameter Tuning Stage
The LLM will systematically explore:
- **Learning rates**: Different values and schedules
- **Batch sizes**: Impact on training dynamics
- **Optimizers**: Adam vs SGD vs AdamW with different settings
- **Regularization**: Dropout rates, weight decay
- **Architecture**: Hidden sizes, layer configurations
- **Data augmentation**: Different augmentation strategies
- **Training**: Number of epochs, early stopping

### Ablation Studies Stage
The LLM will analyze:
- **Component removal**: What happens without specific layers/components
- **Architecture variants**: Different activation functions, normalization
- **Training strategies**: Impact of different training techniques
- **Data preprocessing**: Effect of different preprocessing steps
- **Feature importance**: Which parts of the model matter most

## Output and Results

Each experiment creates:

```
ablation_workspaces/ablation_{mode}_{timestamp}/
├── task_description.json          # Experiment configuration
├── results/                       # Training outputs
├── data/                         # Data files
└── token_tracker.json           # LLM usage tracking

ablation_logs/ablation_{mode}_{timestamp}/
├── stage_hyperparameter_tuning/  # Hyperparameter results
│   ├── journal.json              # Experiment tree
│   ├── tree_plot.html           # Visualization
│   └── best_solution_*.py       # Best configuration
├── stage_ablation_studies/       # Ablation results
│   └── ...
└── manager.pkl                   # Complete experiment state
```

### Key Output Files

- **`journal.json`**: Complete tree of all experiments with results
- **`tree_plot.html`**: Interactive visualization of the experiment tree
- **`best_solution_*.py`**: Code for the best performing configuration
- **`token_tracker.json`**: LLM usage and cost tracking

## Tips for Best Results

1. **Clear Task Description**: Provide specific optimization objectives
2. **Well-Structured Baseline**: Ensure your code is modular and configurable
3. **Reasonable Iterations**: Start with fewer iterations for testing
4. **Monitor Progress**: Check intermediate results to adjust parameters
5. **Resource Management**: Consider computational cost for complex models

## Advanced Usage

### Custom Configuration

You can modify the base BFTS configuration by editing `bfts_config.yaml` or providing a custom template:

```bash
python launch_ablation.py \
    --config_template my_custom_config.yaml \
    --baseline_code my_model.py \
    # ... other args
```

### Integration with Existing Workflows

The tool can be integrated into larger ML pipelines:

```python
from launch_ablation import AblationAgentManager, create_custom_config

# Use programmatically
config = create_custom_config(...)
manager = AblationAgentManager(...)
manager.run()
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure AI-Scientist dependencies are installed
2. **CUDA Issues**: Check GPU availability and PyTorch installation
3. **Memory Errors**: Reduce batch size or number of workers
4. **API Limits**: Monitor token usage and adjust iteration counts

### Debug Mode

For debugging, you can:
- Reduce `max_*_iterations` to 1-2
- Use `--num_workers 1` for simpler debugging
- Check logs in the experiment directories

## Contributing

To extend the ablation tool:

1. Modify `AblationAgentManager` for new experiment types
2. Add custom stage goals for domain-specific optimization
3. Implement new completion criteria for different metrics
4. Add support for additional baseline code formats

## License

Same as AI-Scientist project license. 