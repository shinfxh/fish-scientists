# Quick Start Guide: AI-Scientist Ablation Tool

## 🚀 5-Minute Setup

### Step 1: Test Your Setup
```bash
python test_ablation_setup.py
```

If all tests pass, you're ready to go! If not, check the error messages and ensure all dependencies are installed.

### Step 2: Run Your First Experiment

**Hyperparameter Tuning Example:**
```bash
python launch_ablation.py \
    --baseline_code example_baseline_cifar10.py \
    --task_description "Optimize CIFAR-10 CNN for maximum accuracy" \
    --mode hyperparameter \
    --max_hyperparameter_iterations 5
```

**Ablation Study Example:**
```bash
python launch_ablation.py \
    --baseline_code example_baseline_cifar10.py \
    --task_description "Analyze which components are most important for CIFAR-10 performance" \
    --mode ablation \
    --max_ablation_iterations 5
```

## 📊 What You'll Get

After running experiments, you'll find:

```
ablation_workspaces/
└── ablation_{mode}_{timestamp}/
    ├── task_description.json     # Your experiment setup
    └── results/                  # Training outputs from experiments

ablation_logs/
└── ablation_{mode}_{timestamp}/
    ├── stage_hyperparameter_tuning/
    │   ├── journal.json          # All experiment results
    │   ├── tree_plot.html       # Interactive experiment tree
    │   └── best_solution_*.py   # Best configuration found
    └── token_tracker.json       # LLM usage costs
```

## 🔍 Understanding the Results

1. **Open `tree_plot.html`** in your browser to see the experiment tree
2. **Check `journal.json`** for detailed results of each experiment  
3. **Run `best_solution_*.py`** to reproduce the best configuration
4. **Review token costs** in `token_tracker.json`

## 💡 Using Your Own Code

Replace `example_baseline_cifar10.py` with your own baseline:

```bash
python launch_ablation.py \
    --baseline_code YOUR_MODEL.py \
    --task_description "YOUR OPTIMIZATION GOAL" \
    --mode both
```

**Your baseline code should:**
- Have a `main()` function that trains and evaluates your model
- Include configurable hyperparameters (learning rate, batch size, etc.)
- Save results that the AI can analyze
- Return a performance metric

## 🛠️ Troubleshooting

**Common Issues:**

1. **"Import Error"** → Install AI-Scientist dependencies: `pip install -r requirements.txt`
2. **"CUDA out of memory"** → Reduce `--num_workers` or batch size in your baseline
3. **"No improvements found"** → Increase `--max_*_iterations` or check your baseline code
4. **API rate limits** → Reduce workers or add delays in configuration

**Getting Help:**
- Run `python test_ablation_setup.py` to diagnose issues
- Check the full documentation in `README_ablation.md`
- Review example outputs in the generated directories

## 🎯 Pro Tips

1. **Start Small**: Use 5-10 iterations for initial testing
2. **Clear Objectives**: Be specific in your task description  
3. **Monitor Costs**: Check token usage regularly
4. **Iterative Improvement**: Use results to refine your baseline code

## 📚 Next Steps

- Read the full documentation: `README_ablation.md`
- Explore advanced configuration options
- Try different optimization objectives
- Scale up to larger experiments

---

**Happy Experimenting! 🧪**

The AI will systematically explore your hyperparameter space and analyze your model components, giving you insights that would take days of manual experimentation. 