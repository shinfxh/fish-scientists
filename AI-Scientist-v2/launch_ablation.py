#!/usr/bin/env python3
"""
Custom AI-Scientist launcher focused on hyperparameter sweeping and ablation studies.

This script leverages the tree search capabilities of AI-Scientist while bypassing 
stages focused on ideation and creative research. Instead, it focuses on:
1. Hyperparameter tuning (systematic parameter optimization)
2. Ablation studies (component analysis and feature importance)

Usage:
    python launch_ablation.py --baseline_code path/to/baseline.py --task_description "description" --mode hyperparameter
    python launch_ablation.py --baseline_code path/to/baseline.py --task_description "description" --mode ablation
    python launch_ablation.py --baseline_code path/to/baseline.py --task_description "description" --mode both
"""

import os
import os.path as osp
import json
import argparse
import shutil
import torch
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from ai_scientist.llm import create_client
from ai_scientist.treesearch.utils.config import load_cfg, prep_cfg
from ai_scientist.treesearch.agent_manager import AgentManager, Stage
from ai_scientist.treesearch.journal import Journal
from ai_scientist.treesearch.parallel_agent import ParallelAgent
from ai_scientist.treesearch.utils.config import save_run
from ai_scientist.perform_plotting import aggregate_plots
from ai_scientist.utils.token_tracker import token_tracker


@dataclass
class AblationConfig:
    """Configuration for ablation-focused experiments"""
    baseline_code_path: str
    task_description: str
    mode: str  # 'hyperparameter', 'ablation', or 'both'
    max_hyperparameter_iterations: int = 15
    max_ablation_iterations: int = 10
    num_workers: int = 4
    workspace_dir: str = "ablation_workspaces"
    log_dir: str = "ablation_logs"
    model_code: str = "gpt-4o-2024-11-20"
    model_feedback: str = "gpt-4o-2024-11-20"
    skip_plotting: bool = False


class AblationAgentManager(AgentManager):
    """Custom AgentManager focused on hyperparameter tuning and ablation studies"""
    
    def __init__(self, task_desc: Dict[str, Any], cfg: Any, workspace_dir: Path, mode: str):
        # Initialize parent but override key components
        self.task_desc = task_desc
        self.cfg = cfg
        self.workspace_dir = workspace_dir
        self.current_stage_number = 0
        self.stages: List[Stage] = []
        self.current_stage: Optional[Stage] = None
        self.journals: Dict[str, Journal] = {}
        self.stage_history = []
        self.completed_stages: List[str] = []
        self.mode = mode
        
        # Custom stage definitions for ablation focus
        self.ablation_stage_goals = {
            "hyperparameter": """
                - Systematically tune hyperparameters to optimize model performance
                - Focus on learning rate, batch size, optimizer settings, regularization
                - Test different architectural hyperparameters (hidden sizes, layers, dropout rates)
                - Explore data augmentation and preprocessing parameters
                - DO NOT change the core model architecture or training logic
                - Aim to find the optimal hyperparameter configuration for best performance
            """,
            "ablation": """
                - Conduct systematic ablation studies to understand component contributions
                - Remove or modify specific components to measure their impact
                - Test the importance of different model components (layers, normalization, activation functions)
                - Analyze the effect of different training strategies and data augmentation techniques
                - Generate insights about what components are most critical for performance
                - Provide quantitative analysis of each component's contribution
            """
        }
        
        # Create appropriate stages based on mode
        self._create_ablation_stages()
    
    def _create_ablation_stages(self):
        """Create stages focused on hyperparameter tuning and/or ablation studies"""
        if self.mode in ['hyperparameter', 'both']:
            hyperparameter_stage = Stage(
                name="hyperparameter_tuning",
                description="Systematic hyperparameter optimization",
                goals=self.ablation_stage_goals["hyperparameter"],
                max_iterations=self.cfg.agent.stages.get('hyperparameter_max_iters', 15),
                num_drafts=self.cfg.agent.search.num_drafts,
                stage_number=1
            )
            self.stages.append(hyperparameter_stage)
            self.journals[hyperparameter_stage.name] = Journal()
            
        if self.mode in ['ablation', 'both']:
            ablation_stage = Stage(
                name="ablation_studies", 
                description="Component analysis and ablation studies",
                goals=self.ablation_stage_goals["ablation"],
                max_iterations=self.cfg.agent.stages.get('ablation_max_iters', 10),
                num_drafts=self.cfg.agent.search.num_drafts,
                stage_number=2 if self.mode == 'both' else 1
            )
            self.stages.append(ablation_stage)
            self.journals[ablation_stage.name] = Journal()
        
        # Set current stage to the first one
        if self.stages:
            self.current_stage = self.stages[0]
    
    def _curate_task_desc(self, stage: Stage) -> str:
        """Create task description focused on ablation objectives"""
        task_desc = f"""You are an AI researcher conducting systematic {stage.name} experiments.

Your objective: {self.task_desc.get('objective', 'Optimize model performance through systematic experimentation')}

Baseline Implementation:
{self.task_desc.get('baseline_description', 'Use the provided baseline code as your starting point')}

Current Stage: {stage.name}
Stage Goals: {stage.goals}

Important Guidelines:
- Use the provided baseline code as your foundation
- Make systematic, measurable changes
- Document all modifications clearly
- Focus on quantitative performance improvements
- Test changes thoroughly with proper evaluation metrics
"""
        
        if "Code" in self.task_desc:
            task_desc += f"\n\nBaseline Code:\n{self.task_desc['Code']}\n"
            
        return task_desc
    
    def _check_stage_completion(self, stage: Stage) -> tuple[bool, str]:
        """Check if current ablation stage is complete"""
        journal = self.journals[stage.name]
        
        # Complete if max iterations reached
        if len(journal.nodes) >= stage.max_iterations:
            return True, f"Reached maximum iterations ({stage.max_iterations})"
        
        # For hyperparameter tuning, complete when we have good improvements
        if stage.name == "hyperparameter_tuning":
            if len(journal.good_nodes) >= 3:  # Have at least 3 good hyperparameter configurations
                best_node = journal.get_best_node()
                if best_node and len(journal.nodes) >= 5:  # Have tested enough configurations
                    return True, "Found multiple good hyperparameter configurations"
        
        # For ablation studies, complete when we have systematic component analysis
        if stage.name == "ablation_studies":
            if len(journal.good_nodes) >= 2:  # Have analyzed multiple components
                if len(journal.nodes) >= 6:  # Have done enough ablations
                    return True, "Completed systematic component analysis"
        
        return False, "Continue exploring"
    
    def run(self, exec_callback=None, step_callback=None):
        """Run the ablation-focused experiment pipeline"""
        for stage in self.stages:
            print(f"\n{'='*50}")
            print(f"Starting Stage: {stage.name}")
            print(f"Goals: {stage.goals}")
            print(f"{'='*50}\n")
            
            self.current_stage = stage
            agent = self._create_agent_for_stage(stage)
            
            # Run the stage until completion
            while not self._check_stage_completion(stage)[0]:
                try:
                    agent.step(exec_callback=exec_callback)
                    if step_callback:
                        step_callback(stage, self.journals[stage.name])
                except Exception as e:
                    print(f"Error in stage {stage.name}: {e}")
                    break
            
            completion_status = self._check_stage_completion(stage)
            print(f"\nStage {stage.name} completed: {completion_status[1]}")
            self.completed_stages.append(stage.name)
            
            # Save stage results
            save_run(self.cfg, self.journals[stage.name], f"stage_{stage.name}")
    
    def _create_agent_for_stage(self, stage: Stage) -> ParallelAgent:
        """Create agent for ablation-focused stage"""
        stage_cfg = self.cfg.copy()
        stage_cfg.agent.search.num_drafts = stage.num_drafts
        task_desc = self._curate_task_desc(stage)
        
        return ParallelAgent(
            task_desc=task_desc,
            cfg=stage_cfg,
            journal=self.journals[stage.name],
            stage_name=stage.name,
            best_stage3_node=None,
            best_stage2_node=None,
            best_stage1_node=None,
        )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run ablation-focused AI scientist experiments")
    parser.add_argument(
        "--baseline_code",
        type=str,
        required=True,
        help="Path to baseline code file (.py) to use as starting point"
    )
    parser.add_argument(
        "--task_description", 
        type=str,
        required=True,
        help="Description of the task/objective for the experiments"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hyperparameter", "ablation", "both"],
        default="both",
        help="Type of experiments to run"
    )
    parser.add_argument(
        "--max_hyperparameter_iterations",
        type=int,
        default=15,
        help="Maximum iterations for hyperparameter tuning stage"
    )
    parser.add_argument(
        "--max_ablation_iterations",
        type=int,
        default=10,
        help="Maximum iterations for ablation studies stage"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--workspace_dir",
        type=str,
        default="ablation_workspaces",
        help="Directory for workspace files"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="ablation_logs", 
        help="Directory for log files"
    )
    parser.add_argument(
        "--model_code",
        type=str,
        default="gpt-4o-2024-11-20",
        help="Model to use for code generation"
    )
    parser.add_argument(
        "--model_feedback",
        type=str,
        default="gpt-4o-2024-11-20",
        help="Model to use for feedback and evaluation"
    )
    parser.add_argument(
        "--skip_plotting",
        action="store_true",
        help="Skip plot aggregation step"
    )
    parser.add_argument(
        "--config_template",
        type=str,
        default="bfts_config.yaml",
        help="Base configuration file to use as template"
    )
    return parser.parse_args()


def create_ablation_config(args) -> AblationConfig:
    """Create configuration for ablation experiments"""
    return AblationConfig(
        baseline_code_path=args.baseline_code,
        task_description=args.task_description,
        mode=args.mode,
        max_hyperparameter_iterations=args.max_hyperparameter_iterations,
        max_ablation_iterations=args.max_ablation_iterations,
        num_workers=args.num_workers,
        workspace_dir=args.workspace_dir,
        log_dir=args.log_dir,
        model_code=args.model_code,
        model_feedback=args.model_feedback,
        skip_plotting=args.skip_plotting
    )


def setup_experiment_directory(ablation_config: AblationConfig) -> tuple[str, Dict[str, Any]]:
    """Set up experiment directory and task description"""
    # Verify baseline code exists
    if not os.path.exists(ablation_config.baseline_code_path):
        raise FileNotFoundError(f"Baseline code file not found: {ablation_config.baseline_code_path}")
    
    # Read baseline code
    with open(ablation_config.baseline_code_path, 'r') as f:
        baseline_code = f.read()
    
    # Create experiment directory
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = f"ablation_{ablation_config.mode}_{date}"
    exp_dir = osp.join(ablation_config.workspace_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create task description
    task_desc = {
        "Title": f"Ablation Study: {ablation_config.mode.title()} Optimization",
        "Abstract": f"Systematic {ablation_config.mode} experiments on provided baseline implementation.",
        "Short Hypothesis": f"Systematic {ablation_config.mode} analysis will reveal optimization opportunities and component importance.",
        "objective": ablation_config.task_description,
        "baseline_description": f"Baseline code loaded from {ablation_config.baseline_code_path}",
        "Code": baseline_code,
        "Experiments": f"Conduct {ablation_config.mode} experiments",
        "Risk Factors and Limitations": "Limited to modifications of provided baseline code"
    }
    
    # Save task description
    with open(osp.join(exp_dir, "task_description.json"), 'w') as f:
        json.dump(task_desc, f, indent=2)
    
    return exp_dir, task_desc


def create_custom_config(base_config_path: str, ablation_config: AblationConfig, exp_dir: str):
    """Create custom configuration for ablation experiments"""
    # Load base config
    base_cfg = load_cfg(Path(base_config_path))
    
    # Override settings for ablation focus
    base_cfg.workspace_dir = exp_dir
    base_cfg.log_dir = osp.join(ablation_config.log_dir, osp.basename(exp_dir))
    base_cfg.data_dir = osp.join(exp_dir, "data")
    base_cfg.exp_name = osp.basename(exp_dir)
    
    # Create data directory
    os.makedirs(base_cfg.data_dir, exist_ok=True)
    
    # Ablation-specific agent settings
    base_cfg.agent.num_workers = ablation_config.num_workers
    base_cfg.agent.code.model = ablation_config.model_code
    base_cfg.agent.feedback.model = ablation_config.model_feedback
    
    # Stage-specific iterations
    if not hasattr(base_cfg.agent, 'stages'):
        base_cfg.agent.stages = {}
    
    base_cfg.agent.stages.hyperparameter_max_iters = ablation_config.max_hyperparameter_iterations
    base_cfg.agent.stages.ablation_max_iters = ablation_config.max_ablation_iterations
    
    return prep_cfg(base_cfg)


def print_experiment_summary(exp_dir: str, ablation_config: AblationConfig, journals: Dict[str, Journal]):
    """Print summary of ablation experiment results"""
    print(f"\n{'='*60}")
    print("ABLATION EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Experiment Directory: {exp_dir}")
    print(f"Mode: {ablation_config.mode}")
    print(f"Baseline Code: {ablation_config.baseline_code_path}")
    
    for stage_name, journal in journals.items():
        print(f"\n--- {stage_name.upper()} RESULTS ---")
        print(f"Total experiments: {len(journal.nodes)}")
        print(f"Successful experiments: {len(journal.good_nodes)}")
        print(f"Failed experiments: {len(journal.buggy_nodes)}")
        
        best_node = journal.get_best_node()
        if best_node:
            print(f"Best performance: {best_node.metric}")
            print(f"Best experiment ID: {best_node.id}")
        else:
            print("No successful experiments found")
    
    print(f"\n{'='*60}")


def save_token_tracker(exp_dir: str):
    """Save token usage tracking information"""
    with open(osp.join(exp_dir, "token_tracker.json"), "w") as f:
        json.dump(token_tracker.get_summary(), f)
    with open(osp.join(exp_dir, "token_tracker_interactions.json"), "w") as f:
        json.dump(token_tracker.get_interactions(), f)


def main():
    args = parse_arguments()
    
    # Set up AI-Scientist root
    os.environ["AI_SCIENTIST_ROOT"] = os.path.dirname(os.path.abspath(__file__))
    print(f"Set AI_SCIENTIST_ROOT to {os.environ['AI_SCIENTIST_ROOT']}")
    
    # Create ablation configuration
    ablation_config = create_ablation_config(args)
    
    # Setup experiment
    exp_dir, task_desc = setup_experiment_directory(ablation_config)
    print(f"Experiment directory: {exp_dir}")
    
    # Create custom configuration
    cfg = create_custom_config(args.config_template, ablation_config, exp_dir)
    
    # Create ablation-focused agent manager
    manager = AblationAgentManager(
        task_desc=task_desc,
        cfg=cfg,
        workspace_dir=Path(exp_dir),
        mode=ablation_config.mode
    )
    
    def step_callback(stage, journal):
        print(f"Completed iteration in {stage.name}")
        print(f"Total experiments: {len(journal.nodes)}")
        if journal.get_best_node():
            print(f"Best performance: {journal.get_best_node().metric}")
    
    # Run experiments
    print(f"\nStarting {ablation_config.mode} experiments...")
    manager.run(step_callback=step_callback)
    
    # Generate plots if not skipped
    if not ablation_config.skip_plotting:
        try:
            print("Generating plots...")
            aggregate_plots(base_folder=exp_dir, model=ablation_config.model_feedback)
        except Exception as e:
            print(f"Warning: Plot generation failed: {e}")
    
    # Save token tracking
    save_token_tracker(exp_dir)
    
    # Print summary
    print_experiment_summary(exp_dir, ablation_config, manager.journals)
    
    print(f"\nAblation experiments completed!")
    print(f"Results saved in: {exp_dir}")
    print(f"Logs saved in: {cfg.log_dir}")


if __name__ == "__main__":
    main() 