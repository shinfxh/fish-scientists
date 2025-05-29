#!/usr/bin/env python3
"""
Test script to verify the ablation setup is working correctly.

This script performs basic checks without running full experiments:
1. Imports and dependencies
2. Configuration loading
3. Manager initialization
4. Basic functionality tests
"""

import sys
import os
import tempfile
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required imports work"""
    print("Testing imports...")
    
    try:
        from ai_scientist.llm import create_client
        from ai_scientist.treesearch.utils.config import load_cfg, prep_cfg
        from ai_scientist.treesearch.agent_manager import AgentManager, Stage
        from ai_scientist.treesearch.journal import Journal
        from ai_scientist.treesearch.parallel_agent import ParallelAgent
        from ai_scientist.treesearch.utils.config import save_run
        from ai_scientist.perform_plotting import aggregate_plots
        from ai_scientist.utils.token_tracker import token_tracker
        print("✓ All AI-Scientist imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    try:
        from launch_ablation import AblationAgentManager, AblationConfig
        print("✓ Ablation launcher imports successful")
    except ImportError as e:
        print(f"✗ Ablation launcher import error: {e}")
        return False
    
    return True


def test_config_loading():
    """Test configuration loading"""
    print("\nTesting configuration loading...")
    
    try:
        from ai_scientist.treesearch.utils.config import load_cfg
        
        # Check if config file exists
        config_path = Path("bfts_config.yaml")
        if not config_path.exists():
            print(f"✗ Config file not found: {config_path}")
            return False
        
        # Try loading config
        cfg = load_cfg(config_path)
        print(f"✓ Configuration loaded successfully")
        print(f"  - Agent type: {cfg.agent.type}")
        print(f"  - Number of workers: {cfg.agent.num_workers}")
        print(f"  - Code model: {cfg.agent.code.model}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration loading error: {e}")
        return False


def test_baseline_code():
    """Test baseline code exists and is valid"""
    print("\nTesting baseline code...")
    
    baseline_path = Path("example_baseline_cifar10.py")
    if not baseline_path.exists():
        print(f"✗ Baseline code not found: {baseline_path}")
        return False
    
    # Try to parse the baseline code
    try:
        with open(baseline_path, 'r') as f:
            code = f.read()
        
        # Basic checks
        if "def main(" in code:
            print("✓ Baseline code has main() function")
        else:
            print("✗ Baseline code missing main() function")
            return False
        
        if "import torch" in code:
            print("✓ Baseline code imports PyTorch")
        else:
            print("✗ Baseline code missing PyTorch imports")
            return False
        
        print(f"✓ Baseline code is valid ({len(code)} characters)")
        return True
        
    except Exception as e:
        print(f"✗ Error reading baseline code: {e}")
        return False


def test_manager_initialization():
    """Test that AblationAgentManager can be initialized"""
    print("\nTesting AblationAgentManager initialization...")
    
    try:
        from launch_ablation import AblationAgentManager, create_custom_config
        from ai_scientist.treesearch.utils.config import load_cfg
        
        # Create minimal task description
        task_desc = {
            "Title": "Test Ablation Study",
            "Abstract": "Test ablation experiments",
            "Short Hypothesis": "Test systematic analysis",
            "objective": "Test optimization",
            "baseline_description": "Test baseline",
            "Code": "# Test code\ndef main():\n    return 42",
            "Experiments": "Test experiments",
            "Risk Factors and Limitations": "Test limitations"
        }
        
        # Create temporary workspace
        with tempfile.TemporaryDirectory() as temp_dir:
            # Load and modify config
            cfg = load_cfg(Path("bfts_config.yaml"))
            cfg.workspace_dir = temp_dir
            cfg.log_dir = os.path.join(temp_dir, "logs")
            cfg.data_dir = os.path.join(temp_dir, "data")
            cfg.exp_name = "test_experiment"
            
            # Create required directories
            os.makedirs(cfg.data_dir, exist_ok=True)
            os.makedirs(cfg.log_dir, exist_ok=True)
            
            # Test manager initialization for different modes
            for mode in ["hyperparameter", "ablation", "both"]:
                print(f"  Testing mode: {mode}")
                manager = AblationAgentManager(
                    task_desc=task_desc,
                    cfg=cfg,
                    workspace_dir=Path(temp_dir),
                    mode=mode
                )
                
                # Check that stages were created correctly
                if mode == "hyperparameter":
                    assert len(manager.stages) == 1
                    assert manager.stages[0].name == "hyperparameter_tuning"
                elif mode == "ablation":
                    assert len(manager.stages) == 1
                    assert manager.stages[0].name == "ablation_studies"
                elif mode == "both":
                    assert len(manager.stages) == 2
                    assert manager.stages[0].name == "hyperparameter_tuning"
                    assert manager.stages[1].name == "ablation_studies"
                
                print(f"    ✓ {mode} mode: {len(manager.stages)} stages created")
        
        print("✓ AblationAgentManager initialization successful")
        return True
        
    except Exception as e:
        print(f"✗ Manager initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_command_line_parsing():
    """Test command line argument parsing"""
    print("\nTesting command line parsing...")
    
    try:
        from launch_ablation import parse_arguments, create_ablation_config
        
        # Mock sys.argv for testing
        original_argv = sys.argv
        sys.argv = [
            "launch_ablation.py",
            "--baseline_code", "example_baseline_cifar10.py",
            "--task_description", "Test optimization",
            "--mode", "hyperparameter",
            "--max_hyperparameter_iterations", "5"
        ]
        
        try:
            args = parse_arguments()
            config = create_ablation_config(args)
            
            assert args.baseline_code == "example_baseline_cifar10.py"
            assert args.task_description == "Test optimization"
            assert args.mode == "hyperparameter"
            assert args.max_hyperparameter_iterations == 5
            
            print("✓ Command line parsing successful")
            return True
            
        finally:
            sys.argv = original_argv
            
    except Exception as e:
        print(f"✗ Command line parsing error: {e}")
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("=" * 50)
    print("AI-SCIENTIST ABLATION TOOL - SETUP TEST")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration Loading", test_config_loading),
        ("Baseline Code", test_baseline_code),
        ("Manager Initialization", test_manager_initialization),
        ("Command Line Parsing", test_command_line_parsing),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print("-" * 50)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The ablation tool is ready to use.")
        print("\nTo get started, try:")
        print("python launch_ablation.py \\")
        print("    --baseline_code example_baseline_cifar10.py \\")
        print("    --task_description 'Optimize CIFAR-10 CNN performance' \\")
        print("    --mode hyperparameter \\")
        print("    --max_hyperparameter_iterations 5")
        return True
    else:
        print(f"\n❌ {total - passed} tests failed. Please fix the issues before using the tool.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 