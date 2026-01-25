#!/usr/bin/env python3
"""
Batch Test Runner

Executes comprehensive contact maintenance tests across all configurations.

Test Matrix (16 combinations):
- Robot Count: single (1) / multi (3)
- Model: dummy / wheel
- Kinematics: holonomic / diffdrive
- Controller: velocity / wrench

Usage:
    # Run all 16 combinations:
    python run_all_tests.py --output-dir /tmp/full_test/

    # Run specific subset:
    python run_all_tests.py --kinematics holonomic --output-dir /tmp/holo_tests/

    # Run with custom robot count:
    python run_all_tests.py --num-robots 5 --output-dir /tmp/multi_test/

    # Quick test (shorter duration):
    python run_all_tests.py --duration 5 --output-dir /tmp/quick_test/
"""
import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import rospkg
rospack = rospkg.RosPack()
pkg_path = rospack.get_path("contact_maintain")

# Path to comprehensive test script
COMPREHENSIVE_TEST_SCRIPT = Path(pkg_path) / "scripts" / "test" / "comprehensive_contact_test.py"


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

@dataclass
class TestScenario:
    """Single test scenario configuration."""
    num_robots: int
    kinematics: str
    model: str
    controller: str
    duration: float
    t_params: Optional[str] = None
    
    def get_name(self):
        robots_str = f"n{self.num_robots}" if self.num_robots > 1 else "single"
        return f"{self.kinematics}_{self.model}_{self.controller}_{robots_str}"
    
    def to_args(self, output_dir: str) -> List[str]:
        """Convert to command line arguments."""
        args = [
            sys.executable,
            str(COMPREHENSIVE_TEST_SCRIPT),
            "--num-robots", str(self.num_robots),
            "--kinematics", self.kinematics,
            "--model", self.model,
            "--controller", self.controller,
            "--duration", str(self.duration),
            "--no-gui",
            "--save-dir", os.path.join(output_dir, self.get_name()),
        ]
        if self.t_params:
            args.extend(["--t-params", self.t_params])
        return args


def generate_scenarios(
    num_robots_options: List[int],
    kinematics_options: List[str],
    model_options: List[str],
    controller_options: List[str],
    duration: float,
) -> List[TestScenario]:
    """Generate all test scenario combinations."""
    scenarios = []
    
    for num_robots, kinematics, model, controller in itertools.product(
        num_robots_options, kinematics_options, model_options, controller_options
    ):
        # Generate t_params for multi-robot
        if num_robots > 1:
            t_params = ",".join([f"{i/num_robots:.2f}" for i in range(num_robots)])
        else:
            t_params = "0.25"
        
        scenario = TestScenario(
            num_robots=num_robots,
            kinematics=kinematics,
            model=model,
            controller=controller,
            duration=duration,
            t_params=t_params,
        )
        scenarios.append(scenario)
    
    return scenarios


# ============================================================================
# TEST EXECUTION
# ============================================================================

def run_single_test(scenario: TestScenario, output_dir: str, verbose: bool = True) -> Dict:
    """Run a single test scenario.
    
    Returns
    -------
    dict
        Test result with success status and metrics.
    """
    name = scenario.get_name()
    result = {
        'name': name,
        'scenario': {
            'num_robots': scenario.num_robots,
            'kinematics': scenario.kinematics,
            'model': scenario.model,
            'controller': scenario.controller,
        },
        'success': False,
        'metrics': None,
        'error': None,
        'duration': 0,
    }
    
    start_time = time.time()
    
    try:
        args = scenario.to_args(output_dir)
        
        if verbose:
            print(f"\n  Running: {name}")
        
        # Run the test
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=scenario.duration * 10 + 60,  # Generous timeout
        )
        
        result['duration'] = time.time() - start_time
        
        # Return codes: 0=success, 134=SIGABRT (PyBullet cleanup), -6=SIGABRT on some systems
        if proc.returncode == 0 or proc.returncode == 134 or proc.returncode == -6:
            # Try to load metrics
            metrics_path = Path(output_dir) / name / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    result['metrics'] = json.load(f)
                result['success'] = True
            else:
                result['error'] = "Metrics file not found"
        else:
            result['error'] = f"Process failed with code {proc.returncode}"
            if proc.stderr:
                result['error'] += f": {proc.stderr[:500]}"
        
    except subprocess.TimeoutExpired:
        result['error'] = "Test timed out"
        result['duration'] = time.time() - start_time
    except Exception as e:
        result['error'] = str(e)
        result['duration'] = time.time() - start_time
    
    return result


def run_tests_sequential(scenarios: List[TestScenario], output_dir: str) -> List[Dict]:
    """Run all tests sequentially."""
    results = []
    
    for i, scenario in enumerate(scenarios):
        print(f"\n[{i+1}/{len(scenarios)}] {scenario.get_name()}")
        result = run_single_test(scenario, output_dir)
        results.append(result)
        
        status = "✓" if result['success'] else "✗"
        print(f"  {status} Completed in {result['duration']:.1f}s")
        if not result['success']:
            print(f"    Error: {result['error']}")
    
    return results


def run_tests_parallel(scenarios: List[TestScenario], output_dir: str, max_workers: int) -> List[Dict]:
    """Run tests in parallel."""
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_scenario = {
            executor.submit(run_single_test, scenario, output_dir, verbose=False): scenario
            for scenario in scenarios
        }
        
        completed = 0
        for future in as_completed(future_to_scenario):
            scenario = future_to_scenario[future]
            completed += 1
            
            try:
                result = future.result()
                results.append(result)
                status = "✓" if result['success'] else "✗"
                print(f"[{completed}/{len(scenarios)}] {status} {scenario.get_name()} ({result['duration']:.1f}s)")
            except Exception as e:
                results.append({
                    'name': scenario.get_name(),
                    'scenario': {},
                    'success': False,
                    'error': str(e),
                    'metrics': None,
                    'duration': 0,
                })
                print(f"[{completed}/{len(scenarios)}] ✗ {scenario.get_name()} (EXCEPTION)")
    
    return results


# ============================================================================
# RESULTS SUMMARY
# ============================================================================

def generate_summary(results: List[Dict], output_dir: str):
    """Generate summary report."""
    output_path = Path(output_dir)
    
    # JSON summary
    summary_json = {
        'total': len(results),
        'passed': sum(1 for r in results if r['success']),
        'failed': sum(1 for r in results if not r['success']),
        'results': results,
    }
    
    with open(output_path / "summary.json", 'w') as f:
        json.dump(summary_json, f, indent=2)
    
    # CSV summary
    csv_path = output_path / "summary.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Name', 'NumRobots', 'Kinematics', 'Model', 'Controller',
            'Success', 'AvgContactRatio', 'TotalContactLosses', 'OverallErrorRMSE',
            'Duration', 'Error'
        ])
        
        for r in results:
            m = r.get('metrics', {})
            s = m.get('summary', {}) if m else {}
            
            writer.writerow([
                r['name'],
                r['scenario'].get('num_robots', ''),
                r['scenario'].get('kinematics', ''),
                r['scenario'].get('model', ''),
                r['scenario'].get('controller', ''),
                r['success'],
                f"{s.get('avg_contact_ratio', 0)*100:.1f}%" if s else '',
                s.get('total_contact_losses', ''),
                f"{s.get('overall_position_error_rmse', 0)*100:.2f}cm" if s.get('overall_position_error_rmse') else '',
                f"{r['duration']:.1f}s",
                r.get('error', '')[:100] if r.get('error') else '',
            ])
    
    # Print summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    print(f"  Total:  {summary_json['total']}")
    print(f"  Passed: {summary_json['passed']}")
    print(f"  Failed: {summary_json['failed']}")
    print("-"*70)
    
    # Group by configuration dimension
    print("\nBy Kinematics:")
    for kin in ['holonomic', 'diffdrive']:
        subset = [r for r in results if r['scenario'].get('kinematics') == kin]
        passed = sum(1 for r in subset if r['success'])
        print(f"  {kin}: {passed}/{len(subset)} passed")
    
    print("\nBy Model:")
    for model in ['dummy', 'wheel']:
        subset = [r for r in results if r['scenario'].get('model') == model]
        passed = sum(1 for r in subset if r['success'])
        print(f"  {model}: {passed}/{len(subset)} passed")
    
    print("\nBy Controller:")
    for ctrl in ['velocity', 'wrench']:
        subset = [r for r in results if r['scenario'].get('controller') == ctrl]
        passed = sum(1 for r in subset if r['success'])
        print(f"  {ctrl}: {passed}/{len(subset)} passed")
    
    print("\n" + "="*70)
    print(f"  Results saved to: {output_path}")
    print("="*70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch Contact Maintenance Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all 16 combinations:
  python run_all_tests.py --output-dir /tmp/full_test/

  # Run only holonomic tests:
  python run_all_tests.py --kinematics holonomic --output-dir /tmp/holo/

  # Run only wheel model tests:
  python run_all_tests.py --model wheel --output-dir /tmp/wheel/

  # Run with 5 robots:
  python run_all_tests.py --num-robots 5 --output-dir /tmp/multi/

  # Quick test (5s duration):
  python run_all_tests.py --duration 5 --output-dir /tmp/quick/

  # Parallel execution (4 workers):
  python run_all_tests.py --parallel 4 --output-dir /tmp/parallel/

Test Matrix (16 default combinations):
  - Robot Count: single (1) / multi (3)
  - Model: dummy / wheel  
  - Kinematics: holonomic / diffdrive
  - Controller: velocity / wrench
"""
    )
    parser.add_argument("--output-dir", "-o", required=True,
                       help="Output directory for results (required)")
    parser.add_argument("--num-robots", type=str, default="1,3",
                       help="Robot counts to test, comma-separated (default: 1,3)")
    parser.add_argument("--kinematics", type=str, default="holonomic,diffdrive",
                       help="Kinematics types (default: holonomic,diffdrive)")
    parser.add_argument("--model", type=str, default="dummy,wheel",
                       help="Robot models (default: dummy,wheel)")
    parser.add_argument("--controller", type=str, default="velocity,wrench",
                       help="Controller types (default: velocity,wrench)")
    parser.add_argument("--duration", "-d", type=float, default=10.0,
                       help="Test duration in seconds (default: 10.0)")
    parser.add_argument("--parallel", "-p", type=int, default=0,
                       help="Number of parallel workers (default: 0 = sequential)")
    args = parser.parse_args()
    
    # Parse options
    num_robots_options = [int(x) for x in args.num_robots.split(',')]
    kinematics_options = [x.strip() for x in args.kinematics.split(',')]
    model_options = [x.strip() for x in args.model.split(',')]
    controller_options = [x.strip() for x in args.controller.split(',')]
    
    # Generate scenarios
    scenarios = generate_scenarios(
        num_robots_options=num_robots_options,
        kinematics_options=kinematics_options,
        model_options=model_options,
        controller_options=controller_options,
        duration=args.duration,
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print("="*70)
    print("  BATCH CONTACT MAINTENANCE TEST RUNNER")
    print("="*70)
    print(f"  Scenarios: {len(scenarios)}")
    print(f"  Robot counts: {num_robots_options}")
    print(f"  Kinematics: {kinematics_options}")
    print(f"  Models: {model_options}")
    print(f"  Controllers: {controller_options}")
    print(f"  Duration: {args.duration}s each")
    print(f"  Parallel: {args.parallel if args.parallel > 0 else 'Sequential'}")
    print(f"  Output: {output_dir}")
    print("="*70)
    
    # List scenarios
    print("\nScenarios to run:")
    for i, s in enumerate(scenarios):
        print(f"  {i+1}. {s.get_name()}")
    
    # Run tests
    start_time = time.time()
    
    if args.parallel > 0:
        print(f"\nRunning {len(scenarios)} tests with {args.parallel} workers...")
        results = run_tests_parallel(scenarios, str(output_dir), args.parallel)
    else:
        print(f"\nRunning {len(scenarios)} tests sequentially...")
        results = run_tests_sequential(scenarios, str(output_dir))
    
    total_time = time.time() - start_time
    
    # Generate summary
    generate_summary(results, str(output_dir))
    
    print(f"\n  Total execution time: {total_time:.1f}s")
    print("="*70)


if __name__ == "__main__":
    main()

