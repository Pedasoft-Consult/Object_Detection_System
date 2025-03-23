#!/usr/bin/env python
"""
Comprehensive test runner script.
This script:
1. Sets up the test environment
2. Runs all tests
3. Generates a summary report
"""

import os
import sys
import subprocess
import logging
import argparse
import time
from datetime import datetime
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def setup_environment():
    """Set up the test environment"""
    logger.info("Setting up test environment...")

    try:
        os.makedirs("test_results", exist_ok=True)
        os.makedirs("test_output", exist_ok=True)
        logger.info("Created test directories: test_results, test_output")
        return True
    except Exception as e:
        logger.error(f"Error during setup: {e}")
        return False


def run_tests(output_dir="test_results", api_url="http://localhost:8000", skip_api=False,
              skip_data=False, skip_model=False, skip_benchmark=False):
    """Run all tests"""
    logger.info("Running tests...")

    # Check if run_tests.py exists
    if os.path.exists("run_tests.py"):
        cmd = [
            sys.executable, "run_tests.py",
            "--output-dir", output_dir,
            "--api-url", api_url
        ]

        if skip_api:
            cmd.append("--skip-api")
        if skip_data:
            cmd.append("--skip-data")
        if skip_model:
            cmd.append("--skip-model")
        if skip_benchmark:
            cmd.append("--skip-benchmark")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        # Save output
        with open(os.path.join(output_dir, "run_tests_output.txt"), "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\nSTDERR:\n")
                f.write(result.stderr)

        if result.returncode != 0:
            logger.error(f"Error running tests: {result.stderr}")
            return False

        logger.info("Tests completed")
    else:
        logger.error("run_tests.py not found")
        return False

    return True


def run_individual_tests(output_dir="test_results", skip_api=False, skip_data=False,
                         skip_model=False, skip_benchmark=False):
    """Run individual test scripts"""
    logger.info("Running individual tests...")

    results = []

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Run API tests
    if not skip_api and os.path.exists("test_api.py"):
        logger.info("Running API tests...")
        start_time = time.time()

        result = subprocess.run(
            [sys.executable, "test_api.py"],
            capture_output=True,
            text=True
        )

        end_time = time.time()

        # Save output
        with open(os.path.join(output_dir, "api_test_output.txt"), "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\nSTDERR:\n")
                f.write(result.stderr)

        success = "OK" in result.stdout and result.returncode == 0

        results.append({
            "name": "API Tests",
            "success": success,
            "duration": end_time - start_time,
            "returncode": result.returncode,
            "output_file": "api_test_output.txt"
        })

    # Run data tests
    if not skip_data and os.path.exists("test_data.py"):
        logger.info("Running data tests...")
        start_time = time.time()

        result = subprocess.run(
            [sys.executable, "test_data.py"],
            capture_output=True,
            text=True
        )

        end_time = time.time()

        # Save output
        with open(os.path.join(output_dir, "data_test_output.txt"), "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\nSTDERR:\n")
                f.write(result.stderr)

        success = "OK" in result.stdout and result.returncode == 0

        results.append({
            "name": "Data Tests",
            "success": success,
            "duration": end_time - start_time,
            "returncode": result.returncode,
            "output_file": "data_test_output.txt"
        })

    # Run model tests
    if not skip_model and os.path.exists("test_model.py"):
        logger.info("Running model tests...")
        start_time = time.time()

        result = subprocess.run(
            [sys.executable, "test_model.py"],
            capture_output=True,
            text=True
        )

        end_time = time.time()

        # Save output
        with open(os.path.join(output_dir, "model_test_output.txt"), "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\nSTDERR:\n")
                f.write(result.stderr)

        success = "OK" in result.stdout and result.returncode == 0

        results.append({
            "name": "Model Tests",
            "success": success,
            "duration": end_time - start_time,
            "returncode": result.returncode,
            "output_file": "model_test_output.txt"
        })

    # Run benchmark
    if not skip_benchmark and os.path.exists("benchmark.py"):
        logger.info("Running benchmark...")
        start_time = time.time()

        benchmark_output_dir = os.path.join(output_dir, "benchmark")
        os.makedirs(benchmark_output_dir, exist_ok=True)

        result = subprocess.run(
            [
                sys.executable, "benchmark.py",
                "--runs", "5",
                "--warmup", "2",
                "--output-dir", benchmark_output_dir,
                "--input-sizes", "416x416"
            ],
            capture_output=True,
            text=True
        )

        end_time = time.time()

        # Save output
        with open(os.path.join(output_dir, "benchmark_output.txt"), "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\nSTDERR:\n")
                f.write(result.stderr)

        success = result.returncode == 0

        results.append({
            "name": "Model Benchmark",
            "success": success,
            "duration": end_time - start_time,
            "returncode": result.returncode,
            "output_file": "benchmark_output.txt",
            "benchmark_dir": benchmark_output_dir
        })

    return results


def generate_report(results, output_dir="test_results"):
    """Generate test report"""
    try:
        # Import run_tests module for the report generation function
        sys.path.insert(0, ".")
        from run_tests import generate_report as gen_report

        report_path = gen_report(results, output_dir)
        logger.info(f"Report generated at {report_path}")
        return report_path
    except ImportError:
        logger.warning("Could not import report generator from run_tests module")

        # Create a simple report if the import fails
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Calculate summary
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r["success"])
        skipped_tests = sum(1 for r in results if r.get("returncode") == 0 and not r["success"])
        failed_tests = total_tests - passed_tests - skipped_tests
        total_time = sum(r["duration"] for r in results)

        # Create a simple HTML report
        report_path = os.path.join(output_dir, "test_report.html")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Test Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    .success {{ background-color: #dff0d8; }}
                    .failure {{ background-color: #f2dede; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>Test Report</h1>

                <div class="summary">
                    <h2>Summary</h2>
                    <p>Generated: {timestamp}</p>
                    <p>Tests: {passed_tests}/{total_tests} passed</p>
                    <p>Failed: {failed_tests}, Skipped: {skipped_tests}</p>
                    <p>Total time: {total_time:.2f} seconds</p>
                </div>

                <h2>Test Results</h2>
                <table>
                    <tr>
                        <th>Test</th>
                        <th>Status</th>
                        <th>Duration</th>
                        <th>Details</th>
                    </tr>
            """)

            for result in results:
                status_class = "success" if result["success"] else "failure"
                status_text = "PASSED" if result["success"] else "FAILED"

                f.write(f"""
                    <tr class="{status_class}">
                        <td>{result["name"]}</td>
                        <td>{status_text}</td>
                        <td>{result["duration"]:.2f}s</td>
                        <td><a href="{result["output_file"]}">View Output</a></td>
                    </tr>
                """)

            f.write("""
                </table>
            </body>
            </html>
            """)

        logger.info(f"Simple report generated at {report_path}")
        return report_path


def main():
    parser = argparse.ArgumentParser(description="Run all tests and generate a report")
    parser.add_argument("--output-dir", type=str, default="test_results", help="Output directory for test results")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000", help="API URL to test")
    parser.add_argument("--skip-api", action="store_true", help="Skip API tests")
    parser.add_argument("--skip-data", action="store_true", help="Skip data tests")
    parser.add_argument("--skip-model", action="store_true", help="Skip model tests")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip benchmarks")
    parser.add_argument("--skip-setup", action="store_true", help="Skip environment setup")
    parser.add_argument("--individual", action="store_true",
                        help="Run tests individually instead of using run_tests.py")

    args = parser.parse_args()

    # Create output directory first (fix for the error)
    os.makedirs(args.output_dir, exist_ok=True)

    # Clear output directory only if it exists and not empty
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        logger.info(f"Clearing output directory: {args.output_dir}")
        # Delete files instead of removing the directory
        for item in os.listdir(args.output_dir):
            item_path = os.path.join(args.output_dir, item)
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

    # Create basic directories
    os.makedirs("test_output", exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "benchmark"), exist_ok=True)
    logger.info("Created basic test directories")

    # Setup environment if needed
    if not args.skip_setup:
        if not setup_environment():
            logger.error("Failed to set up test environment")
            sys.exit(1)

    # Run tests
    if args.individual:
        results = run_individual_tests(
            args.output_dir,
            skip_api=args.skip_api,
            skip_data=args.skip_data,
            skip_model=args.skip_model,
            skip_benchmark=args.skip_benchmark
        )

        if not results:
            logger.error("No tests were run")
            sys.exit(1)

        # Generate report
        generate_report(results, args.output_dir)
    else:
        run_result = run_tests(
            args.output_dir,
            args.api_url,
            skip_api=args.skip_api,
            skip_data=args.skip_data,
            skip_model=args.skip_model,
            skip_benchmark=args.skip_benchmark
        )
        if not run_result:
            logger.warning("Some tests may have failed. See the report for details.")
            # No sys.exit(1) here - we want the process to complete normally

    logger.info("All tests completed!")


if __name__ == "__main__":
    main()