# Mini MLflow

A minimal ML experiment tracker - simple, file-based, no dependencies on databases or servers.

## Overview

Mini MLflow is a lightweight experiment tracking library for machine learning projects. It provides a simple way to log parameters, metrics, and organize experiments without the complexity of a full MLflow setup.

**Key Features:**
- Save parameters (config)
- Save metrics (results)
- Organize by experiment run
- File-based persistence - no database required
- Dual API support - MLflow-like API and context manager style
- Thread-safe active run tracking

## Installation

```bash
pip install mini-mlflow
```

### Requirements

- Python >= 3.7
- PyYAML (automatically installed)

## Quick Start

### MLflow-like API

```python
from mini_mlflow import start_run, log_param, log_metric, end_run

# Start a new run
run = start_run(run_name="my_experiment")
log_param("learning_rate", 0.01)
log_param("batch_size", 32)
log_metric("accuracy", 0.95)
log_metric("loss", 0.05)
end_run()
```

### Context Manager API

```python
from mini_mlflow import ExperimentTracker

with ExperimentTracker(run_name="my_experiment") as run:
    run.log_param("learning_rate", 0.01)
    run.log_param("batch_size", 32)
    run.log_metric("accuracy", 0.95)
    run.log_metric("loss", 0.05)
# Run automatically ends when exiting context
```

## Usage Examples

### Example 1: Basic Experiment Tracking

```python
from mini_mlflow import ExperimentTracker

with ExperimentTracker(run_name="neural_network_training") as run:
    # Log hyperparameters
    run.log_param("learning_rate", 0.001)
    run.log_param("batch_size", 64)
    run.log_param("epochs", 50)
    
    # Simulate training loop
    for epoch in range(50):
        # ... training code ...
        accuracy = train_one_epoch()
        run.log_metric("accuracy", accuracy)
```

### Example 2: Multiple Runs Comparison

```python
from mini_mlflow import ExperimentTracker

tracker = ExperimentTracker(experiment_id=0)

# Try different learning rates
for lr in [0.001, 0.01, 0.1]:
    with tracker as run:
        run.log_param("learning_rate", lr)
        # ... train model ...
        run.log_metric("final_accuracy", train_and_evaluate(lr))
```

### Example 3: Retrieving Past Runs

```python
from mini_mlflow import ExperimentTracker

tracker = ExperimentTracker(experiment_id=0)

# List all runs
runs = tracker.list_runs()
print(f"Total runs: {len(runs)}")

# Get specific run data
if runs:
    run_data = tracker.get_run(runs[0])
    print(f"Parameters: {run_data['params']}")
    print(f"Metrics: {run_data['metrics']}")
```

## API Reference

### MLflow-like API (Global Functions)

#### `start_run(run_name=None, experiment_id=0, run_id=None)`

Start a new experiment run.

**Parameters:**
- `run_name` (str, optional): Name for the run
- `experiment_id` (int): Experiment ID (default: 0)
- `run_id` (str, optional): Custom run ID. If None, a UUID is generated.

**Returns:**
- `Run`: The created Run object

**Example:**
```python
run = start_run(run_name="my_experiment")
```

#### `active_run()`

Get the currently active run for this thread.

**Returns:**
- `Run` or `None`: The active Run object, or None if no run is active

**Example:**
```python
run = active_run()
if run:
    print(f"Active run: {run.run_id}")
```

#### `end_run()`

End the currently active run.

**Raises:**
- `RuntimeError`: If there is no active run

**Example:**
```python
end_run()
```

#### `log_param(key, value)`

Log a parameter to the active run.

**Parameters:**
- `key` (str): Parameter name (must be a valid filename)
- `value`: Parameter value (must be JSON-serializable)

**Raises:**
- `RuntimeError`: If there is no active run
- `ValueError`: If the key is invalid

**Example:**
```python
log_param("learning_rate", 0.01)
```

#### `log_metric(key, value)`

Log a metric to the active run.

**Parameters:**
- `key` (str): Metric name (must be a valid filename)
- `value` (float): Metric value

**Raises:**
- `RuntimeError`: If there is no active run
- `ValueError`: If the key is invalid

**Example:**
```python
log_metric("accuracy", 0.95)
```

### Context Manager API

#### `ExperimentTracker(run_name=None, experiment_id=0, runs_dir="mlruns")`

Create an experiment tracker that can be used as a context manager.

**Parameters:**
- `run_name` (str, optional): Default name for runs started with this tracker
- `experiment_id` (int): Experiment ID (default: 0)
- `runs_dir` (str): Base directory for storing runs (default: "mlruns")

**Example:**
```python
with ExperimentTracker(run_name="my_experiment") as run:
    run.log_param("learning_rate", 0.01)
    run.log_metric("accuracy", 0.95)
```

### Run Class

#### `Run(run_id=None, experiment_id=0, name=None, runs_dir="mlruns")`

Represents a single experiment run.

**Parameters:**
- `run_id` (str, optional): Unique identifier. If None, a UUID is generated.
- `experiment_id` (int): Experiment ID (default: 0)
- `name` (str, optional): Name for the run
- `runs_dir` (str): Base directory for storing runs (default: "mlruns")

**Methods:**

- `log_param(key, value)`: Log a parameter
- `log_metric(key, value)`: Log a metric
- `end(status="FINISHED")`: End the run (status: "FINISHED" or "FAILED")

**Attributes:**

- `run_id`: Unique run identifier
- `experiment_id`: Experiment ID
- `name`: Run name
- `status`: Run status ("RUNNING", "FINISHED", "FAILED")
- `start_time`: ISO timestamp when run started
- `end_time`: ISO timestamp when run ended (None if still running)

**Example:**
```python
from mini_mlflow import Run

run = Run(name="my_run")
run.log_param("learning_rate", 0.01)
run.log_metric("accuracy", 0.95)
run.end()
```

### ExperimentTracker Methods

#### `start_run(run_name=None, experiment_id=None, run_id=None)`

Start a new run with this tracker.

**Returns:**
- `Run`: The created Run object

#### `active_run()`

Get the currently active run.

**Returns:**
- `Run` or `None`: The active Run object

#### `end_run()`

End the currently active run.

#### `list_runs(experiment_id=None)`

List all run IDs in an experiment.

**Parameters:**
- `experiment_id` (int, optional): Experiment ID. If None, uses tracker's experiment_id.

**Returns:**
- `list`: List of run IDs (sorted newest first)

#### `get_run(run_id, experiment_id=None)`

Retrieve data for a specific run.

**Parameters:**
- `run_id` (str): The run ID to retrieve
- `experiment_id` (int, optional): Experiment ID. If None, uses tracker's experiment_id.

**Returns:**
- `dict`: Dictionary containing:
  - `metadata`: Run metadata (name, status, timestamps, etc.)
  - `params`: Dictionary of parameters
  - `metrics`: Dictionary of metrics

## File Structure

Experiments are stored in the `mlruns/` directory (MLflow-compatible format):

```
mlruns/
└── 0/                      # Experiment ID
    └── <run_id>/           # UUID-based run ID
        ├── meta.yaml       # Run metadata
        ├── params/         # Parameters directory
        │   └── <param_name>.json
        └── metrics/        # Metrics directory
            └── <metric_name>.json
```

## Thread Safety

Mini MLflow uses thread-local storage for active run tracking. This means:
- Each thread has its own active run
- Multiple threads can run experiments concurrently without conflicts
- Global API functions (`log_param`, `log_metric`) work correctly in multi-threaded environments

## Error Handling

The library provides clear error messages for common issues:
- Attempting to log without an active run
- Starting multiple runs without ending the previous one
- Logging to a run that has already ended
- Invalid parameter/metric keys (e.g., containing slashes)

## Examples

See `examples/basic_usage.py` for complete working examples demonstrating:
1. MLflow-like API usage
2. Context manager API usage
3. Multiple runs in one experiment
4. Retrieving past runs

Run the examples:
```bash
python examples/basic_usage.py
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Version

Current version: 0.1.0
