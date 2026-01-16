# Mini MLflow

A minimal ML experiment tracker - simple, file-based, no dependencies on databases or servers.

## Features

- **Save parameters** (config)
- **Save metrics** (results)
- **Organize by experiment run**
- **File-based persistence** - no database required
- **Dual API support** - MLflow-like API and context manager style

## Installation

```bash
pip install mini-mlflow
```

## Quick Start

### MLflow-like API

```python
from mini_mlflow import start_run, log_param, log_metric

run = start_run(run_name="my_experiment")
log_param("learning_rate", 0.01)
log_metric("accuracy", 0.95)
run.end()
```

### Context Manager API

```python
from mini_mlflow import ExperimentTracker

with ExperimentTracker(run_name="my_experiment") as run:
    run.log_param("learning_rate", 0.01)
    run.log_metric("accuracy", 0.95)
```

## Documentation

Full documentation coming soon.

## License

MIT License - see LICENSE file for details.
