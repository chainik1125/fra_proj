"""View MLflow metrics from the command line, bypassing UI issues with inf values."""

import mlflow
import pandas as pd

mlflow.set_tracking_uri("file:./mlruns")

# Get all experiments and runs
experiments = mlflow.search_experiments()
print(f"Found {len(experiments)} experiments")

# Get all runs across all experiments
experiment_ids = [exp.experiment_id for exp in experiments]
runs = mlflow.search_runs(experiment_ids=experiment_ids) if experiment_ids else pd.DataFrame()

if runs.empty:
    print("No runs found.")
else:
    print("=" * 60)
    print("RUNS")
    print("=" * 60)
    for i, run in runs.iterrows():
        print(f"\nRun: {run['run_id'][:8]}...")
        print(f"  Name: {run.get('tags.mlflow.runName', 'N/A')}")
        print(f"  Status: {run['status']}")
        print(f"  Start: {run['start_time']}")

    print("\n" + "=" * 60)
    print("METRICS (latest run)")
    print("=" * 60)

    # Get the most recent run
    latest_run_id = runs.iloc[0]['run_id']
    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(latest_run_id)

    # Print metrics
    metrics = run_data.data.metrics
    for name, value in sorted(metrics.items()):
        print(f"  {name}: {value}")

    print("\n" + "=" * 60)
    print("METRIC HISTORY (latest run)")
    print("=" * 60)

    # Get history for each metric
    for metric_name in sorted(metrics.keys()):
        history = client.get_metric_history(latest_run_id, metric_name)
        print(f"\n  {metric_name}:")
        for h in history[-5:]:  # Last 5 values
            print(f"    step {h.step}: {h.value}")
        if len(history) > 5:
            print(f"    ... ({len(history)} total values)")

    print("\n" + "=" * 60)
    print("ARTIFACTS")
    print("=" * 60)
    artifacts = client.list_artifacts(latest_run_id)
    for artifact in artifacts:
        print(f"  {artifact.path}")
