from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.conda_dependencies import CondaDependencies
import os

# Set your Azure workspace details
workspace_name = os.environ.get("WORKSPACE_NAME")
subscription_id = os.environ.get("SUBSCRIPTION_ID")
resource_group = os.environ.get("RESOURCE_GROUP")

# Authenticate and get workspace
try:
    ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=InteractiveLoginAuthentication()
    )
    print("Found workspace:", ws.name)
except Exception as e:
    print("Could not find workspace")
    print("Error:", str(e))

# Create (or find) compute target
compute_name = "cpu-cluster"
try:
    compute_target = ComputeTarget(workspace=ws, name=compute_name)
    print('Found existing compute target')
except ComputeTargetException:
    print('Creating new compute target...')
    compute_config = AmlCompute.provisioning_configuration(
        vm_size='STANDARD_DS11_V2', 
        min_nodes=0,
        max_nodes=4
    )
    compute_target = ComputeTarget.create(ws, compute_name, compute_config)
    compute_target.wait_for_completion(show_output=True)

# Create an experiment
experiment_name = 'xgboost-co2-prediction'
experiment = Experiment(workspace=ws, name=experiment_name)
print("Created experiment:", experiment_name)

# Define environment with required packages
conda_deps = CondaDependencies()
conda_deps.add_pip_package('pandas')
conda_deps.add_pip_package('numpy')
conda_deps.add_pip_package('scikit-learn')
conda_deps.add_pip_package('xgboost')
conda_deps.add_pip_package('optuna')
conda_deps.add_pip_package('lightgbm')
conda_deps.add_pip_package('catboost')

env = Environment('xgboost_env')
env.python.conda_dependencies = conda_deps

# Create script config
src = ScriptRunConfig(
    source_directory='.',  # Directory containing the training script
    script='train1.py',     # Training script
    compute_target=compute_target,
    environment=env
)

# Submit the experiment
print("Submitting experiment...")
run = experiment.submit(src)
print("Experiment submitted. Run ID:", run.id)

# Wait for the run to complete and show output
print("Waiting for run completion...")
run.wait_for_completion(show_output=True)

# Get the metrics and logs
print("\nRun completed!")
print("Metrics:", run.get_metrics())
try:
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    run.download_files(prefix="outputs/", output_directory=output_dir)
    print(f"Downloaded output files to {output_dir}")
except Exception as e:
    print("Error downloading files:", str(e))

print("Setup and run completed successfully!")