import streamlit as st

# Define constants
SAMPLE_EXTRACTION_COST = 0.02
CROMWELL_HOURLY = 0.375
CROMWELL_ADDITIONAL_HOURLY = 0.22
WORKSPACE_COST = 0.20
STORAGE_COST_PER_GB = 0.026
DATAPROC_PRICE_PER_VCPU_HOUR = 0.01

# Define VM configurations and costs
JUPYTER_VM_CONFIGS = {
    "1 CPU, 3.75GB RAM": {"cpu": 1, "ram": 3.75, "running": 0.06, "paused": 0.01},
    "2 CPU, 7.5GB RAM": {"cpu": 2, "ram": 7.5, "running": 0.11, "paused": 0.01},
    "2 CPU, 13GB RAM": {"cpu": 2, "ram": 13, "running": 0.13, "paused": 0.01},
    "4 CPU, 3.6GB RAM": {"cpu": 4, "ram": 3.6, "running": 0.15, "paused": 0.01},
    "4 CPU, 15GB RAM": {"cpu": 4, "ram": 15, "running": 0.20, "paused": 0.01},
    "4 CPU, 26GB RAM": {"cpu": 4, "ram": 26, "running": 0.25, "paused": 0.01},
    "8 CPU, 7.2GB RAM": {"cpu": 8, "ram": 7.2, "running": 0.29, "paused": 0.01},
    "8 CPU, 30GB RAM": {"cpu": 8, "ram": 30, "running": 0.39, "paused": 0.01},
    "8 CPU, 52GB RAM": {"cpu": 8, "ram": 52, "running": 0.48, "paused": 0.01},
    "16 CPU, 14.4GB RAM": {"cpu": 16, "ram": 14.4, "running": 0.58, "paused": 0.01},
    "16 CPU, 60GB RAM": {"cpu": 16, "ram": 60, "running": 0.77, "paused": 0.01},
    "16 CPU, 104GB RAM": {"cpu": 16, "ram": 104, "running": 0.96, "paused": 0.01}
}

# Define GPU configurations and costs
JUPYTER_GPU_CONFIGS = {
    "No GPU": {"price": 0, "memory": 0, "type": "None"},
    "NVIDIA T4 - 1 GPU": {"price": 0.35, "memory": 16, "type": "GDDR6"},
    "NVIDIA T4 - 2 GPUs": {"price": 0.70, "memory": 32, "type": "GDDR6"},
    "NVIDIA T4 - 4 GPUs": {"price": 1.40, "memory": 64, "type": "GDDR6"},
    "NVIDIA P4 - 1 GPU": {"price": 0.60, "memory": 8, "type": "GDDR5"},
    "NVIDIA P4 - 2 GPUs": {"price": 1.20, "memory": 16, "type": "GDDR5"},
    "NVIDIA P4 - 4 GPUs": {"price": 2.40, "memory": 32, "type": "GDDR5"},
    "NVIDIA V100 - 1 GPU": {"price": 2.48, "memory": 16, "type": "HBM2"},
    "NVIDIA V100 - 2 GPUs": {"price": 4.96, "memory": 32, "type": "HBM2"},
    "NVIDIA V100 - 4 GPUs": {"price": 9.92, "memory": 64, "type": "HBM2"},
    "NVIDIA V100 - 8 GPUs": {"price": 19.84, "memory": 128, "type": "HBM2"},
    "NVIDIA P100 - 1 GPU": {"price": 1.46, "memory": 16, "type": "HBM2"},
    "NVIDIA P100 - 2 GPUs": {"price": 2.92, "memory": 32, "type": "HBM2"},
    "NVIDIA P100 - 4 GPUs": {"price": 5.84, "memory": 64, "type": "HBM2"},
    "NVIDIA T4 Virtual Workstation - 1 GPU": {"price": 0.55, "memory": 16, "type": "GDDR6"},
    "NVIDIA T4 Virtual Workstation - 2 GPUs": {"price": 1.10, "memory": 32, "type": "GDDR6"},
    "NVIDIA T4 Virtual Workstation - 4 GPUs": {"price": 2.20, "memory": 64, "type": "GDDR6"},
    "NVIDIA P4 Virtual Workstation - 1 GPU": {"price": 0.80, "memory": 8, "type": "GDDR5"},
    "NVIDIA P4 Virtual Workstation - 2 GPUs": {"price": 1.60, "memory": 16, "type": "GDDR5"},
    "NVIDIA P4 Virtual Workstation - 4 GPUs": {"price": 3.20, "memory": 32, "type": "GDDR5"},
    "NVIDIA P100 Virtual Workstation - 1 GPU": {"price": 1.66, "memory": 16, "type": "HBM2"},
    "NVIDIA P100 Virtual Workstation - 2 GPUs": {"price": 3.32, "memory": 32, "type": "HBM2"},
    "NVIDIA P100 Virtual Workstation - 4 GPUs": {"price": 6.64, "memory": 64, "type": "HBM2"}
}

# Dataproc configurations
DATAPROC_MACHINE_TYPES = {
    "n1-standard-2": {"vcpu": 2, "memory": 7.5},
    "n1-standard-4": {"vcpu": 4, "memory": 15},
    "n1-standard-8": {"vcpu": 8, "memory": 30},
    "n1-standard-16": {"vcpu": 16, "memory": 60},
    "n1-standard-32": {"vcpu": 32, "memory": 120},
    "n1-standard-64": {"vcpu": 64, "memory": 240},
}

DATAPROC_CLUSTER_CONFIGS = {
    "Small Cluster": {
        "master": {"type": "n1-standard-4", "count": 1, "disk": 500},
        "worker": {"type": "n1-standard-4", "count": 2, "disk": 500},
    },
    "Medium Cluster": {
        "master": {"type": "n1-standard-4", "count": 1, "disk": 500},
        "worker": {"type": "n1-standard-4", "count": 5, "disk": 500},
    },
    "Large Cluster": {
        "master": {"type": "n1-standard-8", "count": 1, "disk": 1000},
        "worker": {"type": "n1-standard-8", "count": 10, "disk": 1000},
    },
    "Custom": None
}

APP_COSTS = {
    "Jupyter Notebooks": {
        "running": 0.20,
        "paused": 0.01,
        "storage": 4.80
    },
    "RStudio": {
        "running": 0.40,
        "paused": 0.21,
        "storage": 4.00
    },
    "SAS": {
        "running": 0.40,
        "paused": 0.21,
        "storage": 10.00
    }
}

def calculate_dataproc_costs(cluster_config, hours, custom_config=None):
    if cluster_config == "Custom" and custom_config:
        config = custom_config
    else:
        config = DATAPROC_CLUSTER_CONFIGS[cluster_config]
    
    total_vcpus = 0
    
    # Calculate master nodes VCPUs
    master_vcpus = (DATAPROC_MACHINE_TYPES[config["master"]["type"]]["vcpu"] * 
                   config["master"]["count"])
    
    # Calculate worker nodes VCPUs
    worker_vcpus = (DATAPROC_MACHINE_TYPES[config["worker"]["type"]]["vcpu"] * 
                    config["worker"]["count"])
    
    total_vcpus = master_vcpus + worker_vcpus
    
    # Calculate Dataproc cost
    dataproc_cost = total_vcpus * hours * DATAPROC_PRICE_PER_VCPU_HOUR
    
    return {
        "total_vcpus": total_vcpus,
        "dataproc_cost": dataproc_cost,
        "master_nodes": config["master"]["count"],
        "worker_nodes": config["worker"]["count"],
        "master_type": config["master"]["type"],
        "worker_type": config["worker"]["type"]
    }

def calculate_costs(apps, analysis_type, num_samples, storage, analysis_hours, paused_hours, 
                   workspaces, cromwell_user_type, use_cromwell, cromwell_apps, 
                   jupyter_config, jupyter_gpu_config, delete_environment, project_duration,
                   use_dataproc=False, dataproc_cluster_config=None, dataproc_custom_config=None):
    # Initialize costs
    running_costs = 0
    paused_costs = 0
    app_storage_costs = 0
    gpu_costs = 0
    dataproc_costs = 0
    dataproc_details = None
    initial_costs = 0
    cromwell_costs = 0

    # Calculate costs for each app
    for app in apps:
        if app == "Jupyter Notebooks":
            # VM costs
            running_costs += JUPYTER_VM_CONFIGS[jupyter_config]["running"] * analysis_hours
            paused_costs += JUPYTER_VM_CONFIGS[jupyter_config]["paused"] * paused_hours
            # Only add storage cost if not deleting environment
            if not delete_environment:
                app_storage_costs += APP_COSTS[app]["storage"]
            
            # GPU costs (only applied during running hours)
            gpu_costs += JUPYTER_GPU_CONFIGS[jupyter_gpu_config]["price"] * analysis_hours
            
            # Dataproc costs if enabled
            if use_dataproc:
                dataproc_details = calculate_dataproc_costs(
                    dataproc_cluster_config, 
                    analysis_hours, 
                    dataproc_custom_config
                )
                dataproc_costs = dataproc_details["dataproc_cost"]
        else:
            running_costs += APP_COSTS[app]["running"] * analysis_hours
            paused_costs += APP_COSTS[app]["paused"] * paused_hours
            if not delete_environment:
                app_storage_costs += APP_COSTS[app]["storage"]

    bucket_storage_costs = storage * STORAGE_COST_PER_GB
    workspace_costs = workspaces * WORKSPACE_COST

    # Calculate WGS costs
    if analysis_type == "WGS":
        initial_costs = num_samples * SAMPLE_EXTRACTION_COST

    # Calculate Cromwell costs when used with Jupyter
    if use_cromwell and "Jupyter Notebooks" in apps:
        cromwell_costs = CROMWELL_HOURLY * analysis_hours
        if cromwell_apps > 1:
            cromwell_additional = CROMWELL_ADDITIONAL_HOURLY * analysis_hours * (cromwell_apps - 1)
            cromwell_costs += cromwell_additional

    # Calculate storage costs for project duration
    monthly_storage_costs = (app_storage_costs + bucket_storage_costs + workspace_costs)
    total_storage_costs = monthly_storage_costs * project_duration

    # Calculate total project costs
    total_costs = (initial_costs + running_costs + paused_costs + 
                  total_storage_costs + gpu_costs + dataproc_costs + cromwell_costs)

    return {
        "initial_costs": initial_costs,
        "running_costs": running_costs,
        "paused_costs": paused_costs,
        "storage_costs": app_storage_costs,
        "bucket_storage_costs": bucket_storage_costs,
        "workspace_costs": workspace_costs,
        "cromwell_costs": cromwell_costs,
        "gpu_costs": gpu_costs,
        "dataproc_costs": dataproc_costs,
        "dataproc_details": dataproc_details,
        "total_storage_costs": total_storage_costs,
        "total_costs": total_costs
    }

def main():
    st.title("All of Us Cloud Computing Cost Calculator")

    # Sidebar inputs
    st.sidebar.header("Input Parameters")
    
    apps = st.sidebar.multiselect(
        "Select Applications Used",
        ["Jupyter Notebooks", "RStudio", "SAS"],
        default=["Jupyter Notebooks"]
    )

    delete_environment = st.sidebar.checkbox(
        "Delete persistent disk environment when not in use?",
        help="Deleting your environment when not in use saves on monthly persistent disk costs"
    )

    # Jupyter, GPU, and Dataproc configuration
    jupyter_config = "4 CPU, 15GB RAM"
    jupyter_gpu_config = "No GPU"
    use_dataproc = False
    dataproc_cluster_config = None
    dataproc_custom_config = None

    if "Jupyter Notebooks" in apps:
        st.sidebar.subheader("Jupyter Configuration")
        jupyter_config = st.sidebar.selectbox(
            "VM Configuration",
            list(JUPYTER_VM_CONFIGS.keys()),
            index=4,
            help="Select CPU and RAM configuration for Jupyter Notebook"
        )
        
        jupyter_gpu_config = st.sidebar.selectbox(
            "GPU Configuration",
            list(JUPYTER_GPU_CONFIGS.keys()),
            index=0,
            help="Select GPU configuration for Jupyter Notebook"
        )

        # Dataproc configuration
        use_dataproc = st.sidebar.checkbox("Use Dataproc?", False)
        if use_dataproc:
            dataproc_cluster_config = st.sidebar.selectbox(
                "Dataproc Cluster Configuration",
                list(DATAPROC_CLUSTER_CONFIGS.keys()),
                help="Select predefined cluster configuration or custom"
            )
            
            if dataproc_cluster_config == "Custom":
                st.sidebar.subheader("Custom Cluster Configuration")
                dataproc_custom_config = {
                    "master": {
                        "type": st.sidebar.selectbox(
                            "Master Node Machine Type",
                            list(DATAPROC_MACHINE_TYPES.keys()),
                            index=1
                        ),
                        "count": st.sidebar.number_input(
                            "Number of Master Nodes",
                            min_value=1,
                            value=1
                        ),
                        "disk": st.sidebar.number_input(
                            "Master Node Disk Size (GB)",
                            min_value=100,
                            value=500,
                            step=100
                        )
                    },
                    "worker": {
                        "type": st.sidebar.selectbox(
                            "Worker Node Machine Type",
                            list(DATAPROC_MACHINE_TYPES.keys()),
                            index=1
                        ),
                        "count": st.sidebar.number_input(
                            "Number of Worker Nodes",
                            min_value=2,
                            value=2
                        ),
                        "disk": st.sidebar.number_input(
                            "Worker Node Disk Size (GB)",
                            min_value=100,
                            value=500,
                            step=100
                        )
                    }
                }

    # Cromwell options
    use_cromwell = False
    cromwell_apps = 1
    cromwell_user_type = "First User"
    if "Jupyter Notebooks" in apps:
        use_cromwell = st.sidebar.checkbox("Use Cromwell with Jupyter?", False)
        if use_cromwell:
            cromwell_apps = st.sidebar.number_input(
                "Number of Cromwell Applications",
                min_value=1,
                value=1,
                help="First app: $0.375/hr, Additional apps: $0.22/hr each"
            )
            cromwell_user_type = st.sidebar.selectbox(
                "Cromwell User Type",
                ["First User", "Second User"],
                help="First User: $0.375/hr, Second User: $0.22/hr"
            )

    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Other", "WGS"]
    )

    num_samples = 0
    if analysis_type == "WGS":
        num_samples = st.sidebar.number_input(
            "Number of WGS Samples",
            min_value=1,
            value=6000
        )

    storage = st.sidebar.number_input(
        "Storage Required (GB)",
        min_value=0,
        value=100
    )

    analysis_hours = st.sidebar.number_input(
        "Active Analysis Hours for Project",  # Changed from "per Month"
        min_value=0,
        value=40,
        help="Total number of hours you plan to actively use the environment during your project"
    )

    paused_hours = st.sidebar.number_input(
        "Paused Analysis Hours for Project",  # Changed from "per Month"
        min_value=0,
        value=128,
        help="Total number of hours your environment will be paused during your project"
    )

    workspaces = st.sidebar.number_input(
        "Number of Workspaces",
        min_value=1,
        value=1
    )

    project_duration = st.sidebar.number_input(
        "Project Duration (months)",
        min_value=1,
        value=3,
        help="Total number of months your project will run"
    )

    # Calculate costs
    cost_details = calculate_costs(
        apps, analysis_type, num_samples, storage, analysis_hours, 
        paused_hours, workspaces, cromwell_user_type, use_cromwell, 
        cromwell_apps, jupyter_config, jupyter_gpu_config, delete_environment,
        project_duration,  # Add this parameter
        use_dataproc, dataproc_cluster_config, dataproc_custom_config
    )

    # Display results
    st.header("Cost Breakdown")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("Upfront Costs")
        st.write(f"${cost_details['initial_costs']:.2f}")
        st.write("Data extraction fees")

    with col2:
        st.subheader("Total Compute Costs")
        # Calculate total costs for all running hours
        total_running_costs = 0
        # VM and application running costs
        for app in apps:
            if app == "Jupyter Notebooks":
                # VM costs
                total_running_costs += JUPYTER_VM_CONFIGS[jupyter_config]["running"] * analysis_hours
                # GPU costs if applicable
                if jupyter_gpu_config != "No GPU":
                    total_running_costs += JUPYTER_GPU_CONFIGS[jupyter_gpu_config]["price"] * analysis_hours
                # Dataproc costs if applicable
                if use_dataproc and cost_details['dataproc_costs']:
                    total_running_costs += cost_details['dataproc_costs']
            else:
                total_running_costs += APP_COSTS[app]["running"] * analysis_hours

        # Paused costs
        total_paused_costs = sum(APP_COSTS[app]["paused"] * paused_hours for app in apps)
        total_running_costs += total_paused_costs

        # Add Cromwell costs if applicable
        if use_cromwell and "Jupyter Notebooks" in apps:
            total_running_costs += CROMWELL_HOURLY * analysis_hours
            if cromwell_apps > 1:
                total_running_costs += (CROMWELL_ADDITIONAL_HOURLY * analysis_hours * (cromwell_apps - 1))
        
        st.write(f"${total_running_costs:.2f}")
        st.write("`(Active Hours × 0.40) + (Paused Hours × 0.01)`")

    with col3:
        st.subheader("Storage Costs")
        monthly_storage_cost = 0
        # Persistent disk storage
        if not delete_environment:
            for app in apps:
                monthly_storage_cost += APP_COSTS[app]["storage"]
        # Bucket storage
        monthly_storage_cost += storage * STORAGE_COST_PER_GB
        # Workspace costs
        monthly_storage_cost += workspaces * WORKSPACE_COST
        
        # Calculate total storage cost for the project duration
        total_storage_cost = monthly_storage_cost * project_duration
        
        st.write(f"${total_storage_cost:.2f}")
        st.write(f"Storage costs for {project_duration} months")

    with col4:
        st.subheader("Total Project Cost")
        # Calculate total project cost
        total_project_cost = (cost_details['initial_costs'] + 
                            total_running_costs + 
                            total_storage_cost)
        
        st.write(f"${total_project_cost:.2f}")
        st.write("Total estimated cost")

    # Detailed breakdown
    with st.expander("View Detailed Cost Breakdown"):
        st.write("Hourly Running Costs (Direct Costs):")
        direct_hourly_cost = 0

        for app in apps:
            st.write(f"\n{app}:")
            if app == "Jupyter Notebooks":
                # VM running cost
                vm_cost = JUPYTER_VM_CONFIGS[jupyter_config]['running']
                st.write(f"- VM cost: ${vm_cost:.2f}/hr")
                direct_hourly_cost += vm_cost

                # GPU costs if applicable
                if jupyter_gpu_config != "No GPU":
                    gpu_cost = JUPYTER_GPU_CONFIGS[jupyter_gpu_config]['price']
                    st.write(f"- GPU cost: ${gpu_cost:.2f}/hr")
                    direct_hourly_cost += gpu_cost

                # Dataproc costs if applicable
                if use_dataproc and cost_details['dataproc_details']:
                    dataproc_hourly = cost_details['dataproc_costs'] / analysis_hours
                    st.write(f"- Dataproc cluster cost: ${dataproc_hourly:.2f}/hr")
                    direct_hourly_cost += dataproc_hourly

        if use_cromwell and "Jupyter Notebooks" in apps:
            st.write("\nCromwell costs:")
            cromwell_cost = CROMWELL_HOURLY
            st.write(f"- First application: ${cromwell_cost:.2f}/hr")
            direct_hourly_cost += cromwell_cost
            
            if cromwell_apps > 1:
                additional_cost = CROMWELL_ADDITIONAL_HOURLY * (cromwell_apps - 1)
                st.write(f"- Additional applications: ${additional_cost:.2f}/hr")
                direct_hourly_cost += additional_cost

        st.write(f"\nTotal Direct Hourly Costs: ${direct_hourly_cost:.2f}/hr")
        
        st.write("\nFixed Monthly Costs:")
        monthly_costs = 0

        # Storage costs
        if not delete_environment:
            for app in apps:
                st.write(f"- {app} persistent disk: ${APP_COSTS[app]['storage']:.2f}/month")
                monthly_costs += APP_COSTS[app]['storage']

        # Bucket storage
        bucket_cost = storage * STORAGE_COST_PER_GB
        st.write(f"- Bucket storage ({storage} GB): ${bucket_cost:.2f}/month")
        monthly_costs += bucket_cost

        # Workspace costs
        workspace_cost = workspaces * WORKSPACE_COST
        st.write(f"- Workspace cost: ${workspace_cost:.2f}/month")
        monthly_costs += workspace_cost

        st.write(f"\nTotal Monthly Fixed Costs: ${monthly_costs:.2f}/month")

        # Paused Costs
        if paused_hours > 0:
            st.write("\nPaused Instance Costs:")
            paused_costs = 0
            for app in apps:
                app_paused_cost = APP_COSTS[app]["paused"] * paused_hours
                st.write(f"- {app}: ${APP_COSTS[app]['paused']}/hr × {paused_hours} hours = ${app_paused_cost:.2f}")
                paused_costs += app_paused_cost
            st.write(f"\nTotal Paused Costs: ${paused_costs:.2f}")

        # Project Total Calculation
        st.write("\nTotal Project Cost Calculation:")
        if analysis_type == "WGS":
            st.write(f"One-time extraction cost: ${cost_details['initial_costs']:.2f}")
        
        active_compute = direct_hourly_cost * analysis_hours
        st.write(f"Active compute costs: ${direct_hourly_cost:.2f}/hr × {analysis_hours} hours = ${active_compute:.2f}")
        
        if paused_hours > 0:
            st.write(f"Paused costs: ${paused_costs:.2f}")
        
        total_storage = monthly_costs * project_duration
        st.write(f"Storage costs: ${monthly_costs:.2f}/month × {project_duration} months = ${total_storage:.2f}")
        
        total_project = (cost_details['initial_costs'] + active_compute + 
                        paused_costs + total_storage)
        st.write(f"\nTotal Project Cost: ${total_project:.2f}")

        st.markdown("---")
        st.write("Note: This breakdown shows:")
        st.write("- Direct hourly costs that apply while your instances are running")
        st.write("- Fixed monthly costs for storage and workspaces")
        st.write("- Costs during paused periods")
        st.write("- Total project cost based on your specified duration")

        if analysis_type == "WGS":
            st.write("WGS Analysis Costs:")
            st.write(f"- Data extraction (${SAMPLE_EXTRACTION_COST:.2f}/sample × {num_samples:,} samples): ${cost_details['initial_costs']:.2f}")

        st.write("\nApplication Costs:")
        for app in apps:
            st.write(f"\n{app}:")
            if app == "Jupyter Notebooks":
                st.write(f"- VM Configuration: {jupyter_config}")
                st.write(f"- Running costs ({analysis_hours} hrs @ ${JUPYTER_VM_CONFIGS[jupyter_config]['running']}/hr): ${JUPYTER_VM_CONFIGS[jupyter_config]['running'] * analysis_hours:.2f}")
                st.write(f"- Paused costs ({paused_hours} hrs @ ${JUPYTER_VM_CONFIGS[jupyter_config]['paused']}/hr): ${JUPYTER_VM_CONFIGS[jupyter_config]['paused'] * paused_hours:.2f}")
                
                if jupyter_gpu_config != "No GPU":
                    st.write(f"- GPU Configuration: {jupyter_gpu_config}")
                    st.write(f"- GPU costs ({analysis_hours} hrs @ ${JUPYTER_GPU_CONFIGS[jupyter_gpu_config]['price']}/hr): ${cost_details['gpu_costs']:.2f}")
                
                if use_dataproc and cost_details['dataproc_details']:
                    st.write("\nDataproc Costs:")
                    details = cost_details['dataproc_details']
                    st.write(f"- Cluster Configuration: {dataproc_cluster_config}")
                    st.write(f"- Total vCPUs: {details['total_vcpus']}")
                    st.write(f"- Master: {details['master_nodes']} × {details['master_type']}")
                    st.write(f"- Workers: {details['worker_nodes']} × {details['worker_type']}")
                    st.write(f"- Total Dataproc cost ({analysis_hours} hrs): ${cost_details['dataproc_costs']:.2f}")
            else:
                st.write(f"- Running costs ({analysis_hours} hrs @ ${APP_COSTS[app]['running']}/hr): ${APP_COSTS[app]['running'] * analysis_hours:.2f}")
                st.write(f"- Paused costs ({paused_hours} hrs @ ${APP_COSTS[app]['paused']}/hr): ${APP_COSTS[app]['paused'] * paused_hours:.2f}")
                st.write(f"- Storage costs: ${APP_COSTS[app]['storage']:.2f}/month")

        if use_cromwell and "Jupyter Notebooks" in apps:
            st.write("\nCromwell Costs:")
            st.write(f"- First application ({analysis_hours} hours @ ${CROMWELL_HOURLY}/hr): ${CROMWELL_HOURLY * analysis_hours:.2f}")
            if cromwell_apps > 1:
                additional_cost = CROMWELL_ADDITIONAL_HOURLY * analysis_hours * (cromwell_apps-1)
                st.write(f"- Additional applications ({cromwell_apps-1}) ({analysis_hours} hours @ ${CROMWELL_ADDITIONAL_HOURLY}/hr): ${additional_cost:.2f}")

        st.write(f"\nBucket Storage ({storage} GB @ ${STORAGE_COST_PER_GB}/GB): ${cost_details['bucket_storage_costs']:.2f}")
        st.write(f"Workspace costs ({workspaces} workspace(s) @ ${WORKSPACE_COST}/workspace): ${cost_details['workspace_costs']:.2f}")
        st.write("\nPersistent Disk Status:")
        if delete_environment:
            st.write("- Environments will be deleted when not in use (no persistent disk costs)")
        else:
            st.write("- Environments will be maintained when not in use")
            for app in apps:
                st.write(f"  - {app} persistent disk cost: ${APP_COSTS[app]['storage']:.2f}/month")

    # Notes section
    st.markdown("---")
    st.markdown("### Notes")
    if "Jupyter Notebooks" in apps:
        st.write("Current Jupyter configuration:")
        st.write(f"- VM: {jupyter_config}")
        st.write(f"  - Running cost: ${JUPYTER_VM_CONFIGS[jupyter_config]['running']}/hr")
        st.write(f"  - Paused cost: ${JUPYTER_VM_CONFIGS[jupyter_config]['paused']}/hr")
        
        if jupyter_gpu_config != "No GPU":
            gpu_config = JUPYTER_GPU_CONFIGS[jupyter_gpu_config]
            st.write(f"- GPU: {jupyter_gpu_config}")
            st.write(f"  - Cost: ${gpu_config['price']}/hr")
            st.write(f"  - Memory: {gpu_config['memory']} GB {gpu_config['type']}")
        
        if use_dataproc:
            st.write("\nDataproc configuration:")
            st.write("- Pricing: $0.01 per vCPU per hour")
            st.write("- Billed by the second (1-minute minimum)")
            st.write("- Additional Compute Engine costs apply")

    if use_cromwell:
        st.write("\nCromwell pricing:")
        st.write("- First application: $0.375 per hour")
        st.write("- Additional applications: $0.22 per hour each")

    st.write("\nGeneral notes:")
    st.write("- Storage costs include both application persistent disk and bucket storage")
    st.write("- WGS extraction costs are $0.02 per sample")
    st.write("- All prices are in USD")

if __name__ == "__main__":
    main()