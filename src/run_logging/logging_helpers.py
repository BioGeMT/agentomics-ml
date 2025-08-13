import wandb

def _is_wandb_active():
    """Check if wandb is initialized and active."""
    try:
        return wandb.run is not None
    except:
        return False

def log_inference_stage_and_metrics(stage, metrics=None):
    print(f"🔍 Logging inference stage {stage} to W&B")
    
    if not _is_wandb_active():
        print("   ⚠️  WandB not initialized - skipping logging")
        return
    
    wandb.log({"inference_stage": stage})
    
    if stage == 0 or stage == 1:
        null_metrics = {
            'AUPRC': -1,
            'AUROC': -1,
            'ACC': -1,
        }
        print(f"   📊 Logging null metrics (stage {stage}): {null_metrics}")
        wandb.log(null_metrics)
    else:
        print(f"   📊 Logging final metrics (stage {stage}): {metrics}")
        wandb.log(metrics)

def log_serial_metrics(prefix, metrics=None, iteration=None):
    """
    Log metrics to W&B with enhanced debugging and error handling.
    
    Args:
        prefix: Metric prefix (e.g., 'train', 'validation')
        metrics: Dict of metrics to log, or None for null metrics
        iteration: Iteration number for step parameter
    """
    print(f"🔍 Logging {prefix} metrics to W&B for iteration {iteration}")
    
    if not _is_wandb_active():
        print("   ⚠️  WandB not initialized - skipping logging")
        return
    
    if not metrics:
        null_metrics = {
            # Classification metrics
            f'{prefix}/AUPRC': -1,
            f'{prefix}/AUROC': -1,
            f'{prefix}/ACC': -1,
            # Regression metrics  
            f'{prefix}/MSE': -1,
            f'{prefix}/RMSE': -1,
            f'{prefix}/MAE': -1,
            f'{prefix}/R2': -1,
        }
        print(f"   📊 Logging null metrics: {null_metrics}")
        try:
            wandb.log(null_metrics, step=iteration)
            print(f"   ✅ Null metrics logged successfully")
        except Exception as e:
            print(f"   ❌ Failed to log null metrics: {e}")
    else:
        # Add prefix to all metric keys
        prefixed_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        print(f"   📊 Logging actual metrics: {prefixed_metrics}")
        
        try:
            wandb.log(prefixed_metrics, step=iteration)
            print(f"   ✅ Metrics logged successfully to W&B")
            
            # Verify the metrics were logged by checking if they're reasonable
            for key, value in prefixed_metrics.items():
                if isinstance(value, (int, float)):
                    if value < -1 or value > 1:
                        print(f"   ⚠️  Warning: Metric {key} has unusual value: {value}")
                else:
                    print(f"   ⚠️  Warning: Metric {key} has non-numeric value: {value}")
                    
        except Exception as e:
            print(f"   ❌ Failed to log metrics to W&B: {e}")
            print(f"   ❌ This is why {prefix} metrics are not appearing in W&B graphs!")
            # Try to log without the step parameter as fallback
            try:
                wandb.log(prefixed_metrics)
                print(f"   ✅ Metrics logged successfully without step parameter")
            except Exception as e2:
                print(f"   ❌ Failed to log metrics even without step parameter: {e2}")
   