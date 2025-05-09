import wandb
def log_inference_stage_and_metrics(stage, metrics=None):
    wandb.log({"inference_stage": stage})
    if stage == 0 or stage == 1:
        wandb.log({
            'AUPRC': -1,
            'AUROC': -1,
            'ACC': -1,
        })
    else:
        wandb.log(metrics)

def log_serial_metrics(prefix, metrics=None, iteration=None):
    if(not metrics):
        wandb.log({
            f'{prefix}/AUPRC': -1,
            f'{prefix}/AUROC': -1,
            f'{prefix}/ACC': -1,
        }, step=iteration)
    else:
        metrics = {f"{prefix}/{k}": v for k,v in metrics.items()}
        wandb.log(metrics, step=iteration)
   