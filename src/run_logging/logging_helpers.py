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