import wandb
run = wandb.init()
artifact = run.use_artifact('aksoym/recovery_rate/model-35cgln33:v24', type='model')
artifact_dir = artifact.download()