import wandb
import pandas as pd

metric_list = ["Eval Images Metrics Dict (all images)/psnr", \
"Eval Images Metrics Dict (all images)/ssim", \
"Eval Images Metrics Dict (all images)/lpips", \
"Eval Images Metrics Dict (all images)/depth_abs_rel", \
"Eval Images Metrics Dict (all images)/depth_silog", \
"Eval Images Metrics Dict (all images)/depth_mse", \
"Eval Images Metrics Dict (all images)/depth_sq_rel", \
"Eval Images Metrics Dict (all images)/depth_rms", \
"Eval Images Metrics Dict (all images)/depth_rms_log", \
"Eval Images Metrics Dict (all images)/a1", \
"Eval Images Metrics Dict (all images)/a2", \
"Eval Images Metrics Dict (all images)/a3"]

sweeps = wandb.Api().project("nerfstudio-project").sweeps()
# sweeps = [list(sweeps)[0]]

all_metrics = []

for sweep in sweeps:
    run = sweep.best_run()

    if run is None:
        continue

    # print(run.config)

    history = run.scan_history(metric_list)

    if len(list(history)) == 0:
        continue

    final_metrics = list(history)[-1]
    final_metrics["scene"] = run.config["pipeline"]["datamanager"]["dataparser"]["data"].split('/')[-1]

    # print(final_metrics)
    all_metrics.append(final_metrics)

df = pd.DataFrame.from_dict(all_metrics)

avg_df = df.drop('scene', axis=1).mean()
print(avg_df)

df.to_csv("eval_metrics.csv") 