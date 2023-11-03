import os
from glob import glob
import json
import pandas as pd

models_dir = "/cluster/project/infk/courses/252-0579-00L/group26/sniall/3dv_sdfdepth/training/nerfstudio/outputs-post-sweep/*/"

models = sorted(glob(models_dir))

print("Number of models:", len(models))

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

num_models = 0

all_data = []

for model in models:
    runs = glob(model + "depth-nerfacto/*/")
    latest_subdir = max(runs, key=os.path.getmtime)

    json_path = latest_subdir + "wandb/latest-run/files/wandb-summary.json"   

    try:
        with open(json_path) as json_file:
            data = json.load(json_file)
            metric_dict = {}

            metric_dict["scene"] = model.split('/')[-2]

            if metric_list[0] in data:
                for metric in metric_list:
                    metric_dict[metric] = data[metric]

                if metric_dict["Eval Images Metrics Dict (all images)/lpips"] < 0.22 and metric_dict["Eval Images Metrics Dict (all images)/depth_abs_rel"] < 0.05:
                    num_models += 1
                    all_data.append(metric_dict)
    except KeyError:
        print(f"{model}: Does not contain metrics")
        continue
    except FileNotFoundError:
        print(f"{model}: Could not find W&B Summary")
        continue

print("Valid runs: ", num_models)

df = pd.DataFrame.from_dict(all_data)

avg_df = df.drop('scene', axis=1).mean()
print(avg_df)

df.to_csv("eval_metrics.csv") 
