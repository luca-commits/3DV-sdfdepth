import os
from glob import glob
import json
import pandas as pd

lpips_cutoff = 0.3
abs_rel_cutoff = 0.1

good_models = []


df = pd.read_csv("eval_metrics.csv")

for ind in df.index:

    if df['Eval Images Metrics Dict (all images)/lpips'][ind] < lpips_cutoff and \
        df['Eval Images Metrics Dict (all images)/depth_abs_rel'][ind] < abs_rel_cutoff:

        good_models.append(df['scene'][ind])

print("Models remaining after filtering:", len(good_models))

with open('filtered_models.txt', 'w') as fp:
    for model in good_models:
        fp.write("%s\n" % model)