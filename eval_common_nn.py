# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context("paper", font_scale=1)
colors = sns.color_palette("colorblind", 8)
fig, ax = plt.subplots(figsize=(12,8))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--teacher_output', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--student_outputs', nargs='+', help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--remove_i', action='store_true', help="""If tis a mapping on self, remove the obvious index.""")
    parser.add_argument('--save', default='/tmp/test.png', help="""Saving the facetgrid plot""")
    args = parser.parse_args()
    t_scores = torch.load(args.teacher_output)
    print(t_scores.shape)

    scores = {}
    name_map = {'fuss_v2': 'CoSS', 'seed': 'SEED', 'bingo': 'BINGO', 'disco': 'DisCo'}
    col_vals = []
    column_names = ['Method', 'Top', 'NN-IoU']
    values = []
    maxs = [0, 0, 0, 0, 0]
    for idx, student_output in enumerate(args.student_outputs):
        print('*'*50)
        print(student_output)
        s_scores = torch.load(student_output)
        if student_output.split('/')[-2] not in name_map:
            name_map[student_output.split('/')[-2]] = 'lmbda'
        method_name = name_map[student_output.split('/')[-2]]
        u = {2:0, 6:0, 12:0, 22:0, 32:0}
        n = {2:0, 6:0, 12:0, 22:0, 32:0}

        for i in range(t_scores.shape[0]):
            for k in u.keys():
                t_score = set(t_scores[i][:k].numpy().tolist())
                if args.remove_i:
                    try:
                        t_score.remove(i)
                    except:
                        t_score.remove(int(t_scores[i][k-1]))
                else:
                        t_score.remove(int(t_scores[i][k-1]))

        
                s_score = set(s_scores[i][:k].numpy().tolist())
                if args.remove_i:
                    try:
                        s_score.remove(i)
                    except:
                        s_score.remove(int(s_scores[i][k-1]))
                else:
                        s_score.remove(int(s_scores[i][k-1]))
            
                u[k] += len(t_score.intersection(s_score))
        prefix = '\t'
        for i, (k,v) in enumerate(u.items()):
            s = v/((k-1)*t_scores.shape[0])
            prefix+=f'top-{k-1}, {s*100:.2f}\t\t\t'
            values.append([method_name, k-1, s])
            if s > maxs[i]:
                maxs[i] = s
        col_vals.append(colors[idx])
        print(prefix)

    df = pd.DataFrame(values, columns=column_names)
    print(df)
    print(col_vals)
    g = sns.FacetGrid(df, col='Top')
    g.map_dataframe(sns.barplot, x='Method', y='NN-IoU', hue='Method', palette=colors, dodge=False)
    #g.refline(y=df["NN-IoU"].max())
    #g.add_legend()

    g.set_axis_labels('', 'IoU', fontdict={'weight': 'bold',  'fontsize': 12})
    g.set_titles(col_template="NN-{col_name}", fontdict={'weight': 'bold', 'fontsize': 14})

    ax0, ax1, ax2 = g.axes[0]
    ax0.axhline(maxs[0], ls='--', color='r', linewidth='1.5')
    ax1.axhline(maxs[1], ls='--', color='r', linewidth='1.5')
    ax2.axhline(maxs[2], ls='--', color='r', linewidth='1.5')

    g.tight_layout()
    g.savefig(args.save)







        
    
