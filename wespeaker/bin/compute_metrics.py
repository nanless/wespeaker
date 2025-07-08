# Copyright (c) 2022 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2022 Hongji Wang (jijijiang77@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import fire
import numpy as np
import matplotlib.pyplot as plt

from wespeaker.utils.score_metrics import (compute_pmiss_pfa_rbst, compute_eer,
                                           compute_c_norm)


def compute_metrics(scores_file, p_target=0.01, c_miss=1, c_fa=1):
    scores = []
    labels = []

    with open(scores_file) as readlines:
        for line in readlines:
            tokens = line.strip().split()
            # assert len(tokens) == 4
            scores.append(float(tokens[2]))
            labels.append(tokens[3] == 'target')

    scores = np.array(scores)
    labels = np.array(labels)

    fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
    eer, thres = compute_eer(fnr, fpr, scores)

    min_dcf = compute_c_norm(fnr,
                             fpr,
                             p_target=p_target,
                             c_miss=c_miss,
                             c_fa=c_fa)
    
    # Generate plots
    scores_dir = os.path.dirname(scores_file)
    trial_name = os.path.basename(scores_file).replace('.score', '')
    plot_path = os.path.join(scores_dir, f'{trial_name}_eer_curves.png')
    plot_eer_curves(fnr, fpr, scores, thres, labels, plot_path)
    
    print("---- {} -----".format(os.path.basename(scores_file)))
    print("EER = {0:.3f}".format(100 * eer))
    print("threshold = {:.3f}".format(thres))
    print("minDCF (p_target:{} c_miss:{} c_fa:{}) = {:.3f}".format(
        p_target, c_miss, c_fa, min_dcf))


def plot_eer_curves(fnr, fpr, scores, thres, labels, save_path):
    plt.figure(figsize=(15, 5))
    
    # Plot 1: FNR vs FPR
    plt.subplot(131)
    plt.plot(fpr, fnr, 'b-', label='ROC')
    plt.plot([0, 1], [0, 1], 'r--', label='EER line')
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.title('FNR vs FPR')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Error Rates vs Scores
    plt.subplot(132)
    # Create threshold points for plotting
    thresholds = np.sort(scores)
    fnrs = []
    fprs = []
    for t in thresholds:
        predictions = (scores >= t).astype(int)
        fn = np.sum((predictions == 0) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        tn = np.sum((predictions == 0) & (labels == 0))
        tp = np.sum((predictions == 1) & (labels == 1))
        fnrs.append(fn / (fn + tp) if (fn + tp) > 0 else 1.)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 1.)
    
    plt.plot(thresholds, fnrs, 'b-', label='FNR')
    plt.plot(thresholds, fprs, 'r-', label='FPR')
    plt.axvline(x=thres, color='g', linestyle='--', label=f'EER Threshold')
    plt.xlabel('Score Threshold')
    plt.ylabel('Error Rate')
    plt.title('Error Rates vs Score Threshold')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Score Distribution
    plt.subplot(133)
    plt.hist(scores[labels==1], bins=50, density=True, alpha=0.7, label='Target', color='g')
    plt.hist(scores[labels==0], bins=50, density=True, alpha=0.7, label='Non-target', color='r')
    plt.axvline(x=thres, color='b', linestyle='--', label=f'EER Threshold: {thres:.3f}')
    plt.xlabel('Scores')
    plt.ylabel('Density')
    plt.title('Score Distribution')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main(p_target=0.01, c_miss=1, c_fa=1, *scores_files):
    for scores_file in scores_files:
        compute_metrics(scores_file, p_target, c_miss, c_fa)


if __name__ == "__main__":
    fire.Fire(main)
