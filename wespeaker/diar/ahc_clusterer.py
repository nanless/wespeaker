# Copyright (c) 2024 WeSpeaker Team
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

"""
Agglomerative Hierarchical Clustering (AHC) for speaker diarization.
This implementation is adapted from 3D-Speaker and VBx.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    import fastcluster
    from scipy.cluster.hierarchy import fcluster
    from scipy.spatial.distance import squareform
except ImportError:
    raise ImportError(
        "Package 'fastcluster' is required for AHC clustering. "
        "Please install it by: pip install fastcluster"
    )


class AHCluster:
    """
    Agglomerative Hierarchical Clustering, a bottom-up approach which iteratively merges 
    the closest clusters until a termination condition is reached.
    This implementation is adapted from https://github.com/BUTSpeechFIT/VBx and 3D-Speaker.
    """

    def __init__(self, fix_cos_thr=0.3, min_cluster_size=0, mer_cos=None):
        """
        Args:
            fix_cos_thr: Fixed cosine threshold for clustering (default: 0.3)
            min_cluster_size: Minimum cluster size (default: 0, no filtering)
            mer_cos: Merge threshold for similar clusters (default: None, no merging)
        """
        self.fix_cos_thr = fix_cos_thr
        self.min_cluster_size = min_cluster_size
        self.mer_cos = mer_cos

    def __call__(self, X, speaker_num=None):
        """
        Perform AHC clustering on embeddings.
        
        Args:
            X: Embedding matrix [N, D]
            speaker_num: Oracle number of speakers (optional)
        
        Returns:
            labels: Cluster labels [N]
        """
        if len(X) <= 1:
            return np.zeros(len(X), dtype=int)
        
        # Compute cosine similarity matrix
        scr_mx = cosine_similarity(X)
        
        # Convert to distance matrix (negative similarity for linkage)
        scr_mx = squareform(-scr_mx, checks=False)
        
        # Perform hierarchical clustering
        lin_mat = fastcluster.linkage(scr_mx, method='average', preserve_input='False')
        
        # Adjust linkage matrix to ensure non-negative distances
        adjust = abs(lin_mat[:, 2].min())
        lin_mat[:, 2] += adjust
        
        # Form clusters using threshold
        threshold = -self.fix_cos_thr + adjust
        labels = fcluster(lin_mat, threshold, criterion='distance') - 1
        
        # Filter minor clusters
        if self.min_cluster_size > 0:
            labels = self.filter_minor_cluster(labels, X, self.min_cluster_size)
        
        # Merge similar clusters
        if self.mer_cos is not None:
            labels = self.merge_by_cos(labels, X, self.mer_cos)
        
        return labels

    def filter_minor_cluster(self, labels, x, min_cluster_size):
        """Remove extremely minor clusters and reassign to major clusters."""
        cset = np.unique(labels)
        csize = np.array([(labels == i).sum() for i in cset])
        minor_idx = np.where(csize <= min_cluster_size)[0]
        
        if len(minor_idx) == 0:
            return labels
        
        minor_cset = cset[minor_idx]
        major_idx = np.where(csize > min_cluster_size)[0]
        
        if len(major_idx) == 0:
            return np.zeros_like(labels)
        
        major_cset = cset[major_idx]
        major_center = np.stack([x[labels == i].mean(0) for i in major_cset])
        
        for i in range(len(labels)):
            if labels[i] in minor_cset:
                cos_sim = cosine_similarity(x[i][np.newaxis], major_center)
                labels[i] = major_cset[cos_sim.argmax()]
        
        return labels

    def merge_by_cos(self, labels, x, cos_thr):
        """Merge similar speakers by cosine similarity."""
        assert cos_thr > 0 and cos_thr <= 1
        
        while True:
            cset = np.unique(labels)
            if len(cset) == 1:
                break
            
            centers = np.stack([x[labels == i].mean(0) for i in cset])
            affinity = cosine_similarity(centers, centers)
            affinity = np.triu(affinity, 1)
            idx = np.unravel_index(np.argmax(affinity), affinity.shape)
            
            if affinity[idx] < cos_thr:
                break
            
            c1, c2 = cset[np.array(idx)]
            labels[labels == c2] = c1
        
        return labels


def cluster(embeddings, fix_cos_thr=0.3, min_cluster_size=0, mer_cos=None, speaker_num=None):
    """
    Perform AHC clustering on embeddings.
    
    Args:
        embeddings: Embedding matrix [N, D] or list of embeddings
        fix_cos_thr: Fixed cosine threshold for clustering (default: 0.3)
        min_cluster_size: Minimum cluster size (default: 0)
        mer_cos: Merge threshold for similar clusters (default: None)
        speaker_num: Oracle number of speakers (optional)
    
    Returns:
        labels: Cluster labels [N]
    """
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
    
    if len(embeddings) <= 1:
        return np.zeros(len(embeddings), dtype=int)
    
    clusterer = AHCluster(
        fix_cos_thr=fix_cos_thr,
        min_cluster_size=min_cluster_size,
        mer_cos=mer_cos
    )
    
    return clusterer(embeddings, speaker_num=speaker_num)

