import os

from math import ceil, floor

import copy

import tqdm

import numpy as np

from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision.datasets.vision import VisionDataset

from sklearn.neighbors import KNeighborsClassifier

from scipy.stats import multivariate_normal

from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

from sklearn.manifold import TSNE
import umap

from simclr import SimClrBackbone

DEVICE = "cuda"

def load_simclr(simclr_model_name: str) -> SimClrBackbone:
    model = SimClrBackbone()
    out = os.path.join('./saved_models/simclr/', simclr_model_name)
    checkpoint = torch.load(out, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    epochs = checkpoint["epoch"]
    return model, epochs



def extract_simclr_features(model: SimClrBackbone, dataset: VisionDataset, layer: str = "repr"):

    assert layer in ["loss", "repr"]

    simclr_feature_size = 128 if layer == "loss" else 512
    num_examples = len(dataset)

    features = np.zeros((num_examples, simclr_feature_size))
    labels_poison = np.zeros((num_examples))
    labels_true = np.zeros((num_examples))

    batch_size = 256
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for i, (img, labels_batch_poison, labels_batch_true) in enumerate(dataloader):

        with torch.no_grad():
            if layer == "loss":
                features_batch = model(img.to(DEVICE)).cpu().data.numpy()
            elif layer == "repr":
                features_batch = model.forward_repr(img.to(DEVICE)).cpu().data.numpy()
            
        features[i*batch_size : i*batch_size+len(features_batch)] = features_batch
        labels_poison[i*batch_size : i*batch_size+len(labels_batch_poison)] = labels_batch_poison.long()
        labels_true[i*batch_size : i*batch_size+len(labels_batch_true)] = labels_batch_true.long()

    labels_poison = labels_poison.astype(int)
    labels_true = labels_true.astype(int)

    return features, labels_poison, labels_true



def calculate_features_2d(features: np.array, n_neighbors: int = 100, algorithm: str = "umap", min_dist: float = 0.1) -> np.array:
    assert algorithm in ["umap", "tsne"]
    
    if algorithm == "umap":
        alg = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    elif algorithm == "tsne":
        alg = TSNE(n_components = 2, perplexity = n_neighbors)
    features_2d = alg.fit_transform(features)
    return features_2d

def plot_features_2d(features_2d: np.array, labels: np.array, poison_indices: np.array, legend: bool = True, title: str = None, show: bool = True) -> None:
    num_classes = int(max(labels).item())

    # label poison examples as 10
    labels_10 = copy.deepcopy(labels)
    labels_10[poison_indices] = 10

    for i in range(num_classes+1):
        plt.scatter(features_2d[labels_10==i,1], features_2d[labels_10==i,0], s=7)
    plt.scatter(features_2d[labels_10==10,1], features_2d[labels_10==10,0], c = "black", marker= "x", s=1)

    if title: plt.title(title)

    if legend: plt.legend([str(i) for i in range(num_classes+1)] + ["poison"])
    
    if show: plt.show()

def calculate_and_plot_features_2d(features: np.array, labels: np.array, poison_indices: np.array, subset_size: int = None, legend: bool = True) -> np.array:
    # Plot only a subset
    if subset_size is None:
        subset_size = len(features)
    features_subset = features[:subset_size]
    labels_subset = labels[:subset_size]
    poison_indices_subset = poison_indices[:subset_size]
    
    features_2d = calculate_features_2d(features_subset)
    plot_features_2d(features_2d, labels_subset, poison_indices_subset, legend=legend)


def print_evaluate_cleanse(poison_predicted: np.array, poison_indices: np.array, verbose: bool = True) -> float:

    tp = (poison_indices & poison_predicted).sum()
    fp = (np.invert(poison_indices) & poison_predicted).sum()
    fn = (poison_indices & np.invert(poison_predicted)).sum()
    tn = (np.invert(poison_indices) & np.invert(poison_predicted)).sum()

    fnr = fn/(fn+tp) if fn+tp!=0 else 0
    tnr = tn/(tn+fp) if tn+fp!=0 else 0
    poison_rate = fn/(fn+tn) if fn+tn!=0 else 0

    if verbose:
        print(f"{tp} \t {fp}")
        print(f"{fn} \t {tn}")
        print(f"Percentage of poisoned images (out of all poisoned) kept: {100*fnr: .2f}%")
        print(f"Percentage of clean images (out of all clean) kept: {100*tnr: .2f}%")
        print(f"Percentage of remaining poisoned images (out of all remaining): {100*poison_rate: .2f}%")

def evaluate_cleanse(poison_predicted: np.array, poison_indices: np.array, verbose: bool = True) -> float:

    tp = (poison_indices & poison_predicted).sum()
    fp = (np.invert(poison_indices) & poison_predicted).sum()
    fn = (poison_indices & np.invert(poison_predicted)).sum()
    tn = (np.invert(poison_indices) & np.invert(poison_predicted)).sum()

    fnr = fn/(fn+tp) if fn+tp!=0 else 0
    tnr = tn/(tn+fp) if tn+fp!=0 else 0
    poison_rate = fn/(fn+tn) if fn+tn!=0 else 0

    return poison_rate, fnr, tnr

def plot_histogram_poisoned(values: np.array, poison_indices: np.array = None, is_integer: bool = False, bins_num: int = 100, separation_line: float = None,
                            x_axis_label: str = None, y_axis_label: str = None, show: bool = True, legend: bool = True
                            ) -> None:
    if poison_indices is not None:
        values_clean = values[np.invert(poison_indices)]
        values_poisoned = values[poison_indices]
    else:
        values_clean = values[:]
        values_poisoned = []

    bins = np.linspace(floor(np.min(values)), ceil(np.max(values)), int(np.max(values)) if is_integer else bins_num)
    plt.hist(values_clean, bins, alpha=0.5, label='čisti')
    plt.hist(values_poisoned, bins, alpha=0.5, label='otrovani')

    plt.ylabel(y_axis_label)
    plt.xlabel(x_axis_label)

    if separation_line:
        plt.axvline(separation_line, color='red', linestyle='dashed', linewidth=1)

    if legend: plt.legend(loc='upper right')
    
    if show: plt.show()


def knn_cleanse(features: np.array, labels_poison: np.array, n_neighbors: int) -> np.array:
    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(features, labels_poison)
    labels_predicted = knn.predict(features)

    return labels_predicted != labels_poison


def gauss_cleanse(features: np.array, discard_percentage: float, poison_indices: np.array = None) -> np.array:
    mean = np.mean(features, axis=0)
    cov = np.cov(features, rowvar=0)

    probabilities = multivariate_normal.pdf(features, mean=mean, cov=cov, allow_singular=True)
    probabilities[probabilities <= 0] = 1e-100
    probabilities = -np.log(probabilities)

    discard_line = np.percentile(probabilities, (1-discard_percentage)*100)
    plot_histogram_poisoned(probabilities, poison_indices, separation_line=discard_line)

    predicted_poison_indices = probabilities > discard_line
    return predicted_poison_indices


def kmeans_cleanse(features: np.array, means: int = 11, mode: str = "both") -> np.array:

	assert mode in ["distance", "size", "both"]

	kmeans = KMeans(n_clusters=means, init="k-means++", n_init=1)
	kmeans.fit(features)
	
	if mode == "distance":
		centroid = np.mean(features, axis=0)
		cluster_center_distances = [euclidean(center, centroid) for center in kmeans.cluster_centers_]
		poison_cluster_index = cluster_center_distances.index(max(cluster_center_distances))
	
	elif mode == "size":
		predicted_cluster = kmeans.predict(features)
		_, counts = np.unique(predicted_cluster, return_counts=True)
		poison_cluster_index = np.argmin(counts)
	
	elif mode == "both":
		centroid = np.mean(features, axis=0)
		cluster_center_distances = [euclidean(center, centroid) for center in kmeans.cluster_centers_]
		poison_cluster_index_1 = cluster_center_distances.index(max(cluster_center_distances))

		predicted_cluster = kmeans.predict(features)
		_, counts = np.unique(predicted_cluster, return_counts=True)
		poison_cluster_index_2 = np.argmin(counts)

		if poison_cluster_index_1 != poison_cluster_index_2:
			# No poison detected
			return np.zeros(features.shape[0]).astype(bool)
		else:
			poison_cluster_index = poison_cluster_index_1

	predicted_poison_indices = kmeans.predict(features) == poison_cluster_index
	return predicted_poison_indices


class EnergyClassifier():

    def __init__(self, t=1):
        self.t = t

    def fit(self, X, y):
        self.X = X
        self.y = y
        
        self.C = int(np.max(y))
        self.Ic = {c:[i for i in range(len(y)) if y[i]==c] for c in range(self.C)}
        
    def predict_index(self, i):
        # consider improving with numpy and batch

        xi = self.X[i]

        exp_all = np.exp([xi*self.X[k]/self.t for k in range(len(self.X))])
        sum_exp_all_except_xi = np.sum([exp_all[k] for k in range(len(self.X)) if k!=i])
        mean_exp_c = [np.mean([exp_all[k] for k in self.Ic[c] if k!=i]) for c in range(self.C)]
    
        Scs = mean_exp_c / sum_exp_all_except_xi
        return np.argmax(Scs)

    def predict(self):
        predicted = np.zeros((len(self.X)))
        for i in tqdm(range(len(self.X))):
            predicted[i] = self.predict_index(i)
        return predicted
    
def energy_cleanse(features: np.array, labels_poison: np.array, t: float = 10) -> np.array:
    
    # if DATASET == "badnets":
    #     T = 100
    # elif DATASET == "wanet":
    #     T = 10
    # elif DATASET == "sig":
    #     T = 1
    # else:
    #     raise Exception("Invalid dataset")

    energy = EnergyClassifier(t=t)
    energy.fit(features, labels_poison)
    labels_predicted = energy.predict()

    return labels_predicted != labels_poison