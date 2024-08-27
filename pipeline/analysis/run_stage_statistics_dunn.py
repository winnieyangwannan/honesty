import numpy as np
import os
import argparse
from pipeline.configs.honesty_config_generation_intervention import Config
from pipeline.jailbreak_config_generation import Config
import pickle
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
import torch

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='google/gemma-2-9b-it')
    parser.add_argument('--save_path', type=str, required=False, default='D:\Data\jailbreak')
    parser.add_argument('--data_type', type=str, required=False, default=16)
    parser.add_argument('--contrastive_type', metavar='N', type=str, nargs='+',
                        help='a list of strings')

    return parser.parse_args()


def get_intra_cluster_distance(cluster_1, cluster_2):

    """
    intra_cluster_distance measures the compactness or cohesion of data points within a cluster.
    The smaller the intra-cluster distance, the more similar and tightly packed the data points are within the cluster.

    input:
        cluster_1 [n_data, d_data]
        cluster_2 [n_data, d_data]
    output:
        cluster_1 [n_data, n_data]
        cluster_2 [n_data, n_data]
    """

    distance_intra_1 = pairwise_distances(cluster_1, metric='euclidean')
    distance_intra_2 = pairwise_distances(cluster_2, metric='euclidean')
    # distance_intra_1 = np.tril(distance_intra_1, k=-1)
    # distance_intra_2 = np.tril(distance_intra_2, k=-1)

    return distance_intra_1, distance_intra_2


def intra_cluster_distance_high_pca(activations_all, activations_pca):
    """
    measures the separation or dissimilarity between clusters.
    The larger the inter-cluster distance, the more distinct and well-separated the clusters are from each other.
    """
    activations_all_1 = activations_all[0]
    activations_all_2 = activations_all[1]

    activations_pca_1 = activations_pca[0]
    activations_pca_2 = activations_pca[1]

    distance_intra_cluster_1, distance_intra_cluster_2 = get_intra_cluster_distance(activations_all_1,
                                                                                    activations_all_2)
    distance_intra_cluster_1_pca, distance_intra_cluster_2_pca = get_intra_cluster_distance(activations_pca_1,
                                                                                            activations_pca_2)
    distance_intra = np.stack((distance_intra_cluster_1, distance_intra_cluster_2), axis=0)
    distance_intra_pca = np.stack((distance_intra_cluster_1_pca, distance_intra_cluster_2_pca), axis=0)

    return distance_intra, distance_intra_pca


def get_inter_cluster_distance(cluster_1, cluster_2):
    n_data = cluster_1.shape[0]
    cluster_all = np.concatenate((cluster_1, cluster_2), axis=0)
    distance_all = pairwise_distances(cluster_all, metric='euclidean')
    distance_inter_cluster = distance_all[n_data:, :n_data]

    return distance_inter_cluster


def inter_cluster_distance_high_pca(activations_all, activations_pca):
    activations_all_1 = activations_all[0]
    activations_all_2 = activations_all[1]

    activations_pca_1 = activations_pca[0]
    activations_pca_2 = activations_pca[1]

    distance_inter_cluster = get_inter_cluster_distance(activations_all_1, activations_all_2)
    distance_inter_cluster_pca = get_inter_cluster_distance(activations_pca_1, activations_pca_2)

    return distance_inter_cluster, distance_inter_cluster_pca


def get_dunn_index(distance_intra_cluster, distance_inter_cluster):
    """
    Index = min_intercluster_distance / max_intracluster_distance
    see nice explanation here: https://medium.com/@Suraj_Yadav/understanding-intra-cluster-distance-inter-cluster-distance-and-dun-index-a-comprehensive-guide-a8de726f5769 )

    min_intercluster_distance: The minimum distance between any pair of data points from different clusters.
    max_intracluster_distance: The maximum distance between any pair of data points within the same cluster.
    the Dunn Index compares the smallest distance between two clusters with the largest distance within a cluster. A higher Dunn Index value indicates a better clustering solution with more distinct and well-separated clusters.

    """
    # distance_intra_cluster_norm = MinMaxScaler().fit_transform(distance_intra_cluster.reshape(-1, 1))
    # distance_inter_cluster_norm = MinMaxScaler().fit_transform(distance_inter_cluster.reshape(-1, 1))

    # distance_intra_cluster_norm = stats.zscore(distance_intra_cluster, axis=None)
    # distance_inter_cluster_norm = stats.zscore(distance_inter_cluster, axis=None)

    max_intracluster_distance = np.mean(distance_intra_cluster)
    min_intercluster_distance = np.mean(distance_inter_cluster)

    # max_intracluster_distance = np.max(distance_intra_cluster)
    # min_intercluster_distance = np.min(distance_inter_cluster)
    index = min_intercluster_distance/max_intracluster_distance
    return index


def dunn_index_contrastive(distance_intra_cluster, distance_intra_cluster_pca,
                           distance_inter_cluster, distance_inter_cluster_pca):

    dunn_index = get_dunn_index(distance_intra_cluster, distance_inter_cluster)
    dunn_index_pca = get_dunn_index(distance_intra_cluster_pca, distance_inter_cluster_pca)

    return dunn_index, dunn_index_pca


def stage_2_get_distance_contrastive_dunn_index(data_all,
                                         contrastive_type,
                                         ):
    activations_all = data_all['activations_all']
    activations_all_pca = data_all['activations_pca']
    labels_all = data_all['contrastive_labels_all']

    n_data = int(activations_all[0].shape[0]/2)
    n_layers = activations_all.shape[2]

    dunn_index_all = np.zeros((2, n_layers, 1))
    dunn_index_pca_all = np.zeros((2, n_layers, 1))
    for ii in range(2):
        # first half of data
        activations_positive = activations_all[ii][:n_data, :, :]
        activations_positive_pca = activations_all_pca[ii][:n_data, :, :]
        labels_positive = labels_all[ii][:n_data]
        # second half of data
        activations_negative = activations_all[ii][n_data:, :, :]
        activations_negative_pca = activations_all_pca[ii][n_data:, :, :]
        labels_negative = labels_all[ii][n_data:]
        # concatenate two contrastive group
        activations = np.stack((activations_positive, activations_negative))
        activations_pca = np.stack((activations_positive_pca, activations_negative_pca))
        labels = np.stack((labels_positive, labels_negative))


        dist_all = np.zeros((n_layers, 2*n_data, 2*n_data))
        dist_all_pca = np.zeros((n_layers, 2*n_data, 2*n_data))
        for layer in range(n_layers):
            # dist = pairwise_distances(activations[:, layer, :])
            # dist_pca = pairwise_distances(activations_pca[:, layer, :])
            # dist_all[layer] = dist
            # dist_all_pca[layer] = dist_pca

            # step 1: get intra-cluster distance (within one contrastive cluster)
            distance_intra_cluster, distance_intra_cluster_pca = intra_cluster_distance_high_pca(activations[:, :, layer, :],
                                                                                                 activations_pca[:, :, layer, :])

            # step 2: get inter-cluster distance (between contrastive clusters)
            distance_inter_cluster, distance_inter_cluster_pca = inter_cluster_distance_high_pca(activations_all[:, :, layer, :],
                                                                                                 activations_pca[:, :, layer, :])

            # step 3: calculate Dunn-index (see a nice explanation here: https://medium.com/@Suraj_Yadav/understanding-intra-cluster-distance-inter-cluster-distance-and-dun-index-a-comprehensive-guide-a8de726f5769 )
            dunn_index, dunn_index_pca = dunn_index_contrastive(distance_intra_cluster, distance_intra_cluster_pca,
                                                                distance_inter_cluster, distance_inter_cluster_pca)
            dunn_index_all[ii, layer] = dunn_index
            dunn_index_pca_all[ii, layer] = dunn_index_pca

    return dunn_index_all, dunn_index_pca_all


def stage_1_get_distance_contrastive_dunn_index(data_all,
                                                contrastive_type,
                                                ):
    activations_all = data_all['activations_all']
    activations_all_pca = data_all['activations_pca']
    labels_all = data_all['contrastive_labels_all']

    n_data = int(activations_all[0].shape[0]/2)
    n_layers = activations_all.shape[2]

    dunn_index_all = np.zeros((2, n_layers, 1))
    dunn_index_pca_all = np.zeros((2, n_layers, 1))
    for ii in range(2):
        # first half of contrastive group 1
        activations_positive = activations_all[0][ii*n_data:(ii+1)*n_data, :, :]
        activations_positive_pca = activations_all_pca[0][ii*n_data:(ii+1)*n_data, :, :]
        labels_positive = labels_all[0][ii*n_data:(ii+1)*n_data]
        # first half of contrastive group 2
        activations_negative = activations_all[1][ii*n_data:(ii+1)*n_data, :, :]
        activations_negative_pca = activations_all_pca[1][ii*n_data:(ii+1)*n_data, :, :]
        labels_negative = labels_all[1][ii*n_data:(ii+1)*n_data]
        # concatenate two contrastive group
        activations = np.stack((activations_positive, activations_negative))
        activations_pca = np.stack((activations_positive_pca, activations_negative_pca))
        labels = np.stack((labels_positive, labels_negative))


        dist_all = np.zeros((n_layers, 2*n_data, 2*n_data))
        dist_all_pca = np.zeros((n_layers, 2*n_data, 2*n_data))
        for layer in range(n_layers):
            # dist = pairwise_distances(activations[:, layer, :])
            # dist_pca = pairwise_distances(activations_pca[:, layer, :])
            # dist_all[layer] = dist
            # dist_all_pca[layer] = dist_pca

            # step 1: get intra-cluster distance (within one contrastive cluster)
            distance_intra_cluster, distance_intra_cluster_pca = intra_cluster_distance_high_pca(activations[:, :, layer, :],
                                                                                                 activations_pca[:, :, layer, :])

            # step 2: get inter-cluster distance (between contrastive clusters)
            distance_inter_cluster, distance_inter_cluster_pca = inter_cluster_distance_high_pca(activations_all[:, :, layer, :],
                                                                                                 activations_pca[:, :, layer, :])

            # step 3: calculate Dunn-index (see a nice explanation here: https://medium.com/@Suraj_Yadav/understanding-intra-cluster-distance-inter-cluster-distance-and-dun-index-a-comprehensive-guide-a8de726f5769 )
            dunn_index, dunn_index_pca = dunn_index_contrastive(distance_intra_cluster, distance_intra_cluster_pca,
                                                                distance_inter_cluster, distance_inter_cluster_pca)
            dunn_index_all[ii, layer] = dunn_index
            dunn_index_pca_all[ii, layer] = dunn_index_pca

    return dunn_index_all[1], dunn_index_pca_all[1]


def plot_stage_1(dunn_index_all, dunn_index_pca):
    n_layers = dunn_index_all.shape[0]
    line_width = 2
    marker_size = 4
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Original High Dimensional Space', 'PCA',
                                       )
                        )
    fig.add_trace(go.Scatter(
        x=np.arange(n_layers), y=np.mean(dunn_index_all, axis=1),
        mode='lines+markers',
        showlegend=False,
        marker=dict(size=marker_size),
        line=dict(color="royalblue", width=line_width)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=np.arange(n_layers), y=np.mean(dunn_index_pca, axis=1),
        mode='lines+markers',
        showlegend=False,
        marker=dict(size=marker_size),
        line=dict(color="royalblue", width=line_width)
    ), row=1, col=2)
    fig.show()

    return fig


def plot_stage_2(dunn_index_all, dunn_index_pca, contrastive_type):
    n_layers = dunn_index_all.shape[1]
    line_width = 2
    marker_size = 4
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Original High Dimensional Space', 'PCA',
                                       )
                        )
    fig.add_trace(go.Scatter(
        x=np.arange(n_layers), y=np.mean(dunn_index_all[0], axis=1),
        mode='lines+markers',
        name=contrastive_type[0],
        showlegend=False,
        marker=dict(size=marker_size),
        line=dict(color="royalblue", width=line_width)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=np.arange(n_layers), y=np.mean(dunn_index_all[1], axis=1),
        mode='lines+markers',
        name=contrastive_type[1],
        showlegend=False,
        marker=dict(size=marker_size),
        line=dict(color="firebrick", width=line_width)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=np.arange(n_layers), y=np.mean(dunn_index_pca[0], axis=1),
        mode='lines+markers',
        name=contrastive_type[0],
        showlegend=False,
        marker=dict(size=marker_size),
        line=dict(color="royalblue", width=line_width)
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=np.arange(n_layers), y=np.mean(dunn_index_pca[1], axis=1),
        mode='lines+markers',
        name=contrastive_type[1],
        showlegend=False,
        marker=dict(size=marker_size),
        line=dict(color="firebrick", width=line_width)
    ), row=1, col=2)
    fig.show()

    return fig


def get_pca_layer_by_layer(activations_all,
                           n_components=3):
    n_layers = activations_all.shape[2]
    n_samples = activations_all.shape[1]
    n_contrastive_group = activations_all.shape[0]
    pca = PCA(n_components=n_components)

    activations_pca = np.zeros((n_contrastive_group, n_samples, n_layers, n_components))
    for group in range(n_contrastive_group):
        for layer in range(n_layers):
            activations_pca[group, :, layer, :] = pca.fit_transform(activations_all[group, :, layer, :].cpu().numpy())

    return activations_pca


def run_pipeline(model_path, save_path,
                 data_type=["jailbreakbench", "harmless"],
                 contrastive_type=['evil_confidant', 'AIM'],
                 layer_plot=0):
    """Run the full pipeline."""

    model_alias = os.path.basename(model_path)

    cfg = Config(model_alias=model_alias, model_path=model_path,
                 jailbreak_type=contrastive_type[0], save_path=save_path,
                 )

    print(cfg)
    artifact_path = cfg.artifact_path()
    stage_path = os.path.join(artifact_path, 'stage_stats_dunn')
    if not os.path.exists(stage_path):
        os.makedirs(stage_path)

    # positive
    filename = artifact_path + os.sep + \
               model_alias + f'_activation_pca_HHH_{contrastive_type[1]}.pkl'
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    n_data = len(data['activations_positive'])

    activations_positive = data['activations_positive']
    contrastive_labels_positive_str = ['HHH'] * n_data
    contrastive_labels_positive = np.ones((n_data))
    data_labels_positive = data['labels']

    activations_negative = data['activations_negative']
    activations_all = torch.stack((activations_positive, activations_negative), dim=0)
    contrastive_labels_negative_str = [contrastive_type[1]]*n_data
    contrastive_labels_negative = np.zeros((n_data))

    contrastive_labels_all_str = np.stack((np.array(contrastive_labels_positive_str), np.array(contrastive_labels_negative_str)))
    contrastive_labels_all = np.stack((np.array(contrastive_labels_positive), np.array(contrastive_labels_negative)))
    data_labels_all = np.stack((data_labels_positive, np.array(data['labels'])))

    # contrastive_labels_all = contrastive_labels_all.tolist()
    # data_labels_all = data_labels_all.tolist()

    # pca
    activations_pca = get_pca_layer_by_layer(activations_all,
                                             n_components=3)

    activations_all = activations_all.cpu().numpy()
    data_all = {
        'activations_all': activations_all,
        'activations_pca': activations_pca,
        'contrastive_labels_all': contrastive_labels_all,
        'contrastive_labels_all_str': contrastive_labels_all_str,
        'data_labels_all': data_labels_all,
    }

    # stage 1: separation between contrastive clusters with dunn_index as metric
    dunn_index_all, dunn_index_pca_all = stage_1_get_distance_contrastive_dunn_index(data_all, contrastive_type)
    fig = plot_stage_1(dunn_index_all, dunn_index_pca_all)
    fig.write_html(stage_path + os.sep + 'stage_1_dunn_index_' +
                    f'_{contrastive_type[0]}_{contrastive_type[1]}' + '.html')
    pio.write_image(fig, stage_path + os.sep + 'stage_1_dunn_index_' +
                    f'_{contrastive_type[0]}_{contrastive_type[1]}' + '.png',
                    scale=6)
    # stage 2: separation between contrastive clusters with dunn_index as metric
    dunn_index_all, dunn_index_pca_all = stage_2_get_distance_contrastive_dunn_index(data_all,
                                                                                     contrastive_type,
                                                                                     )
    fig = plot_stage_2(dunn_index_all, dunn_index_pca_all, contrastive_type=contrastive_type)
    fig.write_html(stage_path + os.sep + 'stage_2_dunn_index_' +
                   f'_{contrastive_type[0]}_{contrastive_type[1]}' + '.html')
    pio.write_image(fig, stage_path + os.sep + 'stage_2_dunn_index_' +
                    f'_{contrastive_type[0]}_{contrastive_type[1]}' + '.png',
                    scale=6)


if __name__ == "__main__":
    args = parse_arguments()
    data_type = ["jailbreakbench", "harmless"]

    run_pipeline(model_path=args.model_path, save_path=args.save_path,
                 data_type=data_type, contrastive_type=tuple(args.contrastive_type),
                 )
