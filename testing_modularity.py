import igraph as ig
import leidenalg
import json
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

def parse_shenzen_info():
    edge_dict = []
    with open('./data/roads_graph.txt') as f:
        for line in f:
            adjacency_list = str.split(line, ' ')
            edge_dict.append((int(adjacency_list[0]), int(adjacency_list[1])))

    vertex_info = []
    index = 0
    with open('./data/shenzhen_roads.geojson') as f:
        json_file = json.load(f)
    for example in json_file['features']:
        properties = example['properties']
        vertex_info.append(properties['speed'])
        index+=1
    
    return edge_dict, vertex_info

def analyze_per_move_history(per_move_history, iteration):
    history = []

    for i in range(0, len(per_move_history), 6):
        
        move_type = ''
        if per_move_history[i] == 0.0:
            move_type = 'move'
        elif per_move_history[i] == 1.0:
            move_type = 'merge'
        elif per_move_history[i] == 2.0:
            move_type = 'move_constrained'
        elif per_move_history[i] == 3.0:
            move_type = 'merge_constrained'
        
        phase = ''
        if (per_move_history[i+1] > 0.0):
            phase = f'move_phase_{per_move_history[i+1]}'
        elif (per_move_history[i+1] < 0.0):
            phase = f'refine_phase_{per_move_history[i+1] * -1}'

        history.append(
            {
                'move_type':move_type,
                'phase':phase,
                'node_id':per_move_history[i+2],
                'from_community_id':per_move_history[i+3],
                'to_community_id':per_move_history[i+4],
                'diff_move':per_move_history[i+5],
                'iteration':iteration,
            }
        )
    return history

def convert_edge_matrix_to_connectivity_matrix(edge_matrix):
  num_elements = max(edge_matrix['Source'].max(), edge_matrix['Target'].max())
  matrix = np.zeros((num_elements + 1, num_elements + 1))
  for (_, row) in edge_matrix.iterrows():
    target_row = row['Source']
    target_col = row['Target']
    matrix[target_row, target_col] = 1
  return matrix

def calculate_weighted_variance(grouped):
    feature_sum = grouped['Speed'].sum()
    squared_feature_sum = grouped['Squared_Speed'].sum()
    weighted_variance = squared_feature_sum - ((feature_sum * feature_sum) / grouped.size())
    return weighted_variance.sum()

def perform_agglomerative_clustering(edge_matrix, node_attributes, num_clusters):
    connectivity_matrix = convert_edge_matrix_to_connectivity_matrix(edge_matrix)
    agg_clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward', metric='euclidean', connectivity=connectivity_matrix, compute_distances=True)
    clusters = agg_clustering.fit(node_attributes)
    return clusters

def compare_clustering_result(node_attributes, partition, clusters: AgglomerativeClustering):
    flattened_attributes = np.array(node_attributes).flatten()
    squared_speed = flattened_attributes ** 2
    output_df = pd.DataFrame(
        data={
            'Speed':flattened_attributes, 
            'Squared_Speed':squared_speed, 
            'Cluster_Agglomerative':clusters.labels_, 
            'Cluster_Contiguity':partition.membership
        }
    )

    var_speed_by_contigs = calculate_weighted_variance(output_df.groupby('Cluster_Contiguity'))
    var_speed_by_aggloms = calculate_weighted_variance(output_df.groupby('Cluster_Agglomerative'))

    silhouette_contigs = silhouette_score(node_attributes, output_df['Cluster_Contiguity'])
    calinski_contigs = calinski_harabasz_score(node_attributes, output_df['Cluster_Contiguity'])
    davies_bouldin_contigs = davies_bouldin_score(node_attributes, output_df['Cluster_Contiguity'])

    silhouette_agglom = silhouette_score(node_attributes, output_df['Cluster_Agglomerative'])
    calinski_agglom = calinski_harabasz_score(node_attributes, output_df['Cluster_Agglomerative'])
    davies_bouldin_agglom = davies_bouldin_score(node_attributes, output_df['Cluster_Agglomerative'])

    print('-'* 100) 
    print(f'Clustering Analysis')
    print('-'* 100) 
    print(partition.summary())
    # print(f'Partition quality: {adjusted_quality}') Need to replace with the list of qualities
    print(f'Weighted speed variance of contiguity based clusters:    {var_speed_by_contigs}')
    print(f"Silhouette Score: {silhouette_contigs}")
    print(f"Calinski-Harabasz Index: {calinski_contigs}")
    print(f"Davies-Bouldin Index: {davies_bouldin_contigs}")
    print('-'* 100) 
    print(f'Agglomerative clustering on {clusters.n_clusters_} clusters: ')
    print(f'Sum of distances: {sum(clusters.distances_)}')
    print(f'Weighted speed variance of agglomerative based clusters: {var_speed_by_aggloms}')
    print(f"Silhouette Score: {silhouette_agglom}")
    print(f"Calinski-Harabasz Index: {calinski_agglom}")
    print(f"Davies-Bouldin Index: {davies_bouldin_agglom}")
    return output_df

def manually_optimize_partition(edge_matrix, node_attributes, resolution_parameter, n_iterations):
    graph = ig.Graph.DataFrame(
        edges=edge_matrix, 
        directed=False, 
    )
    kwargs = {
        'node_attributes':node_attributes,
        'resolution_parameter':resolution_parameter
    }
    partition = leidenalg.ContiguousConstrainedVertexPartition(graph, **kwargs)
    optimiser = leidenalg.Optimiser()

    historic_raw_qualities = []
    historic_qualities = []
    historic_community_weights = []
    historic_diffs = []
    move_history = []
    collapse_history = []

    itr = 0 
    diff = 0
    continue_iteration = itr < n_iterations or n_iterations < 0
    while continue_iteration:
        diff_inc = leidenalg._c_leiden._Optimiser_optimise_partition(
            optimiser._optimiser,
            partition._partition,
        )   
        partition._update_internal_membership()
        itr += 1
        raw_quality = partition.quality()
        num_clusters = len(partition)
        community_weight = num_clusters * resolution_parameter
        adjusted_quality = (raw_quality *-1) + community_weight
        diff += diff_inc

        historic_raw_qualities.append(raw_quality)
        historic_qualities.append(adjusted_quality)
        historic_community_weights.append(num_clusters)
        historic_diffs.append(diff_inc)

        # per_collapse_history_this_iteration = optimiser.get_optimization_history_per_collapse()
        # collapse_history.extend(per_collapse_history_this_iteration)
        # per_move_history_this_iteration = analyze_per_move_history(optimiser.get_optimization_history_per_move(), itr)
        # move_history.extend(per_move_history_this_iteration)

        if n_iterations < 0:
            continue_iteration = (diff_inc > 0)
        else:
            continue_iteration = itr < n_iterations

    return partition, optimiser, historic_raw_qualities, historic_qualities, historic_community_weights, historic_diffs, collapse_history, move_history

def get_shenzen_clustering_results():
    edge_map, node_attributes = parse_shenzen_info()
    edge_matrix = pd.DataFrame(columns=('Source','Target'), data=edge_map)
    possible_params = [500]
    scaler = StandardScaler()
    node_attributes = np.array(node_attributes).reshape(-1, 1).tolist()
    X = scaler.fit_transform(node_attributes).tolist()
    
    results = {}

    for param in possible_params:
        partition, optimiser, raw_quality, qualities, community_weights, diffs, collapse_history, move_history = manually_optimize_partition(
            edge_matrix=edge_matrix, 
            node_attributes=X, 
            resolution_parameter=param,
            n_iterations=3
        )
        clustering = perform_agglomerative_clustering(edge_matrix=edge_matrix, node_attributes=X, num_clusters=len(partition.sizes()))
        output_df = compare_clustering_result(X, partition=partition, clusters=clustering)
        results[param] = {
            'Partition':partition, 
            'Optimiser': optimiser,
            'Clustering':clustering, 
            'Raw_Quality':raw_quality, 
            'Adjusted_Quality':qualities, 
            'Community_Weight':community_weights, 
            'Diffs':diffs,
            'Output_DF':output_df,
            'Collapse_History':collapse_history,
            'Move_History':move_history,
        }
    
    return results

def manually_optimize_modularity(edge_matrix, node_attributes, resolution_parameter, n_iterations):
    graph = ig.Graph.DataFrame(
        edges=edge_matrix, 
        directed=False, 
    )
    partition = leidenalg.CPMVertexPartition(graph=graph)
    optimiser = leidenalg.Optimiser()

    historic_raw_qualities = []
    # historic_qualities = []
    historic_community_weights = []
    historic_diffs = []

    itr = 0 
    diff = 0
    continue_iteration = itr < n_iterations or n_iterations < 0
    while continue_iteration:
        diff_inc = leidenalg._c_leiden._Optimiser_optimise_partition(
            optimiser._optimiser,
            partition._partition,
        )   
        partition._update_internal_membership()
        itr += 1
        raw_quality = partition.quality()
        num_clusters = len(partition.sizes())
        # community_weight = num_clusters * resolution_parameter
        # adjusted_quality = (raw_quality *-1) + community_weight
        diff += diff_inc

        historic_raw_qualities.append(raw_quality)
        # historic_qualities.append(adjusted_quality)
        historic_community_weights.append(num_clusters)
        historic_diffs.append(diff_inc)

        if n_iterations < 0:
            continue_iteration = (diff_inc > 0)
        else:
            continue_iteration = itr < n_iterations

    return partition, historic_raw_qualities, historic_community_weights, historic_diffs

def get_modularity_custering_results():
    edge_map, node_attributes = parse_shenzen_info()
    edge_matrix = pd.DataFrame(columns=('Source','Target'), data=edge_map)
    possible_params = [500]
    scaler = StandardScaler()
    node_attributes = np.array(node_attributes).reshape(-1, 1).tolist()
    X = scaler.fit_transform(node_attributes).tolist()
    
    results = {}

    for param in possible_params:
        partition, raw_quality, community_weights, diffs = manually_optimize_modularity(
            edge_matrix=edge_matrix, 
            node_attributes=X, 
            resolution_parameter=param,
            n_iterations=2
        )
        clustering = perform_agglomerative_clustering(edge_matrix=edge_matrix, node_attributes=X, num_clusters=len(partition))
        results[param] = {
            'Partition':partition, 
            # 'Optimiser': optimiser,
            'Clustering':clustering, 
            'Raw_Quality':raw_quality, 
            # 'Adjusted_Quality':qualities, 
            'Community_Weight':community_weights, 
            'Diffs':diffs,
            # 'Output_DF':output_df,
            # 'Collapse_History':collapse_history,
            # 'Move_History':move_history,
        }
        compare_clustering_result(X, partition=partition, clusters=clustering)
    
    return results


get_modularity_custering_results()
