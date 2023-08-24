import igraph as ig
import leidenalg
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import json

features_all=['Occupancy', 'Speed']
features_occ=['Occupancy']
features_speed=['Speed']

def load_data(city, root, demand_level):

    """
    this function loads and processes data of Barcelone
    """
    

    if city == 'barcelone':
        # Links        
        links_path = root + '/Links_updated.txt'
        links = pd.read_csv(links_path, sep= "\t", names = ['LinkID', 'Number_of_lanes', 'Length_(m)', 'StartNodeID',
                                                    'EndNodeID', 'Region'], index_col = 'LinkID')
        
        
        # Downstream
        downstream_path = root + '/downstream'
        downstream = pd.read_csv(downstream_path, delimiter = '  ', header = None, 
                         names = ['LinkID', 'Number of downstream links', 
                                  'ID of downstream link 1', 'ID of downstream link 2',
                                  'ID of downstream link 3','ID of downstream link 4',
                                  'ID of downstream link 5','ID of downstream link 6',
                                  'ID of downstream link 7',], index_col = 'LinkID') 
        
        
        # Links + Downstream (Merge links and downstream)
        l_d = pd.merge(links,downstream, on = 'LinkID')
        
        
        # Nodes
        nodes_path = root + '/Nodes.txt'
        nodes = pd.read_csv(nodes_path, sep= "\t", names = ['NodeID', 'X_coordinate', 'Y_coordinate'], index_col = 'NodeID')
        
        
        # Positions
        positions = []
        for link in links.index.tolist():
            positions.append({'LinkID':link,
                                      'StartNode_X': nodes.loc[links.loc[link,'StartNodeID'],'X_coordinate'],
                                      'StartNode_Y': nodes.loc[links.loc[link,'StartNodeID'],'Y_coordinate'],
                                      'EndNode_X' : nodes.loc[links.loc[link,'EndNodeID'], 'X_coordinate'],
                                      'EndNode_Y': nodes.loc[links.loc[link,'EndNodeID'],'Y_coordinate']})
        
        positions = pd.DataFrame(positions)
        positions.set_index('LinkID', drop = True, inplace = True)
        
        
        # ConnectedPairs
        
            ## Link index refers to the row of the link in which it appears in 'Links' and 'downstream'. 
        connectedPairs_path = root + '/connectedPairs'
        connectedPairs = pd.read_csv(connectedPairs_path, delimiter = '  ',
                             names = ['upstream_index_order', 'downstream_index_order'])
            
        connectedPairs.index = [f'connection_{i}' for i in range (1,connectedPairs.shape[0]+1)]  ## set rows index
        
        
        if demand_level == 'Med':
            # AvgOcc_med_NC
            AvgOcc_med_NC_path = root + '/medium demand/extractedFiles_AvgOcc_med_NC'
            AvgOcc_med_NC = pd.read_csv(AvgOcc_med_NC_path, delimiter = '  ', header = None)
            
                ## Define new columns index
            columns = pd.date_range(start = '00:00:00', freq = '90s', periods = 240)
            AvgOcc_med_NC.columns = pd.to_datetime(columns).strftime('%H:%M:%S')
            
                ## Set LinkID as index
            AvgOcc_med_NC.index = links.index
            
            
            # SMS_med_NC
            SMS_med_NC_path = root + '/medium demand/extractedFiles_SMS_med_NC'
            SMS_med_NC = pd.read_csv(SMS_med_NC_path, delimiter = '  ', header = None)
            
                ## Define new columns index
            SMS_med_NC.columns = pd.to_datetime(columns).strftime('%H:%M:%S')
            
                ## set index
            SMS_med_NC.index = links.index
        elif demand_level == 'High':
                        # AvgOcc_med_NC
            AvgOcc_med_NC_path = root + '/high demand/extractedFiles_AvgOcc_high_NC'
            AvgOcc_med_NC = pd.read_csv(AvgOcc_med_NC_path, delimiter = '  ', header = None)
            
                ## Define new columns index
            columns = pd.date_range(start = '00:00:00', freq = '90s', periods = 320)
            AvgOcc_med_NC.columns = pd.to_datetime(columns).strftime('%H:%M:%S')
            
                ## Set LinkID as index
            AvgOcc_med_NC.index = links.index
            
            
            # SMS_med_NC
            SMS_med_NC_path = root + '/high demand/extractedFiles_SMS_high_NC'
            SMS_med_NC = pd.read_csv(SMS_med_NC_path, delimiter = '  ', header = None)
            
                ## Define new columns index
            SMS_med_NC.columns = pd.to_datetime(columns).strftime('%H:%M:%S')
            
                ## set index
            SMS_med_NC.index = links.index
        
        
    elif city == 'shenzen':
        raise NotImplementedError
        
    return links, downstream, l_d, nodes, positions, connectedPairs, AvgOcc_med_NC, SMS_med_NC

def get_subgraphs_by_clusters(edge_matrix_df, partition):
    subgraphs = {}
    for (comm, count) in enumerate(partition.sizes()):
        subgraphs[comm] = []
        for (vertex, v_comm) in enumerate(partition.membership):
            if (comm == v_comm):
                subgraphs[comm].append(vertex)
        if (len(subgraphs[comm]) != count):
            raise Exception("Counts do not match")
    
    subgraph_edge_matrices = {}
    for (subgraph, vertices) in subgraphs.items():
        matrix = []
        for (index, vertex) in enumerate(vertices):
            edges = edge_matrix_df[edge_matrix_df['Source'] == vertex]
            for row in edges.iterrows():
                source = row[1]['Source']
                target = row[1]['Target']
                if target in vertices:
                    index_of_target = vertices.index(target)
                    if (matrix.count((index_of_target, index)) > 0):
                        continue
                    matrix.append((index, index_of_target))
        subgraph_edge_matrices[subgraph] = matrix
        
    return subgraph_edge_matrices, subgraphs

def find_disconnected_communities(results_map):
    subgraph_edges, subgraph_vertices = get_subgraphs_by_clusters(results_map['Edge_Matrix_DF'], results_map['Partition'])
    for graph_number in range(len(results_map['Partition'])):
        edge_list = subgraph_edges[graph_number]
        vertex_list = subgraph_vertices[graph_number]
        subgraph = ig.Graph(n=len(vertex_list), edges=edge_list, directed=False)
        if (not subgraph.is_connected()):
            print(f"Subgraph {graph_number} is NOT connected")

def convert_edge_matrix_to_connectivity_matrix(edge_matrix):
  num_elements = max(edge_matrix['Source'].max(), edge_matrix['Target'].max())
  matrix = np.zeros((num_elements + 1, num_elements + 1))
  for (_, row) in edge_matrix.iterrows():
    target_row = row['Source']
    target_col = row['Target']
    matrix[target_row, target_col] = 1
  return matrix

def calculate_weighted_variance(grouped, features):
    total = 0
    for feature in features:
        feature_sum = grouped[f'{feature}'].sum()
        squared_feature_sum = grouped[f'Squared_{feature}'].sum()
        weighted_variance = squared_feature_sum - ((feature_sum * feature_sum) / grouped.size())
        total += weighted_variance.sum()
    return total

def perform_agglomerative_clustering(edge_matrix, node_attributes, num_clusters):
    connectivity_matrix = convert_edge_matrix_to_connectivity_matrix(edge_matrix)
    agg_clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward', metric='euclidean', connectivity=connectivity_matrix, compute_distances=True)
    clusters = agg_clustering.fit(node_attributes)
    return clusters

def compare_and_print_clustering_result(param, node_attributes, partition, clusters: AgglomerativeClustering, features):
    np_node_attributes = np.array(node_attributes)
    squared_features = np_node_attributes ** 2
    output_df = pd.DataFrame()
    
    if 'Occupancy' in features and 'Speed' in features:
        output_df['Occupancy'] = np_node_attributes[:, 0]
        output_df['Speed'] = np_node_attributes[:, 1]
        output_df['Squared_Occupancy'] = squared_features[:, 0]
        output_df['Squared_Speed'] = squared_features[:, 1]       
    elif ('Speed' in features):
        output_df['Speed'] = np_node_attributes[:, 0]
        output_df['Squared_Speed'] = squared_features[:, 0]
    elif ('Occupancy' in features):
        output_df['Occupancy'] = np_node_attributes[:, 0]
        output_df['Squared_Occupancy'] = squared_features[:, 0]
    
    output_df['Cluster_Contiguity'] = partition.membership
    output_df['Cluster_Agglomerative'] = clusters.labels_

    var_speed_by_contigs = calculate_weighted_variance(output_df.groupby('Cluster_Contiguity'), features=features)
    var_speed_by_aggloms = calculate_weighted_variance(output_df.groupby('Cluster_Agglomerative'), features=features)

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
    print(f'Partition quality: {-(partition.quality() + (param * len(partition)))}')
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

def compare_and_return_clustering_result(node_attributes, partition, clusters: AgglomerativeClustering, features):
    np_node_attributes = np.array(node_attributes)
    squared_features = np_node_attributes ** 2
    output_df = pd.DataFrame()
    
    if 'Occupancy' in features and 'Speed' in features:
        output_df['Occupancy'] = np_node_attributes[:, 0]
        output_df['Speed'] = np_node_attributes[:, 1]
        output_df['Squared_Occupancy'] = squared_features[:, 0]
        output_df['Squared_Speed'] = squared_features[:, 1]       
    elif ('Speed' in features):
        output_df['Speed'] = np_node_attributes[:, 0]
        output_df['Squared_Speed'] = squared_features[:, 0]
    elif ('Occupancy' in features):
        output_df['Occupancy'] = np_node_attributes[:, 0]
        output_df['Squared_Occupancy'] = squared_features[:, 0]
    
    output_df['Cluster_Contiguity'] = partition.membership
    output_df['Cluster_Agglomerative'] = clusters.labels_

    var_speed_by_contigs = calculate_weighted_variance(output_df.groupby('Cluster_Contiguity'), features=features)
    var_speed_by_aggloms = calculate_weighted_variance(output_df.groupby('Cluster_Agglomerative'), features=features)

    silhouette_contigs = silhouette_score(node_attributes, output_df['Cluster_Contiguity'])
    calinski_contigs = calinski_harabasz_score(node_attributes, output_df['Cluster_Contiguity'])
    davies_bouldin_contigs = davies_bouldin_score(node_attributes, output_df['Cluster_Contiguity'])

    silhouette_agglom = silhouette_score(node_attributes, output_df['Cluster_Agglomerative'])
    calinski_agglom = calinski_harabasz_score(node_attributes, output_df['Cluster_Agglomerative'])
    davies_bouldin_agglom = davies_bouldin_score(node_attributes, output_df['Cluster_Agglomerative'])

    contig_results = {
        'weighted_variance':var_speed_by_contigs,
        'silhouette': silhouette_contigs,
        'calinksi': calinski_contigs,
        'davies': davies_bouldin_contigs,
    }
    agglom_results = {
        'weighted_variance':var_speed_by_aggloms,
        'silhouette': silhouette_agglom,
        'calinksi': calinski_agglom,
        'davies': davies_bouldin_agglom,
    }

    return contig_results, agglom_results

def manually_optimize_partition(edge_matrix, node_attributes, resolution_parameter, disconnect_penalty, n_iterations):
    graph = ig.Graph.DataFrame(
        edges=edge_matrix, 
        directed=False, 
    )
    kwargs = {
        'node_attributes':node_attributes,
        'resolution_parameter':resolution_parameter,
        'disconnect_penalty':disconnect_penalty,
    }
    partition = leidenalg.ContiguousConstrainedVertexPartition(graph, **kwargs)
    optimiser = leidenalg.Optimiser()
    optimiser.set_rng_seed(0)

    historic_raw_qualities = []
    historic_qualities = []
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
        num_clusters = len(partition)
        community_weight = num_clusters * resolution_parameter
        adjusted_quality = (raw_quality *-1) + community_weight
        diff += diff_inc

        historic_raw_qualities.append(raw_quality)
        historic_qualities.append(adjusted_quality)
        historic_community_weights.append(num_clusters)
        historic_diffs.append(diff_inc)

        if n_iterations < 0:
            continue_iteration = (diff_inc > 0)
        else:
            continue_iteration = itr < n_iterations

    return partition, optimiser, historic_raw_qualities, historic_qualities, historic_community_weights, historic_diffs

def manually_optimize_partition_toward_cluster_size(graph, node_attributes, resolution_parameter, disconnect_penalty, seed, target_num_clusters, decay=.5):
    kwargs = {
        'node_attributes':node_attributes,
        'resolution_parameter':resolution_parameter,
        'disconnect_penalty':disconnect_penalty,
    }
    partition = leidenalg.ContiguousConstrainedVertexPartition(graph, **kwargs)
    optimiser = leidenalg.Optimiser()
    optimiser.set_rng_seed(seed)

    def do_optimise(partition, optimiser):
        continue_iteration = True
        while continue_iteration:
            diff_inc = leidenalg._c_leiden._Optimiser_optimise_partition(
                optimiser._optimiser,
                partition._partition,
            )   
            partition._update_internal_membership()
            continue_iteration = (diff_inc > 0)
    print('-'* 100) 
    print(f"Optimizing seed {seed}")
    print('-'* 100)
    while(len(partition) != target_num_clusters):
        print(f"\tTesting resolution parameter {kwargs['resolution_parameter']}")
        do_optimise(partition=partition, optimiser=optimiser)
        if (len(partition) < target_num_clusters):
            kwargs['resolution_parameter'] = round(kwargs['resolution_parameter'] - decay, 2)
            partition = leidenalg.ContiguousConstrainedVertexPartition(graph, **kwargs)
        elif (len(partition) > target_num_clusters):
            kwargs['resolution_parameter'] = round(kwargs['resolution_parameter'] + decay, 2)
            partition = leidenalg.ContiguousConstrainedVertexPartition(graph, **kwargs)
    print('-'* 100) 
    return partition

def get_clustering_results_over_whole_timespace(target_num_clusters, demand_level, starting_parameter):
    _, _, _, _, _, connectedPairs, AvgOcc_med_NC, SMS_med_NC = load_data('barcelone', root='./Barcelona/Barcelone_data/', demand_level=demand_level)
    edge_matrix = pd.DataFrame()
    edge_matrix['Source'] = connectedPairs['upstream_index_order'].astype(int) - 1
    edge_matrix['Target'] = connectedPairs['downstream_index_order'].astype(int) - 1
    scaler = StandardScaler()

    graph = ig.Graph.DataFrame(
        edges=edge_matrix, 
        directed=False, 
    )

    results = []
    step = 10
    for index in range(0, len(AvgOcc_med_NC.columns), step):
        print('-'*100)
        print(f"CLUSTERING demand level {demand_level} time slice {AvgOcc_med_NC.columns[index]}")
        print('-'*100)
        cols = AvgOcc_med_NC.columns[index: index + step]
        occ = AvgOcc_med_NC[cols].mean(axis=1)
        speed = SMS_med_NC[cols].mean(axis=1)
        node_attributes = np.column_stack((occ.to_numpy(), speed.to_numpy()))
        X = scaler.fit_transform(node_attributes).tolist()
        partition = manually_optimize_partition_toward_cluster_size(
            graph=graph, 
            node_attributes=X, 
            resolution_parameter=starting_parameter,
            disconnect_penalty=0,
            seed=0,
            target_num_clusters=target_num_clusters,
        )
        results.append({
            'time':cols[0],
            'membership':partition.membership,
            'quality':-(partition.quality() + (partition.resolution_parameter * len(partition)))
        })
        print(f"CLUSTERING DONE QUALITY: {results[-1]['quality']}")

    with open(file=f'./outputs/barcelona_over_time_{demand_level}_{target_num_clusters}_clusters.json', mode='w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        


def get_clustering_results_over_mean_of_timespace(target_num_clusters, starting_parameter, features, demand_level, decay=.5):
    _, _, _, _, _, connectedPairs, AvgOcc_med_NC, SMS_med_NC = load_data('barcelone', root='./Barcelona/Barcelone_data/', demand_level=demand_level)
    edge_matrix = pd.DataFrame()
    edge_matrix['Source'] = connectedPairs['upstream_index_order'].astype(int) - 1
    edge_matrix['Target'] = connectedPairs['downstream_index_order'].astype(int) - 1
    node_attributes = []
    scaler = StandardScaler()

    occ = AvgOcc_med_NC.mean(axis=1)
    speed = SMS_med_NC.mean(axis=1)

    if 'Occupancy' in features and 'Speed' in features:
        node_attributes = np.column_stack((occ.to_numpy(), speed.to_numpy()))
    elif ('Occupancy' in features):
        node_attributes = occ.to_numpy().reshape(-1, 1)
    elif ('Speed' in features):
        node_attributes = speed.to_numpy().reshape(-1, 1)
    
    X = scaler.fit_transform(node_attributes).tolist()

    seeds = range(0, 30)

    contig_results = {
        'weighted_variance':[],
        'silhouette':[],
        'calinksi':[],
        'davies':[],
    }
    agglom_results = {
        'weighted_variance':[],
        'silhouette':[],
        'calinksi':[],
        'davies':[],
    }

    graph = ig.Graph.DataFrame(
        edges=edge_matrix, 
        directed=False, 
    )

    for (seed) in seeds:
        partition = manually_optimize_partition_toward_cluster_size(
            graph=graph, 
            node_attributes=X, 
            resolution_parameter=starting_parameter,
            disconnect_penalty=0,
            seed=seed,
            target_num_clusters=target_num_clusters,
            decay=decay
        )
        clustering = perform_agglomerative_clustering(edge_matrix=edge_matrix, node_attributes=X, num_clusters=target_num_clusters)
        contigs, aggloms = compare_and_return_clustering_result(X, partition=partition, clusters=clustering, features=features)
        for (key, value) in contigs.items():
            contig_results[key].append(value)
            agglom_results[key].append(aggloms[key])

    print('-'* 100) 
    print(f'Mean of {len(seeds)} Clustering Analyses on {len(partition)} Clusters on means of raw data')
    print(f"Demand level {demand_level}. Features {features}")
    print('-'* 100) 
    print(f"Mean of Partition qualities: {sum(contig_results['weighted_variance']) / len(seeds)}")
    print(f"Mean of Silhouette Scores: {sum(contig_results['silhouette']) / len(seeds)}")
    print(f"Mean of Calinski-Harabasz Index: {sum(contig_results['calinksi']) / len(seeds)}")
    print(f"Mean of Davies-Bouldin Index: {sum(contig_results['davies']) / len(seeds)}")
    print("-"* 100) 
    print(f"Agglomerative clustering on {clustering.n_clusters_} clusters: ")
    print(f"Weighted speed variance of agglomerative based clusters: {sum(agglom_results['weighted_variance']) / len(seeds)}")
    print(f"Silhouette Score: {sum(agglom_results['silhouette']) / len(seeds)}")
    print(f"Calinski-Harabasz Index: {sum(agglom_results['calinksi']) / len(seeds)}")
    print(f"Davies-Bouldin Index: {sum(agglom_results['davies']) / len(seeds)}")
    print("-"* 100) 

    return contig_results, agglom_results


def get_clustering_results_over_seeds(target_num_clusters, starting_parameter, features, demand_level, time_slice, decay):
    _, _, _, _, _, connectedPairs, AvgOcc_med_NC, SMS_med_NC = load_data('barcelone', root='./Barcelona/Barcelone_data/', demand_level=demand_level)
    edge_matrix = pd.DataFrame()
    edge_matrix['Source'] = connectedPairs['upstream_index_order'].astype(int) - 1
    edge_matrix['Target'] = connectedPairs['downstream_index_order'].astype(int) - 1
    node_attributes = []
    scaler = StandardScaler()

    if 'Occupancy' in features and 'Speed' in features:
        node_attributes = np.column_stack((AvgOcc_med_NC[time_slice].to_numpy(), SMS_med_NC[time_slice].to_numpy()))
    elif ('Occupancy' in features):
        node_attributes = AvgOcc_med_NC[time_slice].to_numpy().reshape(-1, 1)
    elif ('Speed' in features):
        node_attributes = scaler.fit_transform(SMS_med_NC[time_slice].to_numpy().reshape(-1, 1))
    
    X = scaler.fit_transform(node_attributes).tolist()

    seeds = range(0, 1)

    contig_results = {
        'weighted_variance':[],
        'silhouette':[],
        'calinksi':[],
        'davies':[],
    }
    agglom_results = {
        'weighted_variance':[],
        'silhouette':[],
        'calinksi':[],
        'davies':[],
    }

    graph = ig.Graph.DataFrame(
        edges=edge_matrix, 
        directed=False, 
    )

    for (seed) in seeds:
        partition = manually_optimize_partition_toward_cluster_size(
            graph=graph, 
            node_attributes=X, 
            resolution_parameter=starting_parameter,
            disconnect_penalty=0,
            seed=seed,
            target_num_clusters=target_num_clusters,
            decay=decay
        )
        clustering = perform_agglomerative_clustering(edge_matrix=edge_matrix, node_attributes=X, num_clusters=target_num_clusters)
        contigs, aggloms = compare_and_return_clustering_result(X, partition=partition, clusters=clustering, features=features)
        for (key, value) in contigs.items():
            contig_results[key].append(value)
            agglom_results[key].append(aggloms[key])

    print('-'* 100) 
    print(f'Mean of {len(seeds)} Clustering Analyses on {len(partition)} Clusters on specific time slice of raw data')
    print(f"Demand level {demand_level}. Features {features}")
    print('-'* 100) 
    print(f"Mean of Partition qualities: {sum(contig_results['weighted_variance']) / len(seeds)}")
    print(f"Mean of Silhouette Scores: {sum(contig_results['silhouette']) / len(seeds)}")
    print(f"Mean of Calinski-Harabasz Index: {sum(contig_results['calinksi']) / len(seeds)}")
    print(f"Mean of Davies-Bouldin Index: {sum(contig_results['davies']) / len(seeds)}")
    print("-"* 100) 
    print(f"Agglomerative clustering on {clustering.n_clusters_} clusters: ")
    print(f"Weighted speed variance of agglomerative based clusters: {sum(agglom_results['weighted_variance']) / len(seeds)}")
    print(f"Silhouette Score: {sum(agglom_results['silhouette']) / len(seeds)}")
    print(f"Calinski-Harabasz Index: {sum(agglom_results['calinksi']) / len(seeds)}")
    print(f"Davies-Bouldin Index: {sum(agglom_results['davies']) / len(seeds)}")
    print("-"* 100) 

    return contig_results, agglom_results

def get_specific_clustering_result(possible_params, features, demand_level, time_slice):
    _, _, _, _, _, connectedPairs, AvgOcc_med_NC, SMS_med_NC = load_data('barcelone', root='./Barcelona/Barcelone_data/', demand_level=demand_level)
    edge_matrix = pd.DataFrame()
    edge_matrix['Source'] = connectedPairs['upstream_index_order'].astype(int) - 1
    edge_matrix['Target'] = connectedPairs['downstream_index_order'].astype(int) - 1
    node_attributes = []
    scaler = StandardScaler()
    if 'Occupancy' in features and 'Speed' in features:
        node_attributes = np.column_stack((AvgOcc_med_NC[time_slice].to_numpy(), SMS_med_NC[time_slice].to_numpy()))
    elif ('Occupancy' in features):
        node_attributes = AvgOcc_med_NC[time_slice].to_numpy().reshape(-1, 1)
    elif ('Speed' in features):
        node_attributes = SMS_med_NC[time_slice].to_numpy().reshape(-1, 1)
    
    X = scaler.fit_transform(node_attributes).tolist()
    
    results = {}

    for param in possible_params:
        partition, optimiser, raw_quality, qualities, community_weights, diffs = manually_optimize_partition(
            edge_matrix=edge_matrix, 
            node_attributes=X, 
            resolution_parameter=param,
            disconnect_penalty=0,
            n_iterations=-1
        )
        clustering = perform_agglomerative_clustering(edge_matrix=edge_matrix, node_attributes=X, num_clusters=len(partition))
        output_df = compare_and_print_clustering_result(param, X, partition=partition, clusters=clustering, features=features)
        results[param] = {
            'Partition':partition, 
            'Optimiser': optimiser,
            'Clustering':clustering, 
            'Raw_Quality':raw_quality, 
            'Adjusted_Quality':qualities, 
            'Community_Weight':community_weights, 
            'Diffs':diffs,
            'Output_DF':output_df,
            'Edge_Matrix_DF':edge_matrix
        }
    
    return results

def benchmark_all_features_all_means():
    results = {}
    
    print('-'*100)
    print(f"BEGIN: MED SCENARIO: FEATURES = {features_all}")
    c1, a1 = get_clustering_results_over_mean_of_timespace(target_num_clusters=5, starting_parameter=40, features=features_all, demand_level='Med')
    results['med_all'] = {'contiguity':c1, 'agglomerative':a1}

    print('-'*100)
    print(f"BEGIN: MED SCENARIO: FEATURES = {features_occ}")
    c2, a2 = get_clustering_results_over_mean_of_timespace(target_num_clusters=5, starting_parameter=30, features=features_occ, demand_level='Med')
    results['med_occ'] = {'contiguity':c2, 'agglomerative':a2}

    print('-'*100)
    print(f"BEGIN: MED SCENARIO: FEATURES = {features_speed}")
    c3, a3 = get_clustering_results_over_mean_of_timespace(target_num_clusters=5, starting_parameter=25, features=features_speed, demand_level='Med')
    results['med_speed'] = {'contiguity':c3, 'agglomerative':a3}

    print('-'*100)
    print(f"BEGIN: HIGH SCENARIO: FEATURES = {features_all}")
    c4, a4 = get_clustering_results_over_mean_of_timespace(target_num_clusters=5, starting_parameter=45, features=features_all, demand_level='High')
    results['high_all'] = {'contiguity':c4, 'agglomerative':a4}

    print('-'*100)
    print(f"BEGIN: HIGH SCENARIO: FEATURES = {features_occ}")
    c5, a5 = get_clustering_results_over_mean_of_timespace(target_num_clusters=5, starting_parameter=26, features=features_occ, demand_level='High')
    results['high_occ'] = {'contiguity':c5, 'agglomerative':a5}

    print('-'*100)
    print(f"BEGIN: HIGH SCENARIO: FEATURES = {features_speed}")
    c6, a6 = get_clustering_results_over_mean_of_timespace(target_num_clusters=5, starting_parameter=15, features=features_speed, demand_level='High')
    results['high_speed'] = {'contiguity':c6, 'agglomerative':a6}

    with open(file=f'./outputs/barcelona_5_clusters_means.json', mode='w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def benchmark_all_features_time_slices():
    print('-'*100)
    print(f"BEGIN: MED SCENARIO: FEATURES = {features_all}")
    get_clustering_results_over_seeds(target_num_clusters=10, starting_parameter=40, features=features_all, demand_level='Med', time_slice='02:58:30')

    print('-'*100)
    print(f"BEGIN: MED SCENARIO: FEATURES = {features_occ}")
    get_clustering_results_over_seeds(target_num_clusters=10, starting_parameter=20, features=features_occ, demand_level='Med', time_slice='02:58:30')

    print('-'*100)
    print(f"BEGIN: MED SCENARIO: FEATURES = {features_speed}")
    get_clustering_results_over_seeds(target_num_clusters=10, starting_parameter=18, features=features_speed, demand_level='Med', time_slice='02:58:30')

    print('-'*100)
    print(f"BEGIN: HIGH SCENARIO: FEATURES = {features_all}")
    get_clustering_results_over_seeds(target_num_clusters=10, starting_parameter=30, features=features_all, demand_level='High', time_slice='04:00:00')

    print('-'*100)
    print(f"BEGIN: HIGH SCENARIO: FEATURES = {features_occ}")
    get_clustering_results_over_seeds(target_num_clusters=10, starting_parameter=15, features=features_occ, demand_level='High', time_slice='04:00:00')

    print('-'*100)
    print(f"BEGIN: HIGH SCENARIO: FEATURES = {features_speed}")
    get_clustering_results_over_seeds(target_num_clusters=10, starting_parameter=12, features=features_speed, demand_level='High', time_slice='04:00:00')

def benchmark_single_feature(time):
    params = [49]
    clustering_result = get_specific_clustering_result(params, features_all, demand_level='Med', time_slice=time)
    results_map = clustering_result[params[0]]
    find_disconnected_communities(results_map)

if __name__ == "__main__":
    # benchmark_all_features_all_means()
    benchmark_single_feature('02:58:30')    
