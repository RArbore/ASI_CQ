import numpy as np
import torch
import time
from tqdm import tqdm
import statistics

current_milli_time = lambda: int(round(time.time() * 1000))

def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)

    # ORIGINAL METHOD FOR CREATING initial_state:
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]

    # CREATING initial_state DETERMINISTICALLY INSTEAD:
    # initial_state = []
    # index = -1
    # while len(initial_state) < num_clusters:
    #     index += 1
    #     if X[index].tolist() in initial_state:
    #         continue
    #     initial_state.append(X[index].tolist())
    # initial_state = torch.tensor(initial_state)

    return initial_state


def kmeans(
        fast_kmeans,
        X,
        imagePalette,
        frequencyTensor,
        num_clusters,
        image,
        tol=0,
        iteration_limit=1000,
        device=torch.device('cpu')
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param tol: (float) threshold [default: 0]
    :param iteration_limit: (int) iteration limit [default: 1,000]
    :param device: (torch.device) device [default: cpu]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    print(f'---------------------------------------------------------------------------------------------------------------------\nrunning k-means for image {image} on {device}..')


    # convert to float
    if fast_kmeans:
        X = imagePalette.float()
    else:
        X = X.float()

    # transfer to device
    X = X.to(device)

    # initialize
    initial_state = initialize(X, num_clusters)
    # According to the test below, the initial_state always only picks colors from the actual image to start
    # print(check(X, initial_state))
    # import pdb; pdb.set_trace()

    iteration = 0
    tqdm_meter = tqdm(desc='[running kmeans]')

    times = []

    while True:

        start_time = current_milli_time()

        dis = pairwise_distance(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1) #A size (65536) tensor with the index number of the closest cluster for each pixel

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected_indecies = torch.nonzero(choice_cluster == index).squeeze().to(device) #The indecies of all the pixels that are closest to cluster int(index)

            selected = torch.index_select(X, 0, selected_indecies) #Just X[n][:] for n = all the values in selected.  Returns tensor with only the pixels that are closes to cluster int(index)
            if fast_kmeans:
                selected_frequencies = torch.index_select(frequencyTensor, 0, selected_indecies)
                initial_state[index] = (selected*selected_frequencies.reshape(-1,1)).mean(dim=0)*selected.size(0)/torch.sum(selected_frequencies)
            else:
                initial_state[index] = selected.mean(dim=0)
        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        tqdm_meter.set_postfix(
            iteration=f'{iteration}',
            center_shift=f'{center_shift ** 2:0.6f}',
            iteration_limit=f'{iteration_limit}',
            tolerance=f'{tol}'
        )
        tqdm_meter.update()

        times.append(current_milli_time() - start_time)

        #Added iteration limit
        if center_shift ** 2 <= tol or iteration > iteration_limit:
            horizontal_bar = '\n-----------------------------------'
            print(horizontal_bar+'\nIteration time statistics (ms/it):\nmean: %s\nmedian: %s' % (str(statistics.mean(times)), str(statistics.median(times)))+horizontal_bar)
            break

    return choice_cluster.cpu(), initial_state.cpu()


def kmeans_predict(
        X,
        cluster_centers,
        distance='euclidean',
        device=torch.device('cpu')
):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :return: (torch.tensor) cluster ids
    """
    print(f'predicting on {device}..')

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    dis = pairwise_distance(X, cluster_centers)
    choice_cluster = torch.argmin(dis, dim=1)

    return choice_cluster.cpu()


def pairwise_distance(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis
