import numpy as np
import torch
from tqdm import tqdm

def checkDuplicates(initial_state): # Returns a list of the number of times x appears in initial_state where x is all the values in initial_state.  If no values are duplicated, returns [1,1,1, ... ,1,1]
    return [len(torch.where((initial_state==torch.Tensor(state)).all(dim=1))[0]) for state in initial_state]

def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    initial_state = torch.zeros(num_clusters, 3)
    #Added while loop to the original kmeans package code so that it never picks an initial state with duplicate datapoints.
    whileIteration = 0
    while sum(checkDuplicates(initial_state)) > num_clusters:
        indices = np.random.choice(num_samples, num_clusters, replace=False)
        initial_state = X[indices]
        whileIteration += 1
        if whileIteration > 10000:
            if input('Failed to find valid initial palette after 10000 attempts. Debugger?(y/n)') =='y':
                import pdb; pdb.set_trace()
            whileIteration = 0
    return initial_state


def kmeans(
        X,
        num_clusters,
        image,
        distance='euclidean',
        tol=0,
        iteration_limit=1000,
        device=torch.device('cpu')
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0]
    :param iteration_limit: (int) iteration limit [default: 1,000]
    :param device: (torch.device) device [default: cpu]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    print(f'running k-means for image {image} on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
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
    while True:
        dis = pairwise_distance_function(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1) #A size (65536) tensor with the index number of the closest cluster for each pixel

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device) #The indecies of all the pixels that are closest to cluster int(index)

            selected = torch.index_select(X, 0, selected) #Just X[n][:] for n = all the values in selected.  Returns tensor with only the pixels that are closes to cluster int(index)

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
        #Added iteration limit
        if center_shift ** 2 <= tol or iteration > iteration_limit:
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

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    dis = pairwise_distance_function(X, cluster_centers)
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


def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis

