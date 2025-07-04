import os
import torch

import sys
sys.path.append(os.path.join(os.getcwd(), '..'))

import matplotlib.pyplot as plt
import numpy as np
import cvxopt
from cvxopt import matrix
import pickle
import ipdb
import cvxpy as cp


def apply_cbf(x, u_nom, neigh, r, p, alpha=1):
    cvxopt.solvers.options['show_progress'] = False
    
    dx = x[0] - neigh[:,0]
    dy = x[1] - neigh[:,1]
    neigh_vx = neigh[:,2]
    neigh_vy = neigh[:,3]


    # Vcbf = dx**2 + dy**2 - r**2
    # Wcbf = 2*dx*u_nom[0] + 2*dy*u_nom[1] - 2*dx*neigh_vx - 2*dy*neigh_vy + alpha*Vcbf**p


    P = matrix(np.eye(2))
    f = -1*u_nom
    f = matrix(f.detach().cpu().numpy().astype(np.double))

    ##### min CBF
    # G = np.zeros((1, 2))
    # h = np.zeros((1))
    # sorted_indices = torch.argsort(Wcbf)
    # min_idx = sorted_indices[0].squeeze()
    # G[0][0] = -2 * dx[min_idx]
    # G[0][1] = -2 * dy[min_idx]
    # h[0] = alpha*(dx[min_idx]**2 + dy[min_idx]**2 - r**2)**p
    # h[0] -= (2*dx[min_idx]*neigh_vx[min_idx] + 2*dy[min_idx]*neigh_vy[min_idx])

    ##### min Distance
    G = np.zeros((1, 2))
    h = np.zeros((1))
    dist = dx**2 + dy**2
    min_idx = torch.argmin(dist)
    G[0][0] = -2 * dx[min_idx]
    G[0][1] = -2 * dy[min_idx]
    h[0] = alpha*(dx[min_idx]**2 + dy[min_idx]**2 - r**2)**p
    h[0] -= (2*dx[min_idx]*neigh_vx[min_idx] + 2*dy[min_idx]*neigh_vy[min_idx])
    
    ##### min Distance
    # if len(neigh) == 1:
    #     G = np.zeros((1, 2))
    #     h = np.zeros((1))
    #     dist = dx**2 + dy**2
    #     min_idx = 0
    #     G[0][0] = -2 * dx[min_idx]
    #     G[0][1] = -2 * dy[min_idx]
    #     h[0] = alpha*(dx[min_idx]**2 + dy[min_idx]**2 - r**2)**p
    #     h[0] -= (2*dx[min_idx]*neigh_vx[min_idx] + 2*dy[min_idx]*neigh_vy[min_idx])
    # else:
    #     # G = np.zeros((1, 2))
    #     # h = np.zeros((1))
    #     G = np.zeros((2, 2))
    #     h = np.zeros((2))
    #     dist = dx**2 + dy**2
    #     min_idx = torch.argmin(dist)
    #     G[0][0] = -2 * dx[min_idx]
    #     G[0][1] = -2 * dy[min_idx]
    #     h[0] = alpha*(dx[min_idx]**2 + dy[min_idx]**2 - r**2)**p
    #     h[0] -= (2*dx[min_idx]*neigh_vx[min_idx] + 2*dy[min_idx]*neigh_vy[min_idx])

    #     dist[min_idx] = float('inf')
    #     min_idx = torch.argmin(dist)
    #     G[1][0] = -2 * dx[min_idx]
    #     G[1][1] = -2 * dy[min_idx]
    #     h[1] = alpha*(dx[min_idx]**2 + dy[min_idx]**2 - r**2)**p
    #     h[1] -= (2*dx[min_idx]*neigh_vx[min_idx] + 2*dy[min_idx]*neigh_vy[min_idx])


    override = ((G@u_nom.cpu().numpy() - h)>0).any()
    G = matrix(G)
    h = matrix(h)
    sol = cvxopt.solvers.qp(P=P,q=f,G=G,h=h)
    u = torch.tensor(np.array(sol["x"])).reshape((2))

    return u, override

def apply_cbf2(x, u_nom, neigh, r, p, alpha=1):
    cvxopt.solvers.options['show_progress'] = False
    dx = (x[0] - neigh[:,0]).cpu().numpy()
    dy = (x[1] - neigh[:,1]).cpu().numpy()
    neigh_vx = neigh[:,2].cpu().numpy()
    neigh_vy = neigh[:,3].cpu().numpy()

    P = np.eye(2)
    q = -1*u_nom.cpu().numpy()

    G = np.zeros((1, 2))
    h = np.zeros((1))
    dist = dx**2 + dy**2
    min_idx = np.argmin(dist)
    G[0][0] = -2 * dx[min_idx]
    G[0][1] = -2 * dy[min_idx]
    h[0] = alpha*(dx[min_idx]**2 + dy[min_idx]**2 - r**2)**p
    h[0] -= (2*dx[min_idx]*neigh_vx[min_idx] + 2*dy[min_idx]*neigh_vy[min_idx])
    # if len(neigh)==1:
    #     G = np.zeros((1, 2))
    #     h = np.zeros((1))
    #     dist = dx**2 + dy**2
    #     min_idx = 0
    #     G[0][0] = -2 * dx[min_idx]
    #     G[0][1] = -2 * dy[min_idx]
    #     h[0] = alpha*(dx[min_idx]**2 + dy[min_idx]**2 - r**2)**p
    #     h[0] -= (2*dx[min_idx]*neigh_vx[min_idx] + 2*dy[min_idx]*neigh_vy[min_idx])
    # else:
    #     G = np.zeros((2, 2))
    #     h = np.zeros((2))
    #     dist = dx**2 + dy**2
    #     min_idx = np.argmin(dist)
    #     G[0][0] = -2 * dx[min_idx]
    #     G[0][1] = -2 * dy[min_idx]
    #     h[0] = alpha*(dx[min_idx]**2 + dy[min_idx]**2 - r**2)**p
    #     h[0] -= (2*dx[min_idx]*neigh_vx[min_idx] + 2*dy[min_idx]*neigh_vy[min_idx])

    #     dist[min_idx] = float('inf')
    #     min_idx = np.argmin(dist)

    #     G[1][0] = -2 * dx[min_idx]
    #     G[1][1] = -2 * dy[min_idx]
    #     h[1] = alpha*(dx[min_idx]**2 + dy[min_idx]**2 - r**2)**p
    #     h[1] -= (2*dx[min_idx]*neigh_vx[min_idx] + 2*dy[min_idx]*neigh_vy[min_idx])


    x = cp.Variable(2)
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
                 [G @ x <= h])
    prob.solve()

    override = ((G@u_nom.cpu().numpy() - h)>0).any()

    return torch.tensor(x.value), override

def apply_cbf_np(x, u_nom, pos_neigh, vel_neigh, r, p):
    cvxopt.solvers.options['show_progress'] = False
    dx = x[0] - pos_neigh[:,0]
    dy = x[1] - pos_neigh[:,1]
    neigh_vx = vel_neigh[:,0]
    neigh_vy = vel_neigh[:,1]

    P = matrix(np.eye(2))
    f = -1*u_nom
    f = matrix(f)

    # if len(pos_neigh) == 1:
    G = np.zeros((1, 2))
    h = np.zeros((1))
    dist = dx**2 + dy**2
    min_idx = np.argmin(dist)
    G[0][0] = -2 * dx[min_idx]
    G[0][1] = -2 * dy[min_idx]
    h[0] = (dx[min_idx]**2 + dy[min_idx]**2 - r**2)**p
    h[0] -= (2*dx[min_idx]*neigh_vx[min_idx] + 2*dy[min_idx]*neigh_vy[min_idx])
    # else:
    #     G = np.zeros((2, 2))
    #     h = np.zeros((2))
    #     dist = dx**2 + dy**2
    #     min_idx = np.argmin(dist)
    #     G[0][0] = -2 * dx[min_idx]
    #     G[0][1] = -2 * dy[min_idx]
    #     h[0] = (dx[min_idx]**2 + dy[min_idx]**2 - r**2)**p
    #     h[0] -= (2*dx[min_idx]*neigh_vx[min_idx] + 2*dy[min_idx]*neigh_vy[min_idx])

    #     dist[min_idx] = float('inf')
    #     min_idx = np.argmin(dist)
    #     G[1][0] = -2 * dx[min_idx]
    #     G[1][1] = -2 * dy[min_idx]
    #     h[1] = (dx[min_idx]**2 + dy[min_idx]**2 - r**2)**p
    #     h[1] -= (2*dx[min_idx]*neigh_vx[min_idx] + 2*dy[min_idx]*neigh_vy[min_idx])


    override = (h.squeeze()<0).any()
    G = matrix(G)
    h = matrix(h)
    sol = cvxopt.solvers.qp(P=P,q=f,G=G,h=h)
    u = np.array(sol["x"]).reshape((2))

    return u, override

def apply_cbf_unicycle(x, u_nom, neigh, r, p, alpha=1):
    cvxopt.solvers.options['show_progress'] = False
    dx = x[0] - neigh[:,0]
    dy = x[1] - neigh[:,1]
    neigh_vx = neigh[:,2]
    neigh_vy = neigh[:,3]

    theta = x[2]

    P = matrix(np.eye(2))
    f = -1*u_nom
    f = matrix(f.detach().cpu().numpy().astype(np.double))
    G = np.zeros((1, 2))
    h = np.zeros((1))
    dist = dx**2 + dy**2
    min_idx = torch.argmin(dist)
    G[0][0] = -2 * dx[min_idx] * np.cos(theta)
    G[0][1] = -2 * dy[min_idx] * np.sin(theta)
    h[0] = alpha*(dx[min_idx]**2 + dy[min_idx]**2 - r**2)**p
    h[0] -= (2*dx[min_idx]*neigh_vx[min_idx] + 2*dy[min_idx]*neigh_vy[min_idx])

    override = (h.squeeze()<0)
    G = matrix(G)
    h = matrix(h)
    sol = cvxopt.solvers.qp(P=P,q=f,G=G,h=h)
    u = torch.tensor(np.array(sol["x"])).reshape((2))

    return u, override


def evaluate(u_gen, pos_neigh, goal):

    pos_gen = torch.zeros_like(u_gen)
    for i in range(1, u_gen.shape[2]):
        pos_gen[:,:,i] = pos_gen[:,:,i-1] + u_gen[:,:,i]*0.01

    distance = torch.sqrt(((pos_gen-pos_neigh)**2).sum(2))
    safe = torch.any(distance<0.5).int()

    
    reaching = ((pos_gen[0,:,-1] - goal.squeeze())**2).sum()
    reaching = torch.sqrt(reaching)
    smooth = torch.diff(u_gen, dim=2)**2
    smooth = smooth.sum(1)
    smooth = torch.sqrt(smooth)
    worst_smooth = smooth.max()

    directness = 0
    start = pos_gen[0,:,0]
    g = goal.squeeze()
    m1 = (start[1] - g[1]) / (start[0] - g[0])
    b1 = start[1] - m1 * start[0]
    a, b, c = -m1, 1, -b1
    for i in range(pos_gen.shape[-1]):
        x, y = pos_gen[0,0,i], pos_gen[0,1,i]
        dist = np.abs(a*x + b*y + c) / np.sqrt(a**2 + b**2)
        directness += dist

    directness /= 800

    return pos_gen, safe, reaching, worst_smooth, directness


def evaluate_ctrl(pos_gen, u_gen, pos_neigh, goal, rad):

    distance = torch.sqrt(((pos_gen-pos_neigh)**2).sum(2))
    safe = torch.any(distance<rad).int()

    goal = ((pos_gen[0,:,-1] - goal.squeeze())**2).sum()
    goal = torch.sqrt(goal)
    smooth = torch.diff(u_gen, dim=2)**2
    smooth = smooth.sum(1)
    smooth = torch.sqrt(smooth)

    worst_smooth = smooth.max()

    return pos_gen, safe, goal, worst_smooth

def plot_trajectories(trajectory_A, trajectory_B, start, goal, figname):

    plt.figure(figsize=(10, 8))

    for i in range(trajectory_A.shape[0]):
        plt.plot(trajectory_A[i, 0, :], trajectory_A[i, 1, :], label=f'Human {i+1} Trajectory')

    plt.plot(trajectory_B[0, 0, :], trajectory_B[0, 1, :], 'k', linewidth=2, label='Robot Trajectory')

    plt.scatter(start[0], start[1], s=50, label='start')
    plt.scatter(goal[0], goal[1], s=50, label='goal')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.title('Robot and Human Trajectories on the X-Y Plane')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(figname)

def fill_nan_with_last_value(tensor):
    """
    Fill NaN values in the tensor with the last non-NaN value along the last dimension.

    Parameters:
    tensor (numpy.ndarray): The input tensor of shape (10, 2, 80).

    Returns:
    numpy.ndarray: The tensor with NaN values filled.
    """
    # Iterate over the first and second dimensions
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            # Find the indices where values are NaN
            nan_indices = torch.isnan(tensor[i, j])
            if torch.any(nan_indices):
                # Find the last non-NaN value
                last_valid_index = np.where(~nan_indices)[0][-1]
                last_valid_value = tensor[i, j, last_valid_index]
                # Replace NaN values with the last non-NaN value
                tensor[i, j, nan_indices] = last_valid_value.clone()
    return tensor

def fill_nan_with_last_value_4dim(tensor):
    """
    Fill NaN values in the tensor with the last non-NaN value along the last dimension.

    Parameters:
    tensor (torch.Tensor): The input tensor of shape (10, 2, 80, 50).

    Returns:
    torch.Tensor: The tensor with NaN values filled.
    """
    # Iterate over the first three dimensions
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            for k in range(tensor.shape[2]):
                # Find the indices where values are NaN
                nan_indices = torch.isnan(tensor[i, j, k])
                if torch.any(nan_indices):
                    # Find the indices of non-NaN values
                    non_nan_indices = np.where(~nan_indices.cpu().numpy())[0]
                    if non_nan_indices.size > 0:
                        # Find the last non-NaN value
                        last_valid_index = non_nan_indices[-1]
                        last_valid_value = tensor[i, j, k, last_valid_index]
                        # Replace NaN values with the last non-NaN value
                        tensor[i, j, k, nan_indices] = last_valid_value.clone()
                    else:
                        # Handle the case where all values are NaN (e.g., set to zero or another value)
                        tensor[i, j, k, nan_indices] = 0  # or any other value you prefer
    return tensor

def load_folder_list(input_path):
    with open(input_path, 'rb') as f:
        folder_list = pickle.load(f)
    return folder_list