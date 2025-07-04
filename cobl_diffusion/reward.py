import torch

def single_cbf_reward_fn_pairwise(ego_controls, obst_states, obst_controls, r, a_cbf=1.0, k=3,):
    def f(state):
        return torch.zeros_like(state).to(state.device)
    def g(state):
        state_dim, time_steps = state.shape
        return torch.eye(state_dim).unsqueeze(2).expand(-1, -1, time_steps).to(state.device)

    def h(state, r):
        ego_state = state[:2, :]
        obst = state[2:4, :]
        x, y = ego_state[0, :], ego_state[1, :]
        x_o, y_o = obst[0, :], obst[1, :]
        return (x - x_o)**2 + (y - y_o)**2 - r**2
    
    def single_pairwise_cbf(ego_states, ego_controls, obst_states_i, obst_controls_i, r, a_cbf=1.0):
        pairwise_states = torch.cat([ego_states, obst_states_i], dim=0)
        pairwise_controls = torch.cat([ego_controls, obst_controls_i], dim=0)

        f_x = f(pairwise_states)
        g_x_u = torch.einsum('ijt,jt->it', g(pairwise_states), pairwise_controls)

        h_x = h(pairwise_states, r) # [1, time_steps]
        def h_x_sum(states_in):
            return h(states_in, r).sum()
        grad_fn = torch.func.grad(h_x_sum)
        h_dot = grad_fn(pairwise_states)
        cbf_value = torch.einsum('it,it->t', h_dot, (f_x + g_x_u)) + a_cbf * h_x
        cbf_value = torch.where(cbf_value>0, 0, cbf_value)

        return cbf_value
    
    def single_obst_fn(obst_st_i, obst_ctl_i):
        return single_pairwise_cbf(
            ego_states, ego_controls,
            obst_st_i, obst_ctl_i,
            r, a_cbf
        )  # => [time_steps]

    ego_states = torch.cumsum(ego_controls*0.1, dim=1)
    num_obst = obst_states.shape[0]

    batched_fn = torch.vmap(single_obst_fn, in_dims=(0, 0))
    cbf_constraints = batched_fn(obst_states, obst_controls)  # [n_obst, time_steps]

    k = min(k, num_obst) 
    worst_constraints, _ = torch.topk(cbf_constraints, k=k, dim=0, largest=False) # [3, time_steps]
    weights = 1 * torch.arange(1, k+1, device=worst_constraints.device).unsqueeze(1).flip(0)  # shape: [k, 1]
    worst_constraints = worst_constraints * weights

    return worst_constraints.sum()

def single_clf_reward_fn(ego_controls, goal, a_clf=0.5):
    def f(state):
        return torch.zeros_like(state).to(state.device)
    def g(state):
        state_dim, time_steps = state.shape
        return torch.eye(state_dim).unsqueeze(2).expand(-1, -1, time_steps).to(state.device)

    def V(state, goal):
        dist = torch.norm(state-goal.unsqueeze(1), dim=0)**2
        return dist
    ego_states = torch.cumsum(ego_controls*0.1, dim=1)

    f_x = f(ego_states)
    g_x_u = torch.einsum('ijt,jt->it', g(ego_states), ego_controls)

    V_x = V(ego_states, goal) # [1, time_steps]
    def V_sum(state):
        return V(state, goal).sum()
    grad_fn = torch.func.grad(V_sum)
    V_dot = grad_fn(ego_states)
    clf_value = torch.einsum('it,it->t', V_dot, (f_x + g_x_u)) + a_clf * V_x
    clf_value = - clf_value
    clf_value = torch.where(clf_value>0, 0, clf_value)
    return clf_value.sum()