import torch
import torch.nn as nn 
from torch.distributions import MultivariateNormal
from gymnasium import spaces

# def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
# adapted from stable baselines 3 source code
def unscale_action(scaled_action, action_space):
    """
    Rescale the action from [-1, 1] to [low, high]
    (no need for symmetric action space)

    :param scaled_action: Action to un-scale
    """
    assert isinstance(
        action_space, spaces.Box
    ), f"Trying to unscale an action using an action space that is not a Box(): {action_space}"
    low, high = action_space.low, action_space.high
    _device = scaled_action.device
    low = torch.Tensor(low).to(_device)
    high = torch.Tensor(high).to(_device)
    return low + (0.5 * (scaled_action + 1.0) * (high - low))


class DeepNAF(nn.Module):
    def __init__(self, state_size, action_size, action_space, layer_size, seed):
        super(DeepNAF, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
        self.action_space = action_space
        
        concat_size = state_size+layer_size

        self.fc1 = nn.Linear(self.input_shape, layer_size)
        self.bn1 = nn.BatchNorm1d(layer_size)
        self.fc2 = nn.Linear(concat_size, layer_size)
        self.bn2 = nn.BatchNorm1d(layer_size)
        self.fc3 = nn.Linear(concat_size, layer_size)
        self.bn3 = nn.BatchNorm1d(layer_size)
        self.fc4 = nn.Linear(concat_size, layer_size)
        self.bn4 = nn.BatchNorm1d(layer_size)
        self.action_values = nn.Linear(layer_size, action_size)
        self.value = nn.Linear(layer_size, 1)
        self.matrix_entries = nn.Linear(layer_size, int(self.action_size*(self.action_size+1)/2))
    
    def forward(self, input_, action=None):
        """
        
        """

        x = torch.relu(self.fc1(input_))
        x = self.bn1(x)
        x = torch.relu(self.fc2(torch.cat([x, input_], dim=1)))
        x = self.bn2(x)
        x = torch.relu(self.fc3(torch.cat([x, input_], dim=1)))
        x = self.bn3(x)
        x = torch.relu(self.fc4(torch.cat([x, input_], dim=1)))

        action_value = torch.tanh(self.action_values(x))
        entries = torch.tanh(self.matrix_entries(x))
        V = self.value(x)
        
        action_value = action_value.unsqueeze(-1)
        
        # create lower-triangular matrix
        L = torch.zeros((input_.shape[0], self.action_size, self.action_size)).to(input_.device)

        # get lower triagular indices
        tril_indices = torch.tril_indices(row=self.action_size, col=self.action_size, offset=0)  

        # fill matrix with entries
        L[:, tril_indices[0], tril_indices[1]] = entries
        L.diagonal(dim1=1,dim2=2).exp_()

        # calculate state-dependent, positive-definite square matrix
        P = L*L.transpose(2, 1)
        
        Q = None
        if action is not None:  

            # calculate Advantage:
            A = (-0.5 * torch.matmul(torch.matmul((action.unsqueeze(-1) - action_value).transpose(2, 1), P), (action.unsqueeze(-1) - action_value))).squeeze(-1)

            Q = A + V   
        
        
        # add noise to action mu:
        dist = MultivariateNormal(action_value.squeeze(-1), torch.inverse(P))
        action = dist.sample()
        action = torch.clamp(action, min=-1, max=1)
        
        # unscale actions so it fits the action space
        action_value = unscale_action(action_value, self.action_space)
        action = unscale_action(action, self.action_space)
        
        return action, Q, V, action_value 

class NAF(nn.Module):
    def __init__(self, state_size, action_size, action_space, layer_size, seed):
        super(NAF, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
        self.action_space = action_space
                
        self.head_1 = nn.Linear(self.input_shape, layer_size)
        self.bn1 = nn.BatchNorm1d(layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.bn2 = nn.BatchNorm1d(layer_size)
        self.action_values = nn.Linear(layer_size, action_size)
        self.value = nn.Linear(layer_size, 1)
        self.matrix_entries = nn.Linear(layer_size, int(self.action_size*(self.action_size+1)/2))
        

    
    def forward(self, input_, action=None):
        """
        
        """

        x = torch.relu(self.head_1(input_))
        x = self.bn1(x)
        x = torch.relu(self.ff_1(x))
        x = self.bn2(x)
        action_value = torch.tanh(self.action_values(x))
        entries = torch.tanh(self.matrix_entries(x))
        V = self.value(x)
        
        action_value = action_value.unsqueeze(-1)
        
        # create lower-triangular matrix
        L = torch.zeros((input_.shape[0], self.action_size, self.action_size)).to(input_.device)

        # get lower triagular indices
        tril_indices = torch.tril_indices(row=self.action_size, col=self.action_size, offset=0)  

        # fill matrix with entries
        L[:, tril_indices[0], tril_indices[1]] = entries
        L.diagonal(dim1=1,dim2=2).exp_()

        # calculate state-dependent, positive-definite square matrix
        P = L*L.transpose(2, 1)
        
        Q = None
        if action is not None:  

            # calculate Advantage:
            A = (-0.5 * torch.matmul(torch.matmul((action.unsqueeze(-1) - action_value).transpose(2, 1), P), (action.unsqueeze(-1) - action_value))).squeeze(-1)

            Q = A + V   
        
        # add noise to action mu:
        dist = MultivariateNormal(action_value.squeeze(-1), torch.inverse(P))
        action = dist.sample()
        action = torch.clamp(action, min=-1, max=1)
        
        # unscale actions so it fits the action space
        action_value = unscale_action(action_value, self.action_space)
        action = unscale_action(action, self.action_space)
        
        return action, Q, V, action_value