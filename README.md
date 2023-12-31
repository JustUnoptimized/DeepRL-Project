## Repo for Deep Reinforcement Learning Project
This repo contains the code for my Deep Reinforcement Project: An Exploration of Q-Learning using Normalized Advantage Functions and Model-Based Accelerations in Various Different Domains

The `runner.py` file will handle all putting together the different pieces to run the various algorithms. **This requires the exp_name flag first!** All results will be saved in `evals/exp_name`. The `test.sh` contains examples of how to run the code from the command line.

The code for this project uses mainly uses:
- [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)
- [An implementation of NAF](https://github.com/BY571/Normalized-Advantage-Function-NAF-) by Dittert, Sebastian. Code is in the [naf/](naf/) directory.
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- [Simglucose](https://github.com/jxx123/simglucose)

Modifications:
- I modified my pip installation of Stable Baselines 3 such that the TD3 MLP policy includes BatchNorms after each hidden layer. The modifications are listed below, and are **not** reflected in this repo.
- I modified the NAF implementation to add iLQG and imagination rollouts for model-based accelerations. The modifications are reflected in the repo.

#### SB3 Modification Instructions
1. Add the following code snippet to `stable_baselines3/common/torch_layers.py`:

```
def create_mlp_bn(
    input_dim: int,
    output_dim: int,
    net_arch; List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
    with_bias: bool = True
) -> List[nn.Module]:
    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0], bias=with_bias),
                   nn.BatchNorm1d(net_arch[0]),
                   activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias))
        modules.append(nn.BatchNorm1d(net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))
    
    if squash_output:
        modules.append(nn.TanH())

    return modules
```

2. Import the above function in `stable_baselines3/common/policies.py` and `stable_baselines3/td3/policies.py`

3. Inside `stable_baselines3/td3/policies.py`, add `nobn: bool = False,` as an argument to `TD3Policy.__init__()`. Likewise, add `"nobn": nobn,` to the `self.net_args` dictionary a few lines down.

4. Inside `stable_baselines3/td3/policies.py`, add `nobn: bool = False,` as an argument to `Actor.__init__()`. Then, replace `actor_net = create_mlp(features_dim, action_dim, net_arch, activation_fn, squash_output=True)` with the following if-else block:

```
if nobn:
    actor_net = create_mlp(features_dim, action_dim, net_arch, activation_fn, squash_output=True)
else:
    actor_net = create_mlp_bn(features_dim, action_dim, net_arch, activation_fn, squash_output=True)
```

5. Inside `stable_baselines3/common/policies.py`, add `nobn: bool = False,` as an argument to `ContinuousCritic.__init__()`. Then, inside the `for idx in range(n_critics)` loop, replace `q_net_list = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)` with the following if-else block:

```
if nobn:
    q_net_list = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
else:
    q_net_list = create_mlp_bn(features_dim + action_dim, 1, net_arch, activation_fn)
```