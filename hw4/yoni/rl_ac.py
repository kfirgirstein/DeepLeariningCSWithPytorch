import gym
import torch
import torch.nn as nn
import torch.nn.functional

from .rl_pg import PolicyAgent, TrainBatch, VanillaPolicyGradientLoss


class AACPolicyNet(nn.Module):
    def __init__(self, in_features: int, out_actions: int, **kw):
        """
        Create a model which represents the agent's policy.
        :param in_features: Number of input features (in one observation).
        :param out_actions: Number of output actions.
        :param kw: Any extra args needed to construct the model.
        """
        super().__init__()
        # TODO:
        #  Implement a dual-head neural net to approximate both the
        #  policy and value. You can have a common base part, or not.
        # ====== YOUR CODE: ======
        modules = []
        modules.append(nn.Linear(in_features, 16))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(16, 16))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(16, 16))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(16, out_actions))
        self.actor = nn.Sequential(*modules)
        
        modules = []
        modules.append(nn.Linear(in_features, 16))
        modules.append(nn.Dropout(p=0.5))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(16, 8))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(8, 1))
        self.critic = nn.Sequential(*modules)		
        # ========================

    def forward(self, x):
        """
        :param x: Batch of states, shape (N,O) where N is batch size and O
        is the observation dimension (features).
        :return: A tuple of action values (N,A) and state values (N,1) where
        A is is the number of possible actions.
        """
        # TODO:
        #  Implement the forward pass.
        #  calculate both the action scores (policy) and the value of the
        #  given state.
        # ====== YOUR CODE: ======
        action_scores = self.actor(x)
        state_values = self.critic(x)
        # ========================

        return action_scores, state_values

    @staticmethod
    def build_for_env(env: gym.Env, device='cpu', **kw):
        """
        Creates a A2cNet instance suitable for the given environment.
        :param env: The environment.
        :param kw: Extra hyperparameters.
        :return: An A2CPolicyNet instance.
        """
        # TODO: Implement according to docstring.
        # ====== YOUR CODE: ======
        net = AACPolicyNet(env.observation_space.shape[0], env.action_space.n, **kw)
        # ========================
        return net.to(device)


class AACPolicyAgent(PolicyAgent):

    def current_action_distribution(self) -> torch.Tensor:
        # TODO: Generate the distribution as described above.
        # ====== YOUR CODE: ======
        with torch.no_grad():
          actions_proba = nn.functional.softmax(self.p_net(self.curr_state)[0], dim = 0)
        # ========================
        return actions_proba


class AACPolicyGradientLoss(VanillaPolicyGradientLoss):
    def __init__(self, delta: float):
        """
        Initializes an AAC loss function.
        :param delta: Scalar factor to apply to state-value loss.
        """
        super().__init__()
        self.delta = delta

    def forward(self, batch: TrainBatch, model_output, **kw):

        # Get both outputs of the AAC model
        action_scores, state_values = model_output

        # TODO: Calculate the policy loss loss_p, state-value loss loss_v and
        #  advantage vector per state.
        #  Use the helper functions in this class and its base.
        # ====== YOUR CODE: ======
        loss_v = self._value_loss(batch, state_values)
        advantage = self._policy_weight(batch, state_values)
        loss_p = self._policy_loss(batch, action_scores, advantage)
        # ========================

        loss_v *= self.delta
        loss_t = loss_p + loss_v
        return loss_t, dict(loss_p=loss_p.item(), loss_v=loss_v.item(),
                            adv_m=advantage.mean().item())

    def _policy_weight(self, batch: TrainBatch, state_values: torch.Tensor):
        # TODO:
        #  Calculate the weight term of the AAC policy gradient (advantage).
        #  Notice that we don't want to backprop errors from the policy
        #  loss into the state-value network.
        # ====== YOUR CODE: ======
        advantage = batch.q_vals - state_values.flatten()
        # ========================
        return advantage

    def _value_loss(self, batch: TrainBatch, state_values: torch.Tensor):
        # TODO: Calculate the state-value loss.
        # ====== YOUR CODE: ======
        loss_v = nn.functional.mse_loss(state_values.flatten(), batch.q_vals)
        # ========================
        return loss_v

