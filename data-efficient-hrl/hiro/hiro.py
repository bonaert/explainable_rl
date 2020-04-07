import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from hiro.models import ControllerActor, ControllerCritic, ManagerActor, ManagerCritic
from hiro.utils import has_nan_or_inf

totensor = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def var(tensor: torch.Tensor) -> torch.Tensor:
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


def get_tensor(z: np.ndarray) -> torch.Tensor:
    if len(z.shape) == 1:
        return var(torch.FloatTensor(z.copy())).unsqueeze(0)
    else:
        return var(torch.FloatTensor(z.copy()))


class Controller:
    def __init__(self, state_dim: int, goal_dim: int, action_dim: int, max_action, actor_lr: float,
                 critic_lr: float, ctrl_rew_type: str, repr_dim=15, no_xy=True,
                 policy_noise=0.2, noise_clip=0.5, use_tanh=True
                 ):
        self.actor = ControllerActor(state_dim, goal_dim, action_dim, scale=max_action, use_tanh=use_tanh)
        self.actor_target = ControllerActor(state_dim, goal_dim, action_dim, scale=max_action, use_tanh=use_tanh)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = ControllerCritic(state_dim, goal_dim, action_dim)
        self.critic_target = ControllerCritic(state_dim, goal_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=0.0001)

        self.no_xy = no_xy

        self.subgoal_transition = self.hiro_subgoal_transition

        if torch.cuda.is_available():
            self.actor = self.actor.cuda()
            self.actor_target = self.actor_target.cuda()
            self.critic = self.critic.cuda()
            self.critic_target = self.critic_target.cuda()

        self.criterion = nn.SmoothL1Loss()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.use_tanh = use_tanh

    def clean_obs(self, state: torch.Tensor, dims=2, no_grad=True) -> torch.Tensor:
        """ Sets the first dims elements of the last dimension to 0
        Note: the default is 2, and I think the first 2 elements correspond to the position in the maze (TODO: check)
        Therefore, we're hiding our current position when cleaning the observation
        """
        if self.no_xy:
            if no_grad:
                with torch.no_grad():
                    mask = torch.ones_like(state)
                    if len(state.shape) == 3:
                        mask[:, :, :dims] = 0
                    elif len(state.shape) == 2:
                        mask[:, :dims] = 0
                    elif len(state.shape) == 1:
                        mask[:dims] = 0

                    return state * mask
            else:
                mask = torch.ones_like(state)
                if len(state.shape) == 3:
                    mask[:, :, :dims] = 0
                elif len(state.shape) == 2:
                    mask[:, :dims] = 0
                elif len(state.shape) == 1:
                    mask[:dims] = 0

                return state * mask
        else:
            return state

    def select_action(self, states, subgoals, to_numpy=True, from_numpy=True, no_grad=True):
        if from_numpy:
            states = get_tensor(states)
            subgoals = get_tensor(subgoals)

        states = self.clean_obs(states, no_grad=no_grad)

        if to_numpy:
            res = self.actor(states, subgoals).cpu().data.numpy()
        else:
            res = self.actor(states, subgoals)

        # one element + A actions -> 1 x A -> squueze() -> A :)
        # one element + 1 action -> 1 x 1 -> squeeze() -> () :(   (we want 1 instead of empty tuple as the shape)
        # N elements (batch) + A actions -> N x A -> squeeze -> N x A :)
        # N elements (batch) + 1 action -> N x 1 -> squeeze -> N :(   (we want N x 1)
        # So, in summary, we should only squeeze the first dimension! That way the first 2 cases work and in the
        # last two cases nothing happens (as we want)
        if res.shape[0] == 1:
            res = res.squeeze(0)

        if has_nan_or_inf(res):
            raise Exception("Action has NaNs: %s" % res)

        return res

    def value_estimate(self, state, subgoal, action):
        state = self.clean_obs(get_tensor(state))
        subgoal = get_tensor(subgoal)
        action = get_tensor(action)
        return self.critic(state, subgoal, action)

    def actor_loss(self, state, subgoal):
        """ The actor loss is given by the mean Q value of the actions the controller would take
        at the given state. We minimize -Q-values, which means we maximize the Q-values we obtain.
        So, in other words, we're trying to pick actions which will lead to high Q values.
        """
        return -self.critic.Q1(state, subgoal, self.actor(state, subgoal)).mean()

    def hiro_subgoal_transition(self, state, subgoal, next_state):
        if len(state.shape) == 1:  # check if batched
            return state[:self.goal_dim] + subgoal - next_state[:self.goal_dim]
        else:
            return state[:, :self.goal_dim] + subgoal - next_state[:, :self.goal_dim]

    def multi_subgoal_transition(self, states: torch.Tensor, subgoal: torch.Tensor) -> torch.Tensor:
        subgoals = (subgoal + states[:, 0, :self.goal_dim])[:, None] - states[:, :, :self.goal_dim]
        return subgoals

    def train(self, replay_buffer, iterations, writer, timestep, batch_size=100, discount=0.99, tau=0.005):
        avg_act_loss, avg_crit_loss = 0., 0.

        for it in range(iterations):
            # Sample replay buffer, convert to tensors, get next subgoal and clean observations
            state, next_state, subgoal, action, ctrlr_reward, ctrlr_done, _, _ = replay_buffer.sample(batch_size)
            next_subgoal = get_tensor(self.subgoal_transition(state, subgoal, next_state))
            state = self.clean_obs(get_tensor(state))
            action = get_tensor(action)
            subgoal = get_tensor(subgoal)
            done = get_tensor(1 - ctrlr_done)
            reward = get_tensor(ctrlr_reward)
            next_state = self.clean_obs(get_tensor(next_state))

            # Q target = reward + discount * Q(next_state, pi(next_state))
            # 1) Add clipped normal noise to the action and clip the actions to the bounds
            noise = torch.FloatTensor(action).data.normal_(0, self.policy_noise).to(device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state, next_subgoal) + noise)

            if self.use_tanh:
                next_action = torch.min(next_action, self.actor.scale)
                next_action = torch.max(next_action, -self.actor.scale)

            # 2) Compute the Q values that we want (the target Q values)
            target_Q1, target_Q2 = self.critic_target(next_state, next_subgoal, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q)
            target_Q_no_grad = target_Q.detach()

            # 3) Get current Q estimate
            current_Q1, current_Q2 = self.critic(state, subgoal, action)

            writer.add_scalar('values/mgr_Q_value1', current_Q1[0], timestep + it)
            writer.add_scalar('values/mgr_Q_value2', current_Q2[0], timestep + it)
            writer.add_scalar('values/mgr_target_q', target_Q[0], timestep + it)

            # 4) Compute critic loss for both networks.
            #    This is a dual network achitecture where we take the min predictions of both networks.
            #    Note that each network has a target network which is updated using Polyak averaging!
            #    So in total there's 4 networks (Q1, Q2, Q1_target, Q2_target)
            critic_loss = self.criterion(current_Q1, target_Q_no_grad) + self.criterion(current_Q2, target_Q_no_grad)

            # 5) Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # 6) Compute actor loss
            actor_loss = self.actor_loss(state, subgoal)

            # 7) Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            avg_act_loss += actor_loss
            avg_crit_loss += critic_loss

            # 8) Update the target models using Polyak averaging (for both critic and actor)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return avg_act_loss / iterations, avg_crit_loss / iterations

    def save(self, directory: str):
        torch.save(self.actor.state_dict(), '%s/ControllerActor.pth' % directory)
        torch.save(self.critic.state_dict(), '%s/ControllerCritic.pth' % directory)
        torch.save(self.actor_target.state_dict(), '%s/ControllerActorTarget.pth' % directory)
        torch.save(self.critic_target.state_dict(), '%s/ControllerCriticTarget.pth' % directory)
        torch.save(self.actor_optimizer.state_dict(), '%s/ControllerActorOptim.pth' % directory)
        torch.save(self.critic_optimizer.state_dict(), '%s/ControllerCriticOptim.pth' % directory)

    def load(self, directory: str):
        self.actor.load_state_dict(torch.load('%s/ControllerActor.pth' % directory))
        self.critic.load_state_dict(torch.load('%s/ControllerCritic.pth' % directory))
        self.actor_target.load_state_dict(torch.load('%s/ControllerActorTarget.pth' % directory))
        self.critic_target.load_state_dict(torch.load('%s/ControllerCriticTarget.pth' % directory))
        self.actor_optimizer.load_state_dict(torch.load('%s/ControllerActorOptim.pth' % directory))
        self.critic_optimizer.load_state_dict(torch.load('%s/ControllerCriticOptim.pth' % directory))


class Manager:
    def __init__(self, state_dim, goal_dim, action_dim, actor_lr,
                 critic_lr, candidate_goals, correction=True,
                 scale=10, actions_norm_reg=0, policy_noise=0.2,
                 noise_clip=0.5, should_reach_subgoal=False, subgoal_dist_cost_cf=1):
        self.scale = scale
        self.actor = ManagerActor(state_dim, goal_dim, action_dim, scale=scale)
        self.actor_target = ManagerActor(state_dim, goal_dim, action_dim, scale=scale)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = ManagerCritic(state_dim, goal_dim, action_dim)
        self.critic_target = ManagerCritic(state_dim, goal_dim, action_dim)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 weight_decay=0.0001)

        self.action_norm_reg = 0

        self.should_reach_subgoal = should_reach_subgoal
        self.subgoal_dist_cost_cf = subgoal_dist_cost_cf

        if torch.cuda.is_available():
            self.actor = self.actor.cuda()
            self.actor_target = self.actor_target.cuda()
            self.critic = self.critic.cuda()
            self.critic_target = self.critic_target.cuda()

        self.criterion = nn.SmoothL1Loss()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.candidate_goals = candidate_goals
        self.correction = correction
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

    def set_eval(self):
        self.actor.set_eval()
        self.actor_target.set_eval()

    def set_train(self):
        self.actor.set_train()
        self.actor_target.set_train()

    def sample_subgoal(self, state, goal, to_numpy=True):
        state = get_tensor(state)
        goal = get_tensor(goal)

        if to_numpy:
            res = self.actor(state, goal).cpu().data.numpy().squeeze()
        else:
            res = self.actor(state, goal).squeeze()

        if has_nan_or_inf(res):
            raise Exception("Action has NaNs: %s" % res)
        return res

    def value_estimate(self, state, goal, subgoal):
        return self.critic(state, goal, subgoal)

    def actor_loss(self, state, goal):
        """ The actor loss is composed of two elements (the second one is disabled by default):
        1) The negation of the Q-values of the subgoals at the current state. This will make the actor pick
           subgoals that will maximize the Q-values of the subgoals (e.g. pick good subgoals).
        2) The norm of the actions (multiplied by scale 0 by default)
        """
        actions = self.actor(state, goal)
        subgoal_q_values = -self.critic.Q1(state, goal, actions).mean()
        norm = torch.norm(actions) * self.action_norm_reg
        return subgoal_q_values + norm

    def off_policy_corrections(self, controller_policy: Controller, batch_size: int, subgoals, observations, actions):
        """ For each element of the batch, we will create 10 possible subgoals and then pick the one
        that maximizes the stored actions in the current low-level controller """
        # return subgoals

        # new_subgoals = controller_policy.multi_subgoal_transition(x_seq, subgoals)
        first_obs = [x[0] for x in observations]  # First obs
        last_obs = [x[-1] for x in observations]  # Last obs

        # 1. Create 1 goal with value final_obs - initial_obs
        # Shape: (batch_size, 1, subgoal_dim)
        diff_goal = (np.array(last_obs) - np.array(first_obs))[:, np.newaxis, :self.action_dim]

        # 2. Create 1 goal equal to the original goal
        # Shape: (batch_size, 1, subgoal_dim)
        original_goal = np.array(subgoals)[:, np.newaxis, :]

        # 3. Create 8 goal in the normal around (final obs - initial obs). We ensure they fit the scale
        # Shape: (batch_size, 8, subgoal_dim)
        random_goals = np.random.normal(loc=diff_goal, scale=.5 * self.scale[None, None, :],
                                        size=(batch_size, self.candidate_goals, original_goal.shape[-1]))
        random_goals = random_goals.clip(-self.scale, self.scale)

        # Concatenate all the goal we create into a single tensor
        # Shape: (batch_size, 10, subgoal_dim)
        candidates = np.concatenate([original_goal, diff_goal, random_goals], axis=1)
        observations = np.array(observations)[:, :-1, :]

        actions = np.array(actions)

        seq_len = len(observations[0])

        # To make coding easier
        new_batch_size = seq_len * batch_size
        action_dim = actions[0][0].shape
        obs_dim = observations[0][0].shape
        num_candidate_goals = candidates.shape[1]

        # Reshape the actions, observations and goal so that they have shape (batch_size, *dim_element)
        true_actions = actions.reshape((new_batch_size,) + action_dim)
        reshaped_observations = observations.reshape((new_batch_size,) + obs_dim)
        goal_shape = (new_batch_size, self.action_dim)
        # observations = get_obs_tensor(observations, sg_corrections=True)

        # batched_candidates = np.tile(candidates, [seq_len, 1, 1])
        # batched_candidates = batched_candidates.transpose(1, 0, 2)

        # Predict the actions for each candidate goal
        policy_actions = np.zeros((num_candidate_goals, new_batch_size) + action_dim)
        for goal_num in range(num_candidate_goals):
            # 1. Compute all the subgoal transitions (given the known observations)
            adjusted_subgoals = controller_policy.multi_subgoal_transition(observations, candidates[:, goal_num])
            adjusted_subgoals = adjusted_subgoals.reshape(*goal_shape)
            # 2. Compute the action the agent would have taken given the observations and the subgoals
            policy_actions[goal_num] = controller_policy.select_action(reshaped_observations, adjusted_subgoals)

        # Compute the log(probability) of the actions for each candidate goal (for every batch)
        difference = (policy_actions - true_actions)
        # Replace -inf by 0 (there's inf when we reached the end of the episode and we filled the meta transition
        # with dummy actions so that it has the same number of actions as the other meta transitions. The dummy
        # value they picked is np.inf)
        difference = np.where(difference != -np.inf, difference, 0)
        # Shape: (batch_size, num_candidate_goals, seq_len, action_dim)
        difference = difference.reshape((num_candidate_goals, batch_size, seq_len) + action_dim).transpose(1, 0, 2, 3)
        # Shape: (batch_size, num_candidate_goals)
        logprob = -0.5 * np.sum(np.linalg.norm(difference, axis=-1) ** 2, axis=-1)

        # Pick the candidate goals that maximize the log(probability) of the actions
        # Shape (batch_size, )
        max_indices = np.argmax(logprob, axis=-1)
        # print(diff_goal[0, 0])
        # print(original_goal[0, 0])
        # print(candidates[0, max_indices[0]])

        # Return the goals that maximize the log(probabiliy) of the actions
        # Shape (batch_size, subgoal_dim)
        return candidates[np.arange(batch_size), max_indices]

    def train(self, controller_policy, replay_buffer, iterations, writer, timestep, batch_size=100, discount=0.99, tau=0.005):

        avg_act_loss, avg_crit_loss = 0., 0.
        for it in range(iterations):
            # Sample replay buffer
            # The manager reward is the sum of the rewards of the actions done while that subgoal was active
            # This total reward is scaled by a factor. So the manager tries to pick subgoals that maximize the
            # reward that is collected, and the low level controller tries to get as close as possible to the
            # desired end state; and as fast as possible.
            states, final_state, goal, subgoal_original, mgr_rewards, dones, observations, actions = replay_buffer.sample(
                batch_size)
            # Basically, the only place where it's needed to have all actions array have the same size are
            # in the off policy correction section. If you look at the code, that's literally the only place
            # where we use that variable. This makes sense, because normally the manager shouldn't even care
            # about the actions of the low-level controller. The only reason we keep track of the actions
            # and observations is so that we can adjust the subgoal to ones that corresponds better to the
            # low-level corrections
            if self.correction:
                subgoals = self.off_policy_corrections(controller_policy, batch_size, subgoal_original, observations,
                                                       actions)
            else:
                subgoals = subgoal_original

            states = get_tensor(states)
            next_states = get_tensor(final_state)
            goal = get_tensor(goal)
            subgoals = get_tensor(subgoals)

            rewards = get_tensor(mgr_rewards)
            dones = get_tensor(1 - dones)

            # Q target = reward + discount * Q(next_state, pi(next_state))
            # 1) Add clipped normal noise to the action and clip the actions to the bounds
            noise = torch.FloatTensor(subgoal_original).data.normal_(0, self.policy_noise).to(device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_states, goal) + noise)
            next_action = torch.min(next_action, self.actor.scale)
            next_action = torch.max(next_action, -self.actor.scale)

            # 2) Compute the Q values that we want (the target Q values)
            target_Q1, target_Q2 = self.critic_target(next_states, goal, next_action)
            # target_Q1, target_Q2 = self.critic_target(next_state, goal, self.actor_target(next_state, goal))
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (dones * discount * target_Q)
            target_Q_no_grad = target_Q.detach()

            # 3) Get current Q estimate
            current_Q1, current_Q2 = self.value_estimate(states, goal, subgoals)

            # 4) Compute critic loss for both networks.
            #    This is a dual network achitecture where we take the min predictions of both networks.
            #    Note that each network has a target network which is updated using Polyak averaging!
            #    So in total there's 4 networks (Q1, Q2, Q1_target, Q2_target)
            critic_loss = self.criterion(current_Q1, target_Q_no_grad) + self.criterion(current_Q2, target_Q_no_grad)

            writer.add_scalar('values/batch_0.Q_value1', current_Q1[0], timestep + it)
            writer.add_scalar('values/batch_0.Q_value2', current_Q2[0], timestep + it)
            writer.add_scalar('values/batch_0.target_q', target_Q[0], timestep + it)

            writer.add_scalar('values/loss_td_error', critic_loss, timestep + it)

            # 4b) Optionally add the component that increases the loss if the low level controller couldn't reach
            #     the subgoal. This loss is a measure of the distance. We need to compute the subgoal at the last step
            #     This is simple, we can use the formula g' = s + g - s'. We can then subgoal norm as the distance cost.
            if self.should_reach_subgoal:
                # Shape (batch_size, subgoal_dim)
                final_subgoal = states[..., :self.action_dim] + subgoals - final_state[..., :self.action_dim]
                # The loss if the sum of the norms (for all meta-transition in the batch)
                subgoal_loss = self.subgoal_dist_cost_cf * final_subgoal.norm(dim=-1).sum()
                writer.add_scalar('values/subgoal_dist_loss', subgoal_loss.item(), timestep + it)
                critic_loss += subgoal_loss

            # 5) Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # 6) Compute actor loss
            actor_loss = self.actor_loss(states, goal)

            # 7) Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            avg_act_loss += actor_loss
            avg_crit_loss += critic_loss

            # 8) Update the target models using Polyak averaging (for both critic and actor)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return avg_act_loss / iterations, avg_crit_loss / iterations

    def load_pretrained_weights(self, filename):
        state = torch.load(filename)
        self.actor.encoder.load_state_dict(state)
        self.actor_target.encoder.load_state_dict(state)
        print("Successfully loaded Manager encoder.")

    def save(self, directory):
        torch.save(self.actor.state_dict(), '%s/ManagerActor.pth' % directory)
        torch.save(self.critic.state_dict(), '%s/ManagerCritic.pth' % directory)
        torch.save(self.actor_target.state_dict(), '%s/ManagerActorTarget.pth' % directory)
        torch.save(self.critic_target.state_dict(), '%s/ManagerCriticTarget.pth' % directory)
        torch.save(self.actor_optimizer.state_dict(), '%s/ManagerActorOptim.pth' % directory)
        torch.save(self.critic_optimizer.state_dict(), '%s/ManagerCriticOptim.pth' % directory)

    def load(self, directory):
        self.actor.load_state_dict(torch.load('%s/ManagerActor.pth' % directory))
        self.critic.load_state_dict(torch.load('%s/ManagerCritic.pth' % directory))
        self.actor_target.load_state_dict(torch.load('%s/ManagerActorTarget.pth' % directory))
        self.critic_target.load_state_dict(torch.load('%s/ManagerCriticTarget.pth' % directory))
        self.actor_optimizer.load_state_dict(torch.load('%s/ManagerActorOptim.pth' % directory))
        self.critic_optimizer.load_state_dict(torch.load('%s/ManagerCriticOptim.pth' % directory))
