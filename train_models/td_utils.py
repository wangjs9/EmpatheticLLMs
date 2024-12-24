import os
import time
import torch
import numpy as np
from tqdm import tqdm
from train_models.td_trainer import ArcherTrainer, BCTrainer


class ReplayBuffer:
    def __init__(self, batch_size=2, capacity=10000):
        self.max_size = capacity
        self.size = 0
        self.observations = None
        self.rewards = None
        self.next_observations = None
        self.dones = None
        self.batch_size = batch_size
        self.actions = None
        self.mc_returns = None
    
    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        rand_indices = np.random.randint(0, self.size, size=(batch_size,)) % self.max_size
        return {
            "observation": self.observations[rand_indices],
            "action": self.actions[rand_indices],
            "reward": self.rewards[rand_indices],
            "next_observation": self.next_observations[rand_indices],
            "done": self.dones[rand_indices],
            "mc_return": self.mc_returns[rand_indices],
        }
    
    def __len__(self):
        return self.size
    
    def insert(self, observation, action, reward: np.ndarray, next_observation, done: np.ndarray, mc_return, **kwargs):
        """
        Insert a single transition into the replay buffer.

        Use like:
            replay_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )
        """
        if isinstance(reward, (float, int)):
            reward = np.array(reward)
        if isinstance(mc_return, (float, int)):
            mc_return = np.array(mc_return)
        if isinstance(done, bool):
            done = np.array(done)
        
        if self.observations is None:
            self.observations = np.array([''] * self.max_size, dtype='object')
            self.actions = np.array([''] * self.max_size, dtype='object')
            self.rewards = np.empty((self.max_size, *reward.shape), dtype=reward.dtype)
            self.next_observations = np.array([''] * self.max_size, dtype='object')
            self.dones = np.empty((self.max_size, *done.shape), dtype=done.dtype)
            self.mc_returns = np.empty((self.max_size, *mc_return.shape), dtype=mc_return.dtype)
        
        assert reward.shape == ()
        assert done.shape == ()
        
        self.observations[self.size % self.max_size] = observation
        self.actions[self.size % self.max_size] = action
        self.rewards[self.size % self.max_size] = reward
        self.next_observations[self.size % self.max_size] = next_observation
        self.dones[self.size % self.max_size] = done
        self.mc_returns[self.size % self.max_size] = mc_return
        
        self.size += 1


def add_trajectory_reward(trajectory):
    """
    add trajectory reward to the dict of each interaction
    """
    trajectory_reward = np.sum([d["reward"] for d in trajectory])
    for d in trajectory:
        d.update({"trajectory_reward": trajectory_reward})
    return trajectory


def add_mc_return(trajectory, gamma=0.95):
    """
    add trajectory reward to the dict of each interaction
    """
    trajectory_rewards = np.array([d["reward"] for d in trajectory]).reshape(1, -1)
    gamma_row = np.cumprod(np.ones((1, trajectory_rewards.shape[1])) * gamma)
    gamma_matrix = np.triu(gamma_row.reshape(1, -1) / gamma_row.reshape(-1, 1))
    mc_returns = np.sum(trajectory_rewards * gamma_matrix, axis=1)
    for d, mc in zip(trajectory, mc_returns):
        d.update({"mc_return": mc})
    return trajectory


def batch_interact_environment(
        agent, tokenizer, env, num_trajectories, post_f=lambda x: x, use_tqdm=True, decode_f=lambda x: x, env_idx=None):
    """
    in a bacthed way, interact with the environments  to get a list of trajectories
    [[{"observation":, "next_observation":, "reward":, "done":},...],...]
    post_f: function to add additional attributes to the trajectory
    """
    bsize = env.bsize
    all_trajectories = []
    for num_t in tqdm(range(num_trajectories // bsize), disable=not use_tqdm):
        done = False
        trajectories = [[] for _ in range(bsize)]
        # obs = reset_to(env, 69)
        batch_obs = env.reset(idx=env_idx)
        batch_done = [False, ] * bsize
        steps = 0
        while not all(batch_done):
            steps += 1
            # print(f"Environment stpes {str(steps)}")
            action = agent.get_action(batch_obs)
            batch_return = env.step(decode_f(action))
            for i, result in zip(range(bsize), batch_return):
                if result is None:
                    continue
                next_obs, r, done = result
                trajectories[i].append({
                    "observation": batch_obs[i], "next_observation": next_obs, "reward": r, "done": done,
                    "action": action[i]
                })
                batch_obs[i] = next_obs
                batch_done[i] = done
            # obs = next_obs
        print(trajectories[0][-1]["next_observation"])
        all_trajectories += [post_f(add_mc_return(add_trajectory_reward(trajectory))) for trajectory in trajectories]
        # breakpoint()
        # trajectories.append(post_f(add_trajectory_reward(trajectory)))
    return all_trajectories


def offpolicy_train_loop(
        env, eval_env, agent, tokenizer, accelerator, warmup_iter: int = 20, rollout_size: int = 50, eval_size: int = 1,
        batch_size: int = 2, capacity: int = 500000, iterations: int = 10, epochs: int = 3, grad_accum_steps: int = 1,
        env_idx: int = None, do_sample: bool = False, temperature: float = 2.0, critic_lr: float = 1e-3,
        lm_lr: float = 1e-5, gamma: float = 0.9, tau: float = 0.1, env_load_path: str = '', actor_epochs: int = 3,
        max_grad_norm: float = 0.01, save_path: str = None, save_freq: int = 25, eval_freq: int = 25,
        agent_type: str = "archer", decode_f: callable = lambda x: x, **kwargs
):
    if agent_type.lower() == "chai" or agent_type.lower() == "archer" or agent_type.lower() == "archer_llm":
        trainer = ArcherTrainer(agent=agent, accelerator=accelerator, tokenizer=tokenizer, critic_lr=critic_lr,
                                lm_lr=lm_lr, gamma=gamma, tau=tau, epochs=epochs, actor_epochs=actor_epochs,
                                grad_accum_steps=grad_accum_steps, max_grad_norm=max_grad_norm)
    elif agent_type.lower() == "online_filteredbc":
        trainer = BCTrainer(agent=agent, tokenizer=tokenizer, accelerator=accelerator, lm_lr=lm_lr, epochs=actor_epochs,
                            grad_accum_steps=grad_accum_steps, max_grad_norm=max_grad_norm)
    replay_buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)
    all_trajectories = []
    if accelerator.is_main_process:
        if os.path.exists(os.path.join(save_path, 'trainer.pt')):
            # print("Not using existing checkpoint")
            print("Loading from checkpoint")
            trainer.load(os.path.join(save_path, 'trainer.pt'))
            all_trajectories = torch.load(os.path.join(save_path, 'trajectories.pt'))
            replay_buffer = torch.load(os.path.join(save_path, 'replay_buffer.pt'))
        else:
            print("Creating new checkpoint directory")
            os.makedirs(save_path, exist_ok=True)
    agent.prepare()
    # main training loop
    print(">>>start iterations")
    for i in tqdm(range(iterations)):
        # print(">>>Interacting with Environment")
        if accelerator.is_main_process:
            trajectories = batch_interact_environment(
                agent=agent, tokenizer=tokenizer, env=env, num_trajectories=rollout_size, env_idx=env_idx,
                use_tqdm=False, decode_f=decode_f)
            info = {"rollout.mean": np.mean([d[0]["trajectory_reward"] for d in trajectories]),
                    "rollout.max": np.max([d[0]["trajectory_reward"] for d in trajectories]),
                    "rollout.min": np.min([d[0]["trajectory_reward"] for d in trajectories])}
            if (i + 1) % eval_freq == 0:
                old_sample = agent.do_sample
                agent.do_sample = False
                eval_trajectories = batch_interact_environment(
                    agent=agent, tokenizer=tokenizer, env=eval_env, num_trajectories=max(eval_size, eval_env.bsize),
                    env_idx=env_idx, use_tqdm=False, decode_f=decode_f)
                agent.do_sample = old_sample
                info.update({"eval_rollout.mean": np.mean([d[0]["trajectory_reward"] for d in eval_trajectories]),
                             "eval_rollout.max": np.max([d[0]["trajectory_reward"] for d in eval_trajectories]),
                             "eval_rollout.min": np.min([d[0]["trajectory_reward"] for d in eval_trajectories]), })
            all_trajectories += trajectories
            data = sum(trajectories, [])
            for t in data:
                replay_buffer.insert(**t)
            info.update({"rollout.reward.mean": np.mean([d["reward"] for d in data]),
                         "rollout.reward.max": np.max([d["reward"] for d in data]),
                         "rollout.reward.min": np.min([d["reward"] for d in data])})
            print(">>> Saving Replay Buffer")
            torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))
            torch.save(all_trajectories, os.path.join(save_path, 'trajectories.pt'))
            print(">>> Saved Replay Buffer")
            time.sleep(15)
        else:
            info = {}
        accelerator.wait_for_everyone()
        all_trajectories = torch.load(os.path.join(save_path, 'trajectories.pt'))
        replay_buffer = torch.load(os.path.join(save_path, 'replay_buffer.pt'))
        print("Training")
        if 'filtered' in agent_type.lower():
            filtered_buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)
            episode_rewards = [d[0]["trajectory_reward"] for d in all_trajectories]
            cutoff = np.quantile(episode_rewards, 1 - 0.1)
            print("Episode Reward Cutoff: ", cutoff)
            filtered_trajectories = list(filter(lambda x: x[0]["trajectory_reward"] >= cutoff, all_trajectories))
            data = sum(filtered_trajectories, [])
            for d in data:
                filtered_buffer.insert(**d)
            info.update(trainer.update(filtered_buffer, no_update_actor=(i < warmup_iter)))
        else:
            # data = list(filter(lambda x: x["reward"] >0, data))
            info.update(trainer.update(replay_buffer, no_update_actor=(i < warmup_iter)))
        
        if (i + 1) % save_freq == 0 and save_path is not None and accelerator.is_main_process:
            print("Saving")
            trainer.save(os.path.join(save_path, 'trainer.pt'))
            torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))
    # return model
