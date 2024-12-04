import gymnasium as gym
import wandb
import os

from gymnasium.wrappers import RecordVideo

os.environ["WANDB_DISABLE_SERVICE"] = "True"
os.environ['WANDB_START_METHOD'] = 'thread'


class LogWrapper(gym.Wrapper):
    def __init__(self, env, use_wandb=False, config=None,
                 wandb_project=None, video_folder=None, video_freq=1):
        if video_folder is not None:
            os.makedirs(video_folder, exist_ok=True)

        super().__init__(RecordVideo(
            env, video_folder,
            episode_trigger=lambda x: x % video_freq == 0,
            name_prefix='video'
        ) if video_folder is not None else env)

        self.use_wandb = use_wandb

        self.episode_index = 0
        self.episode_reward = 0
        self.episode_length = 0
        self.total_steps = 0

        if use_wandb:
            if wandb.run is not None:
                wandb.finish()
            wandb.init(project=wandb_project, reinit=True, config=config)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        if self.episode_index == 0:
            self.episode_index = 1
            return obs
        self.episode_index += 1

        if self.use_wandb:
            wandb.log({
                'episode_reward': self.episode_reward,
                'episode_length': self.episode_length,
                'total_steps': self.total_steps,
                'episode': self.episode_index
            }, step=self.episode_index)

        self.episode_reward = 0
        self.episode_length = 0

        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.episode_reward += reward
        self.episode_length += 1
        self.total_steps += 1

        return obs, reward, terminated, truncated, info
