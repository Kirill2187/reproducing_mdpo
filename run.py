import argparse
from config_loader import ConfigLoader
from log_wrapper import LogWrapper
from stable_baselines3 import PPO
from stable_baselines3.mdpo_on import MDPO
import gymnasium as gym

algos = {
    'PPO': PPO,
    'MDPO': MDPO,
}


def main():
    parser = argparse.ArgumentParser(description='Run experiment')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    config = ConfigLoader.load(args.config)

    algo = config['agent']['type']
    if algo not in algos:
        raise ValueError(f'Invalid algorithm: {algo}')

    env_name = config['environment_name']
    hyperparameters = config['agent']['hyperparameters']

    env = gym.make(env_name, render_mode='rgb_array')
    env = LogWrapper(
        env,
        use_wandb=config['logging']['use_wandb'],
        config=config,
        wandb_project=config['logging']['wandb_project'],
        video_folder=config['logging']['video_dir'],
        video_freq=config['logging']['save_video_frequency']
    )

    model = algos[algo](
        policy='MlpPolicy',
        env=env,
        verbose=config['logging']['verbose'],
        learning_rate=lambda t: 3e-4 * t,
        **hyperparameters,
    )

    model.learn(total_timesteps=config['training']['num_steps'])


if __name__ == '__main__':
    main()
