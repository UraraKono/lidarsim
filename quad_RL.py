import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
import random
from sim import LidarSim
import quad_env #quad_env.py
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

#ppo tutorial: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
#custom environments tutorial: https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/5_custom_gym_env.ipynb#scrollTo=PQfLBE28SNDr

# initialize lidar simulator
grid_width = 100  # m
grid_height = 20  # m
voxel_resolution = 0.2  # m
sim = LidarSim(grid_width, grid_height, voxel_resolution, h_res=90, v_res=45, h_fov_deg=360, v_fov_deg=45)
scene = sim.create_scene(num_cylinders=10, create_ground=True)

#make the environment and validate it
env = quad_env.QuadEnv(sim)
# check_env(env, warn=True) #The code keeps running here

#vectorize the environment
env = make_vec_env(lambda: env, n_envs=1)

#train the model w/PPO
model = PPO("MlpPolicy", env, verbose=1)  
model.learn(total_timesteps=1)
model.save("quad_RL")

obs = env.reset()

step = 0
while True:
    action, _states = model.predict(obs)
    print("Step {}".format(step + 1))
    print("Action: ", action)
    obs, rewards, dones, info = env.step(action)
    print('obs=', obs, 'reward=', rewards, 'done=', dones)
    #env.render()
    step += 1

#evaluate the model
obs = env.reset()
n_steps = 20
for step in range(n_steps):
  action, _ = model.predict(obs, deterministic=True)
  print("Step {}".format(step + 1))
  print("Action: ", action)
  obs, reward, done, info = env.step(action)
  print('obs=', obs, 'reward=', reward, 'done=', done)
  env.render(mode='console')
  if done:
    # Note that the VecEnv resets automatically
    # when a done signal is encountered
    print("Goal reached!", "reward=", reward)
    break