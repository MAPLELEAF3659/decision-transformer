import torch
import torchvision
# import imageio
import numpy as np
import cv2
import os
import argparse

from mingpt.trainer_atari import Trainer, TrainerConfig, Env, Args
from mingpt.model_atari import GPTConfig, GPT
from mingpt.utils import sample

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", type=str,
                    default="./checkpoints/checkpoint.pth")
parser.add_argument("--game", type=str, default="Pong")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--max_timestep", type=int,default=5102)
cmd_args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = GPTConfig(vocab_size=6, block_size=90, model_type='reward_conditioned',
                   n_layer=6, n_head=8, n_embd=128, max_timestep=cmd_args.max_timestep)
model = GPT(config)
model.load_state_dict(torch.load(cmd_args.file_path))
model.to(device)
model.eval()

args = Args(cmd_args.game, cmd_args.seed)
env = Env(args)
env.eval()

state = env.reset()
frames = env._get_state()

with torch.no_grad():
    state = state.type(torch.float32).to(
        device).unsqueeze(0).unsqueeze(0)
    rtgs = [20]
    # first state is from env, first rtg is target return, and first timestep is 0
    sampled_action = sample(model, state, 1, temperature=1.0, sample=True, actions=None,
                            rtgs=torch.tensor(rtgs, dtype=torch.long).to(
                                device).unsqueeze(0).unsqueeze(-1),
                            timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(device))

    j = 0
    all_states = state
    actions = []
    done = True
    while True:
        if done:
            state, reward_sum, done = env.reset(), 0, False
        # frames.append(env.render_and_get_img())
        frames = torch.cat([frames, env._get_state()])
        env.render()
        action = sampled_action.cpu().numpy()[0, -1]
        actions += [sampled_action]
        state, reward, done = env.step(action)
        reward_sum += reward
        j += 1

        if done:
            break

        state = state.unsqueeze(0).unsqueeze(0).to(device)

        all_states = torch.cat([all_states, state], dim=0)

        rtgs += [rtgs[-1] - reward]
        # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
        # timestep is just current timestep
        sampled_action = sample(model, all_states.unsqueeze(0), 1, temperature=1.0, sample=True,
                                actions=torch.tensor(actions, dtype=torch.long).to(
                                    device).unsqueeze(1).unsqueeze(0),
                                rtgs=torch.tensor(rtgs, dtype=torch.long).to(
                                    device).unsqueeze(0).unsqueeze(-1),
                                timesteps=(min(j, 2369) * torch.ones((1, 1, 1), dtype=torch.int64).to(device)))

# env.render()
# env.close()

# torch.save(imgs, "./states/imgs.pth")
# torchvision.io.write_png(env.get_state_img(), "./states/state.png")

# torchvision.io.write_video(
#     "./states/state.mp4", video_array=frames.cpu(), fps=30, video_codec="h264")

# out = cv2.VideoWriter("./states/output_state.avi",
#                       cv2.VideoWriter_fourcc(*'DIVX'), 30, (320, 210))
# for i in range(len(frames)-1):
#     out.write(frames[i].int())

# out.release()

# imageio.mimwrite(os.path.join("./states/", "states.gif"), frames, fps=30)
