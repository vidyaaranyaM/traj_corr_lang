from torch.utils.data import Dataset, DataLoader
import pickle
import torch
import sys
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from collect_bc_trajs import CollectBCTrajs
import utils
import wandb
import random
import collections
import os

ltable_path = '../'
sys.path.append(ltable_path)
from environments import blocks
from environments import language_table
from environments.rewards import block2block


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(42)


BC_DATA_FILENAME = "../pickle_files/bc_data.pickle"
PRESENTATION_VIDEOS_FILENAME = "../videos/presentation"
POLICY_MODEL_DIR = "../models/bc_policy/w_lang_100_demo"
POLICY_FILE_PATH = POLICY_MODEL_DIR + "/model_size_512.pth"
FULL_TRAIN_TEST_PICKLE_PATH = "../pickle_files/full_train_test_100_eps_w_lang.pickle"

LLM_NAME = "distilbert-base-uncased"

TRAIN_TEST_SPLIT = 0.9
NUM_EPOCHS = 25000
# NUM_EPOCHS = 1000
LEARNING_RATE = 1e-3
MIN_LR  = 1e-6
LR_MULTIPLIER = 0.9999
BATCH_SIZE = 32


class BCDataset(Dataset):
    def __init__(self, states, actions) -> None:
        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]
    
class BCAgent(nn.Module):
    def __init__(self, env, single_episode=None):
        super(BCAgent, self).__init__()
        self.create_model_dir()
        self.env = env
        self.col_trajs = CollectBCTrajs(self.env)
        self.import_custom_ds()
        self.configure_llm()

        self.single_episode = single_episode
        if not self.single_episode:
            self.episode_num = "All Trajectories are considered"

        input_size = self.get_policy_input_size()
        self.h1_size = 512
        self.h2_size = 512
        output_size = 9
        self.policy = nn.Sequential(
            nn.Linear(input_size, self.h1_size),
            nn.ReLU(),
            nn.Linear(self.h1_size, self.h2_size),
            nn.ReLU(),
            nn.Linear(self.h2_size, output_size)
        ).to(device)
        self.learning_rate = LEARNING_RATE
        self.loss = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
    
    def create_model_dir(self):
        if not os.path.exists(POLICY_MODEL_DIR):
            os.makedirs(POLICY_MODEL_DIR)
    
    def import_custom_ds(self):
        with open(BC_DATA_FILENAME, 'rb') as file:
            self.custom_ds = pickle.load(file)

    def get_policy_input_size(self):
        episode_0 = list(self.custom_ds.values())[0]
        states = episode_0['states']
        instruction = episode_0['instruction']
        return states["effector_target_translation"].shape[1] + \
                states['red_moon'].shape[1] * 4 + \
                self.get_text_embedding(instruction).shape[1]

    def configure_llm(self):
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
        self.llm_model = AutoModel.from_pretrained(LLM_NAME).to(device)

    def get_text_embedding(self, text):
        with torch.no_grad():
            encoded_input = self.tokenizer(text, return_tensors='pt', padding=True)
            input_ids = encoded_input['input_ids'].to(device)
            attention_mask = encoded_input['attention_mask'].to(device)
            lang_embedding = self.llm_model(input_ids, attention_mask=attention_mask).last_hidden_state
            # check this
            lang_embedding = lang_embedding.mean(1)
            return lang_embedding

    def get_state_tensor(self, episode, step):
        for iter, key in enumerate(episode['states'].keys()):
            if iter == 0:
                state_tensor = torch.tensor(episode['states'][key][step, :])
            else:
                state_tensor = torch.cat((state_tensor,
                                            torch.tensor(episode['states'][key][step, :])),
                                            dim = 0)
        instruction_tensor = self.get_text_embedding(episode['instruction']).cpu().reshape(-1)
        return torch.cat((state_tensor, instruction_tensor), dim=0).unsqueeze(0)
    
    def get_state_action_tensor(self, episode):
        num_time_steps = episode['actions'].shape[0]
        for step in range(num_time_steps):
            if step == 0:
                state_tensor = self.get_state_tensor(episode, step)
                action_tensor = torch.tensor(episode['actions'][step, :]).unsqueeze(0)
            else:
                state_tensor = torch.cat((state_tensor,
                                        self.get_state_tensor(episode, step)),
                                        dim=0)
                action_tensor = torch.cat((action_tensor,
                                        torch.tensor(episode['actions'][step, :]).unsqueeze(0)),
                                        dim=0)
        return state_tensor, action_tensor
        
    def process_data(self):
        if self.single_episode:
            # self.episode_num = random.randint(0, len(self.custom_ds.values()))
            self.episode_num = 0
            print("epsiode num: ", self.episode_num)
            episodes = [list(self.custom_ds.values())[self.episode_num]]
        else:
            episodes = list(self.custom_ds.values())[:100]
            print("num episodes: ", len(episodes))

        for iter, episode in enumerate(episodes):
            ep_state_tensor, ep_action_tensor = self.get_state_action_tensor(episode)
            if iter == 0:
                state_tensor = ep_state_tensor
                action_tensor = ep_action_tensor
            else:
                state_tensor = torch.cat((state_tensor,
                                          ep_state_tensor),
                                          dim=0)
                action_tensor = torch.cat((action_tensor,
                                          ep_action_tensor),
                                          dim=0)
        indices = torch.randperm(state_tensor.shape[0])
        state_tensor = state_tensor[indices].type(torch.float32)
        action_tensor = action_tensor[indices].type(torch.float32)
        # print("state_tensor : ", state_tensor.shape)
        # print("action_tensor : ", action_tensor.shape)

        sep_idx = int(TRAIN_TEST_SPLIT * state_tensor.shape[0])
        train_states, train_actions = state_tensor[:sep_idx], action_tensor[:sep_idx]
        test_states, test_actions = state_tensor[sep_idx:], action_tensor[sep_idx:]
        print("state_tensor : ", state_tensor.shape)
        print("train_states : ", train_states.shape)
        print("test_states : ", test_states.shape)

        self.full_loader = DataLoader(BCDataset(state_tensor, action_tensor), 
                                       batch_size=BATCH_SIZE, 
                                       shuffle=False)
        self.train_loader = DataLoader(BCDataset(train_states, train_actions), 
                                       batch_size=BATCH_SIZE, 
                                       shuffle=True)
        self.test_loader = DataLoader(BCDataset(test_states, test_actions), 
                                      batch_size=BATCH_SIZE, 
                                      shuffle=True)
        
        self.full_train_test_tensors = {
            "full_states": state_tensor,
            "full_actions": action_tensor,
            "train_states": train_states,
            "train_actions": train_actions,
            "test_states": test_states,
            "test_actions": test_actions, 
        }
        with open(FULL_TRAIN_TEST_PICKLE_PATH, "wb") as file:
            pickle.dump(self.full_train_test_tensors, file)
    
    def setup_wandb(self):
        wandb.login()
        self.run = wandb.init(
            project="with_language",
            config={
                "architecture": "mlp",
                "init_lr": self.learning_rate,
                "min_lr": MIN_LR,
                "epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr_multiplier": LR_MULTIPLIER,
                # "episode_num": self.episode_num,
                "episode_num": "100 demos",
                "num_trajs": len(self.custom_ds.keys()),
                "h1 size": self.h1_size,
                "h2_size": self.h2_size,
                "noise_injection": False,
                "model_file_name": POLICY_FILE_PATH,
            }
        )

    def train_one_epoch(self):
        running_loss = 0
        total = 0
        correct = 0
        self.policy.train(True)
        for i, batch in enumerate(self.train_loader):
            states, ref_actions = batch
            states = states.to(device)
            ref_actions = ref_actions.to(device)
            pred_actions = self.policy(states)

            # loss computation
            loss = self.loss(pred_actions, ref_actions)
            running_loss += loss.item()
            
            # accuracy computation
            pred_actions_sm = pred_actions.softmax(dim=1)
            pred_actions_oh = torch.where(pred_actions_sm == pred_actions_sm.max(dim=1, keepdim=True)[0], 1, 0)
            total += states.shape[0]
            correct += torch.all(ref_actions == pred_actions_oh, dim=1).sum()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        return running_loss / (i + 1), correct / total
    
    def eval(self):
        running_loss = 0
        total = 0
        correct = 0
        self.policy.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                states, ref_actions = batch
                states = states.to(device)
                ref_actions = ref_actions.to(device)
                pred_actions = self.policy(states)

                # loss computation
                loss = self.loss(pred_actions, ref_actions)
                running_loss += loss.item()

                # accuracy computation
                pred_actions_sm = pred_actions.softmax(dim=1)
                pred_actions_oh = torch.where(pred_actions_sm == pred_actions_sm.max(dim=1, keepdim=True)[0], 1, 0)
                total += states.shape[0]
                correct += torch.all(ref_actions == pred_actions_oh, dim=1).sum()
        return running_loss / (i + 1), correct / total

    def train(self):
        if not os.path.exists(FULL_TRAIN_TEST_PICKLE_PATH):
            print("running process_data")
            self.process_data()
        else:
            print("loading from file")
            with open(FULL_TRAIN_TEST_PICKLE_PATH, "rb") as file:
                self.full_train_test_tensors = pickle.load(file)
            state_tensor, action_tensor = self.full_train_test_tensors['full_states'], self.full_train_test_tensors['full_actions']
            train_states, train_actions = self.full_train_test_tensors['train_states'], self.full_train_test_tensors['train_actions']
            test_states, test_actions = self.full_train_test_tensors['test_states'], self.full_train_test_tensors['test_actions']

            self.full_loader = DataLoader(BCDataset(state_tensor, action_tensor), 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True)
            self.train_loader = DataLoader(BCDataset(train_states, train_actions), 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True)
            self.test_loader = DataLoader(BCDataset(test_states, test_actions), 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True)
            
        self.setup_wandb()
        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = self.train_one_epoch()
            # import pdb
            # pdb.set_trace()
            test_loss, test_acc = self.eval()

            self.run.log({"train loss": train_loss,
                        "train acc": train_acc,
                        "test loss": test_loss,
                        "test acc": test_acc,
                        "epochs": epoch})
            if epoch % 100 == 0:
                print("epochs: ", epoch, "lr: ", self.learning_rate, 
                        "train loss: ", train_loss, "test loss: ", test_loss)

            if self.learning_rate > MIN_LR:
                self.learning_rate *= LR_MULTIPLIER
                for param_group in self.optim.param_groups:
                    param_group['lr'] = self.learning_rate

        wandb.finish()

        torch.save(self.policy.state_dict(), POLICY_FILE_PATH) 


if __name__=="__main__":
    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_4,
        reward_factory=block2block.BlockToBlockReward ,
        control_frequency=10.0,
        render_mode="human"
    )

    bc_agent = BCAgent(env, single_episode=False)
    bc_agent.train()
    




















































        
    # def conv_arr_delta(self, action_arr):
    #     one_hot = np.zeros(action_arr.shape[0])
    #     one_hot[np.argmax(action_arr)] = 1
    #     return np.expand_dims(utils.convert_actions(one_hot), axis=0)
    
    # def load_policy(self, policy_file):
    #     self.policy.load_state_dict(torch.load(policy_file, map_location=device)) 

    # def get_state_tensor_replay(self, obs, instruction=None):
    #     state_tensor = torch.tensor(obs['effector_target_translation']).unsqueeze(0)
    #     for block in self.env._get_urdf_paths().keys():
    #         if block in self.env._blocks_on_table:
    #             block_id = self.env._block_to_pybullet_id[block]
    #             block_pos_quat_tup = self.env._pybullet_client.getBasePositionAndOrientation(block_id)
    #             block_pos = torch.tensor(block_pos_quat_tup[0]).unsqueeze(0)
    #             block_quat = torch.tensor(block_pos_quat_tup[1]).unsqueeze(0)
    #             state_tensor = torch.cat((state_tensor,
    #                                       block_pos,
    #                                       block_quat), dim=1)
    #     if instruction:
    #         instruction_tensor = self.get_text_embedding(instruction).cpu()
    #         return torch.cat((state_tensor, instruction_tensor), dim=1)
    #     return state_tensor

    # def play_cloned_trajs(self, instruction=None, episode_num=None, video_file_path=None):
    #     self.load_policy(POLICY_FILE_PATH)
    #     self.policy.eval()
    #     obs = self.env.reset()
    #     store_data = False

    #     if episode_num != None:
    #         episode = list(self.custom_ds.values())[episode_num]
    #         init_state = episode['init_state']
    #         self.env.set_pybullet_state(init_state)
    #         instruction = episode['instruction']
    #         # self.env._reward_calculator._start_block = episode['start_block']
    #         # self.env._reward_calculator._target_block = episode['target_block']
    #         obs = self.env._compute_observation()
    #         print("instruction: ", instruction)
    #         self.env._reward_calculator._start_block = self.start_target[episode_num]['start']
    #         self.env._reward_calculator._target_block = self.start_target[episode_num]['target']
    #         print("start: ", self.env._reward_calculator._start_block)
    #         print("target: ", self.env._reward_calculator._target_block)
    #     else:
    #         init_state = self.env.get_pybullet_state()
    #         states = collections.defaultdict(list)
    #         actions = []
    #         store_data = True
    #         if not instruction:
    #             instruction = self.env._instruction_str
    #             store_data = False
    #         print("instruction: ", instruction)

    #     qkey = ord('q')
    #     steps = 0
    #     frames = []
    #     total_reward = 0
    #     done = False
    #     while steps < 200 and not done:
    #         keys = self.env._pybullet_client.getKeyboardEvents()
    #         if qkey in keys and keys[qkey]&self.env._pybullet_client.KEY_WAS_TRIGGERED:
    #             break
    #         state_tensor = self.get_state_tensor_replay(obs, instruction).to(device)
    #         pred_action = self.policy(state_tensor).detach().cpu().numpy().reshape(-1)

    #         if store_data:
    #             states['effector_target_translation'].append(obs['effector_target_translation'])
    #             for block in self.env._get_urdf_paths().keys():
    #                 if block in self.env._blocks_on_table:
    #                     block_id = self.env._block_to_pybullet_id[block]
    #                     block_pos_quat_tup = self.env._pybullet_client.getBasePositionAndOrientation(block_id)
    #                     block_pos = np.array(block_pos_quat_tup[0])
    #                     block_quat = np.array(block_pos_quat_tup[1])
    #                     block_pos_quat_np = np.concatenate((block_pos, block_quat), axis=0)
    #                     states[block].append(block_pos_quat_np)
    #             actions.append(pred_action)

    #         delta = self.conv_arr_delta(pred_action).reshape(-1)
    #         obs, reward, done, _, =self.env.step(delta)
    #         total_reward += reward
    #         steps += 1
    #         if video_file_path:
    #             frames.append(obs['rgb'])

    #     if video_file_path:
    #         self.col_trajs.create_videos(frames=frames,
    #                                 idx=1, 
    #                                 instruction=instruction, 
    #                                 video_file_path=video_file_path)

    #     if store_data:
    #         states = {key: np.array(values) for key, values in states.items()}
    #         return init_state, states, np.array(actions), total_reward / steps

    #     return None, None, None, total_reward / steps
    
    # def evaluate_policy(self, from_train_dist=True):
    #     print("from training distribution: ", from_train_dist)
    #     print("model size: ", self.h1_size)
    #     num_rollouts = 20
    #     total_reward = 0
    #     for episode_num in range(num_rollouts):
    #         if from_train_dist:
    #             _, _, _, episode_reward = self.play_cloned_trajs(episode_num=episode_num)
    #         else:
    #             _, _, _, episode_reward = self.play_cloned_trajs()
    #         print("episode reward: ", episode_reward)
    #         total_reward += episode_reward
    #     return total_reward / num_rollouts
    
    # def get_inst_start_target_blocks(self):
    #     self.start_target = {}
    #     for episode_num in range(20):
    #         episode = list(self.custom_ds.values())[episode_num]
    #         if episode_num == 0:
    #             start = "yellow_pentagon"
    #             target = "blue_cube"
    #         if episode_num == 1:
    #             start = "yellow_pentagon"
    #             target = "green_star"
    #         if episode_num == 2:
    #             start = "red_moon"
    #             target = "yellow_pentagon"
    #         if episode_num == 3:
    #             start = "blue_cube"
    #             target = "green_star"
    #         if episode_num == 4:
    #             start = "red_moon"
    #             target = "yellow_pentagon"
    #         if episode_num == 5:
    #             start = "red_moon"
    #             target = "green_star"
    #         if episode_num == 6:
    #             start = "yellow_pentagon"
    #             target = "blue_cube"
    #         if episode_num == 7:
    #             start = "red_moon"
    #             target = "green_star"
    #         if episode_num == 8:
    #             start = "green_star"
    #             target = "yellow_pentagon"
    #         if episode_num == 9:
    #             start = "red_moon"
    #             target = "blue_cube"
    #         if episode_num == 10:
    #             start = "yellow_pentagon"
    #             target = "red_moon"
    #         if episode_num == 11:
    #             start = "green_star"
    #             target = "yellow_pentagon"
    #         if episode_num == 12:
    #             start = "green_star"
    #             target = "yellow_pentagon"
    #         if episode_num == 13:
    #             start = "blue_cube"
    #             target = "yellow_pentagon"
    #         if episode_num == 14:
    #             start = "blue_cube"
    #             target = "green_star"
    #         if episode_num == 15:
    #             start = "blue_cube"
    #             target = "green_star"
    #         if episode_num == 16:
    #             start = "green_star"
    #             target = "yellow_pentagon"
    #         if episode_num == 17:
    #             start = "blue_cube"
    #             target = "green_star"
    #         if episode_num == 18:
    #             start = "yellow_pentagon"
    #             target = "blue_cube"
    #         if episode_num == 19:
    #             start = "blue_cube"
    #             target = "yellow_pentagon"
    #         episode_dict = {
    #             "instuction": episode['instruction'],
    #             "start": start,
    #             "target": target,
    #         }
    #         self.start_target[episode_num] = episode_dict