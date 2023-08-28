import torch
import torch.nn as nn
from train_bc_w_lang import BCAgent
import os
import pickle
import random
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
import wandb

ltable_path = '../'
sys.path.append(ltable_path)
from environments import blocks
from environments import language_table


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(42)


TC_DATA_FILENAME = "../pickle_files/tc_data.pickle"
INST_FILENAME = "../pickle_files/lang_inst.pickle"
TAU1_VIDEOS_FILENAME = "../videos/tc_videos/tau_1"
TAU2_VIDEOS_FILENAME = "../videos/tc_videos/tau_2"
ENC_FILE_PATH = "../models/tc_networks/lstm_enc.pth"
DEC_FILE_PATH = "../models/tc_networks/policy_dec.pth"

TRAIN_TEST_SPLIT = 0.9
BATCH_SIZE = 32
NUM_EPOCHS = 50000
# NUM_EPOCHS = 1000
LEARNING_RATE = 1e-3
MIN_LR  = 1e-6
LR_MULTIPLIER = 0.9999


class TCDataset(Dataset):
    def __init__(self, tau_1, states, actions, corrections) -> None:
        self.tau_1 = tau_1
        self.states = states
        self.actions = actions
        self.corrections = corrections

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.tau_1[idx], self.states[idx], self.actions[idx], self.corrections[idx] 
    
class LstmEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers) -> None:
        super(LstmEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.z = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_tensor):
        h0 = torch.zeros(self.num_layers, input_tensor.size(0), self.hidden_size).to(device=device)
        c0 = torch.zeros(self.num_layers, input_tensor.size(0), self.hidden_size).to(device=device)
        lstm_out, _ = self.lstm(input_tensor, (h0, c0))
        lstm_out = lstm_out[:, -1, :]
        z = self.z(lstm_out)
        return z

class PolicyDecoder(nn.Module):
    def __init__(self, input_size, h1_size, h2_size, output_size):
        super(PolicyDecoder, self).__init__()
        self.policy_dec = nn.Sequential(
            nn.Linear(input_size, h1_size),
            nn.ReLU(),
            nn.Linear(h1_size, h2_size),
            nn.ReLU(),
            nn.Linear(h2_size, output_size)
        )
    
    def forward(self, input_tensor):
        return self.policy_dec(input_tensor)

class TrajCorr(nn.Module):
    def __init__(self, env) -> None:
        super(TrajCorr, self).__init__()
        self.env = env
        self.bc_agent = BCAgent(self.env)
        self.load_tc_ds()

        self.lstm_output_size = 32
        lstm_input_size, policy_dec_input_size = self.get_input_tensor_sizes()
        self.lstm_encoder = LstmEncoder(input_size=lstm_input_size,
                                        hidden_size=128,
                                        output_size=self.lstm_output_size,
                                        num_layers=5).to(device)
        self.policy_decoder = PolicyDecoder(input_size=policy_dec_input_size,
                                            h1_size=128,
                                            h2_size=128,
                                            output_size=9).to(device)
        self.learning_rate = LEARNING_RATE
        self.loss = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(list(self.lstm_encoder.parameters()) + list(self.policy_decoder.parameters()),
                                      lr=LEARNING_RATE)
    
    def load_tc_ds(self):
        if os.path.isfile(TC_DATA_FILENAME):
            with open(TC_DATA_FILENAME, 'rb') as file:
                self.tc_ds = pickle.load(file)
        else:
            self.tc_ds = {}
        if not self.tc_ds:
            self.tc_ds = {}

    def get_policy_input_size(self, episode):
        states = episode['tau1_info']['tau1_states']
        return states["effector_target_translation"].shape[1] + \
                states['red_moon'].shape[1] * 4
        
    def get_input_tensor_sizes(self):
        episode_0 = list(self.tc_ds.values())[0]
        states_size = self.get_policy_input_size(episode_0)
        actions_size = episode_0['tau1_info']['tau1_actions'].shape[1]
        correction = episode_0['correction']
        return states_size + actions_size, \
            states_size + self.bc_agent.get_text_embedding(correction).shape[1] + self.lstm_output_size

    def get_state_tensor(self, states, step):
        for iter, key in enumerate(states.keys()):
            if iter == 0:
                state_tensor = torch.tensor(states[key][step, :])
            else:
                state_tensor = torch.cat((state_tensor,
                                            torch.tensor(states[key][step, :])),
                                            dim = 0)
        return state_tensor.unsqueeze(0)

    def get_tau1_tensor(self, episode):
        tau1_actions = episode['tau1_info']['tau1_actions']
        num_time_steps = tau1_actions.shape[0]
        for step in range(num_time_steps):
            if step == 0:
                state_tensor = self.get_state_tensor(episode['tau1_info']['tau1_states'], step)
            else:
                state_tensor = torch.cat((state_tensor,
                                          self.get_state_tensor(episode['tau1_info']['tau1_states'], step)),
                                          dim=0)
        return torch.cat((state_tensor, torch.tensor(tau1_actions)), dim=-1)
    
    def get_dataset_pairs(self, episode_num):
        episode = list(self.tc_ds.values())[episode_num]
        tau2_actions = episode['tau2_info']['tau2_actions']
        num_time_steps = tau2_actions.shape[0]
        tau_1_tensor = self.get_tau1_tensor(episode).unsqueeze(0).repeat(num_time_steps,1,1)
        for step in range(num_time_steps):
            if step == 0:
                state_tensor = self.get_state_tensor(episode['tau2_info']['tau2_states'], step)
            else:
                state_tensor = torch.cat((state_tensor,
                                          self.get_state_tensor(episode['tau2_info']['tau2_states'], step)),
                                          dim=0)
        correction_tensor = self.bc_agent.get_text_embedding(episode['correction']).repeat(num_time_steps, 1)
        return tau_1_tensor, state_tensor, torch.tensor(tau2_actions), correction_tensor
    
    def get_index_arr(self, length):
        index_list = list(range(length))
        random.shuffle(index_list)
        return np.array(index_list)
    
    def process_data(self):
        for iter, _ in enumerate(self.tc_ds.values()):
            tau_1, states, actions, corrections = self.get_dataset_pairs(iter)
            if iter == 0:
                tau_1_tensor = tau_1
                state_tensor = states
                action_tensor = actions
                correction_tensor = corrections
            else:
                tau_1_tensor = torch.cat((tau_1_tensor,
                                              tau_1),
                                              dim=0)
                state_tensor = torch.cat((state_tensor,
                                          states),
                                          dim=0)
                action_tensor = torch.cat((action_tensor,
                                          actions),
                                          dim=0)
                correction_tensor = torch.cat((correction_tensor,
                                               corrections),
                                               dim=0)
        indices = self.get_index_arr(state_tensor.shape[0])
        tau_1_tensor = tau_1_tensor[indices].type(torch.float32)
        state_tensor = state_tensor[indices].type(torch.float32)
        action_tensor = action_tensor[indices].type(torch.float32)
        correction_tensor = correction_tensor[indices].type(torch.float32)

        sep_idx = int(TRAIN_TEST_SPLIT * state_tensor.shape[0])
        train_tau_1, train_states, train_actions, train_corrections = \
            tau_1_tensor[:sep_idx], state_tensor[:sep_idx], action_tensor[:sep_idx], correction_tensor[:sep_idx]
        test_tau_1, test_states, test_actions, test_corrections = \
            tau_1_tensor[sep_idx:], state_tensor[sep_idx:], action_tensor[sep_idx:], correction_tensor[sep_idx:]
        print("tau_1_tensor: ", tau_1_tensor.shape)
        print("train_tau_1: ", train_tau_1.shape)
        print("test_tau_1: ", test_tau_1.shape)

        self.full_loader = DataLoader(TCDataset(tau_1_tensor, state_tensor, action_tensor, correction_tensor), 
                                       batch_size=BATCH_SIZE, 
                                       shuffle=False)
        self.train_loader = DataLoader(TCDataset(train_tau_1, train_states, train_actions, train_corrections), 
                                       batch_size=BATCH_SIZE, 
                                       shuffle=True)
        self.test_loader = DataLoader(TCDataset(test_tau_1, test_states, test_actions, test_corrections), 
                                       batch_size=BATCH_SIZE, 
                                       shuffle=True)

    def train_one_epoch(self):
        running_loss = 0
        self.lstm_encoder.train(True)
        self.policy_decoder.train(True)
        for i, batch in enumerate(self.train_loader):
            tau_1, states, ref_actions, correction = batch
            tau_1 = tau_1.to(device)
            states = states.to(device)
            ref_actions = ref_actions.to(device)
            correction = correction.to(device)
            # print("tau_1: ", tau_1.shape)
            # print("states: ", states.shape)
            # print("ref_actions: ", ref_actions.shape)
            # print("correction: ", correction.shape)

            z = self.lstm_encoder.forward(tau_1)
            # print("z: ", z.shape)

            input_tensor = torch.cat((states, z, correction), dim=-1)
            # print("input_tensor: ", input_tensor.shape)
            pred_actions = self.policy_decoder.forward(input_tensor)
            # print("pred_actions: ", pred_actions.shape)

            loss = self.loss(pred_actions, ref_actions)
            running_loss += loss.item()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        return running_loss / (i + 1)

    def eval(self):
        running_loss = 0
        self.lstm_encoder.eval()
        self.policy_decoder.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                tau_1, states, ref_actions, correction = batch
                tau_1 = tau_1.to(device)
                states = states.to(device)
                ref_actions = ref_actions.to(device)
                correction = correction.to(device)
                z = self.lstm_encoder.forward(tau_1)
                input_tensor = torch.cat((states, z, correction), dim=-1)
                pred_actions = self.policy_decoder.forward(input_tensor)
                loss = self.loss(pred_actions, ref_actions)
                running_loss += loss.item()
        return running_loss / (i + 1)
    
    def setup_wandb(self):
        wandb.login()
        self.run = wandb.init(
            project="tc_with_language",
            config={
                "architecture": "mlp",
                "init_lr": self.learning_rate,
                "min_lr": MIN_LR,
                "epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr_multiplier": LR_MULTIPLIER,
                "num_trajs": len(self.tc_ds.keys()),
            })

    def calc_accuracies(self, dataset=None):
        if dataset == "train":
            loader = self.train_loader
        elif dataset == "test":
            loader = self.test_loader
        elif dataset == "full":
            loader = self.full_loader
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                tau_1, states, ref_actions, correction = batch
                tau_1 = tau_1.to(device)
                states = states.to(device)
                ref_actions = ref_actions.to(device)
                correction = correction.to(device)
                z = self.lstm_encoder.forward(tau_1)
                input_tensor = torch.cat((states, z, correction), dim=-1)
                pred_actions = self.policy_decoder.forward(input_tensor)

                delta_preds = self.bc_agent.get_deltas(pred_actions.cpu().numpy())
                delta_refs = self.bc_agent.get_deltas(ref_actions.cpu().numpy())

                total += states.shape[0]
                correct += np.all(delta_refs == delta_preds, axis=1).sum()

        return correct / total

    def train_tc(self):
        self.setup_wandb()
        self.process_data()
        for epoch in range(NUM_EPOCHS):
            train_loss = self.train_one_epoch()
            test_loss = self.eval()

            self.run.log({"train loss": train_loss,
                        "test loss": test_loss,
                        "epochs": epoch})
            if epoch % 100 == 0:
                print("epochs: ", epoch, "lr: ", self.learning_rate, 
                        "train loss: ", train_loss, "test loss: ", test_loss)

            if self.learning_rate > MIN_LR:
                self.learning_rate *= LR_MULTIPLIER
                for param_group in self.optim.param_groups:
                    param_group['lr'] = self.learning_rate  

        train_accuracy = self.calc_accuracies(dataset="train")
        test_accuracy = self.calc_accuracies(dataset="test")
        full_accuracy = self.calc_accuracies(dataset="full")            
        table = wandb.Table(columns=["Training Accuracy", "Testing Accuracy", "Full Accuracy"],
                            data=[[train_accuracy, test_accuracy, full_accuracy]])
        self.run.log({"table": table})
        wandb.finish()

        torch.save(self.lstm_encoder.state_dict(), ENC_FILE_PATH)
        torch.save(self.policy_decoder.state_dict(), DEC_FILE_PATH)
    
    def test_tc(self, episode_num=None, video_file_path=None):
        self.lstm_encoder.load_state_dict(torch.load(ENC_FILE_PATH))
        self.policy_decoder.load_state_dict(torch.load(DEC_FILE_PATH))

        episode = list(self.tc_ds.values())[episode_num]
        init_state = episode['init_state']
        instruction = episode['instruction']
        self.env.set_pybullet_state(init_state)
        obs = self.env._compute_observation()
        print("\ninstruction: ", instruction)
        print("correction: ", episode['correction'] + "\n")

        tau_1 = self.get_tau1_tensor(episode).type(torch.float32).unsqueeze(0).to(device)
        z = self.lstm_encoder.forward(tau_1)
        correction_tensor = self.bc_agent.get_text_embedding(episode['correction'])

        qkey = ord('q')
        steps = 0
        frames = []
        while steps < 200:
            keys = self.env._pybullet_client.getKeyboardEvents()
            if qkey in keys and keys[qkey]&self.env._pybullet_client.KEY_WAS_TRIGGERED:
                break
            state_tensor = self.bc_agent.get_state_tensor_replay(obs).to(device)
            input_tensor = torch.cat((state_tensor, z, correction_tensor), dim=-1)
            pred_action = self.policy_decoder(input_tensor).detach().cpu().numpy().reshape(-1)

            delta = self.bc_agent.conv_arr_delta(pred_action).reshape(-1)
            obs, _, _, _, =self.env.step(delta)
            steps += 1

            if video_file_path:
                frames.append(obs['rgb'])

        if video_file_path:
            self.bc_agent.col_trajs.create_videos(frames=frames,
                                    idx=2, 
                                    instruction=instruction, 
                                    video_file_path=video_file_path)


if __name__=="__main__":
    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_4,
        control_frequency=10.0,
        render_mode="human"
    )

    traj_corr_obj = TrajCorr(env)
    traj_corr_obj.train_tc()
    # PRESENTATION_VIDEOS_FILENAME = "../videos/presentation"
    # traj_corr_obj.test_tc(0, video_file_path=PRESENTATION_VIDEOS_FILENAME)
    