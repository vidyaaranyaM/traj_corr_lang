import tensorflow_datasets as tfds
import os
import sys
import cv2
import numpy as np
import pickle
import collections

ltable_path = '../'
sys.path.append(ltable_path)
from environments import blocks
from environments import language_table
from environments.rewards import block2block

import utils


DATASET_VERSION = '0.0.1'
DATASET_NAME = 'language_table_blocktoblock_4block_sim'
DATASET_DIRS = {
    'language_table': 'gs://gresearch/robotics/language_table',
    'language_table_sim': 'gs://gresearch/robotics/language_table_sim',
    'language_table_blocktoblock_sim': 'gs://gresearch/robotics/language_table_blocktoblock_sim',
    'language_table_blocktoblock_4block_sim': 'gs://gresearch/robotics/language_table_blocktoblock_4block_sim',
    'language_table_blocktoblock_oracle_sim': 'gs://gresearch/robotics/language_table_blocktoblock_oracle_sim',
    'language_table_blocktoblockrelative_oracle_sim': 'gs://gresearch/robotics/language_table_blocktoblockrelative_oracle_sim',
    'language_table_blocktoabsolute_oracle_sim': 'gs://gresearch/robotics/language_table_blocktoabsolute_oracle_sim',
    'language_table_blocktorelative_oracle_sim': 'gs://gresearch/robotics/language_table_blocktorelative_oracle_sim',
    'language_table_separate_oracle_sim': 'gs://gresearch/robotics/language_table_separate_oracle_sim',
}
BC_DATA_FILENAME = "../pickle_files/bc_data_one_task.pickle"
BC_VIDEOS_FILENAME = "../videos/bc_videos/one_task"
PAD_SA_LEN = 10
SEED = 42
EPS = 0.5


class CollectBCTrajs:
    def __init__(self, env, noise_injection=True, one_task=False) -> None:
        self.env = env
        self.noise_inj = noise_injection
        self.one_task = one_task
        self.load_dataset()
        self.load_custom_dataset()

    def load_dataset(self):
        dataset_path = os.path.join(DATASET_DIRS[DATASET_NAME], DATASET_VERSION)
        self.ds = tfds.builder_from_directory(dataset_path).as_dataset(split='train', shuffle_files=True)
    
    def load_custom_dataset(self):
        if os.path.isfile(BC_DATA_FILENAME):
            with open(BC_DATA_FILENAME, 'rb') as file:
                self.custom_dataset = pickle.load(file)
        else:
            self.custom_dataset = {}
        if not self.custom_dataset:
            self.custom_dataset = {}
    
    def save_trajs(self):
        print("Saving Trajectories")
        with open(BC_DATA_FILENAME, 'wb') as file:
            pickle.dump(self.custom_dataset, file)
    
    def decode_inst(self, inst):
        return bytes(inst[np.where(inst != 0)].tolist()).decode("utf-8")

    def create_videos(self, frames, idx, instruction, video_file_path):
        height, width, _ = frames[0].shape
        video_path = video_file_path + "/demo" + str(idx) + ".mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, 25.0, (width, height))

        title_text = instruction
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        font_color = (0, 0, 255)
        font_thickness = 1

        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(frame, title_text, (25, 25), font_face, font_scale, font_color, font_thickness)
            video_writer.write(frame)

        video_writer.release()
        print("Video saved successfully.")

    def clear_trajs(self):
        self.custom_dataset = {}
        self.save_trajs()
        video_dir = BC_VIDEOS_FILENAME
        for file in os.listdir(video_dir):
            os.remove(os.path.join(video_dir, file))
    
    def choose_delta_action(self, base_delta):
        if self.noise_inj:
            idx = utils.convert_deltas(base_delta).nonzero()[0][0]
            p = np.random.random()
            if p < EPS:
                return utils.sample_delta(idx)
            else:
                return base_delta, utils.convert_deltas(base_delta)
        else:
            return base_delta, utils.convert_deltas(base_delta)

    def start_teleop(self, obs):
        dkey = ord('d')
        fkey = ord('f')
        actions_list = []
        states = collections.defaultdict(list)
        init_state = self.env.get_pybullet_state()
        frames = []
        while True:
            keys = self.env._pybullet_client.getKeyboardEvents()

            store_data = True
            if self.env._pybullet_client.B3G_UP_ARROW in keys and self.env._pybullet_client.B3G_LEFT_ARROW in keys:
                delta, action = self.choose_delta_action(np.array([-0.01, -0.01]))
            elif self.env._pybullet_client.B3G_UP_ARROW in keys and self.env._pybullet_client.B3G_RIGHT_ARROW in keys:
                delta, action = self.choose_delta_action(np.array([-0.01, 0.01]))
            elif self.env._pybullet_client.B3G_DOWN_ARROW in keys and self.env._pybullet_client.B3G_LEFT_ARROW in keys:
                delta, action = self.choose_delta_action(np.array([0.01, -0.01]))
            elif self.env._pybullet_client.B3G_DOWN_ARROW in keys and self.env._pybullet_client.B3G_RIGHT_ARROW in keys:
                delta, action = self.choose_delta_action(np.array([0.01, 0.01]))
            elif self.env._pybullet_client.B3G_UP_ARROW in keys:
                delta, action = self.choose_delta_action(np.array([-0.01, 0]))
            elif self.env._pybullet_client.B3G_DOWN_ARROW in keys:
                delta, action = self.choose_delta_action(np.array([0.01, 0]))
            elif self.env._pybullet_client.B3G_LEFT_ARROW in keys:
                delta, action = self.choose_delta_action(np.array([0, -0.01]))
            elif self.env._pybullet_client.B3G_RIGHT_ARROW in keys:
                delta, action = self.choose_delta_action(np.array([0, 0.01]))
            elif dkey in keys and keys[dkey]&self.env._pybullet_client.KEY_WAS_TRIGGERED:
                for _ in range(PAD_SA_LEN):
                    actions_list.append(utils.convert_deltas(np.array([0, 0])))
                    for block in states.keys():
                        states[block].append(states[block][-1])
                states = {key: np.array(values) for key, values in states.items()}
                return True, init_state, states, np.array(actions_list), frames
            elif fkey in keys and keys[fkey]&self.env._pybullet_client.KEY_WAS_TRIGGERED:
                return False, None, None, None, None
            else:
                delta = np.array([0, 0])
                utils.convert_deltas(delta)
                store_data = False
            if store_data:
                states['effector_target_translation'].append(obs['effector_target_translation'])
                for block in self.env._get_urdf_paths().keys():
                    if block in self.env._blocks_on_table:
                        block_id = self.env._block_to_pybullet_id[block]
                        block_pos_quat_tup = self.env._pybullet_client.getBasePositionAndOrientation(block_id)
                        block_pos = np.array(block_pos_quat_tup[0])
                        block_quat = np.array(block_pos_quat_tup[1])
                        block_pos_quat_np = np.concatenate((block_pos, block_quat), axis=0)
                        states[block].append(block_pos_quat_np)
                actions_list.append(action)
                frames.append(obs['rgb'])
            obs, _, _, _ = self.env.step(delta)

    def collect_human_demos(self, replace=False, save_data=True):
        if not save_data:
            print("\nNote: You are not saving trajectories\n")
        if replace:
            episode_num = int(input("Enter the id of episode you want to replace:\n"))
            episode = list(self.custom_dataset.values())[episode_num]
            instruction = episode['instruction']
            start_block = episode['start_block']
            target_block = episode['target_block']
            num_trajs = 1
        else:
            start_idx = len(self.custom_dataset.keys())
            num_trajs = int(input("Enter the number of trajs you want to record: \n"))
            if self.one_task:
                start_block = "green_star"
                target_block = "red_moon"
                instruction =  self.env._reward_calculator._sample_instruction(start_block,
                                                                              target_block,
                                                                              self.env._blocks_on_table)
        ykey = ord('y')
        nkey = ord('n')
        count = 0
        flag = True
        while count < num_trajs:
            keys = self.env._pybullet_client.getKeyboardEvents()
            
            if flag:
                obs = self.env.reset()
                if not replace:
                    if not self.one_task:
                        instruction = self.env._instruction_str
                        start_block = self.env._reward_calculator._start_block
                        target_block = self.env._reward_calculator._target_block
                print(f"\ninstruction: {instruction}")
                print("press \"y\" to start recording trajectory or \"n\" to reset env \n")
                flag = False

            if ykey in keys and keys[ykey]&self.env._pybullet_client.KEY_WAS_TRIGGERED:
                print("start moving the robot \n")
                flag = True
                success, init_state, states, actions, frames = self.start_teleop(obs)
                if success:
                    episode_dict = {
                        "init_state": init_state, 
                        "states": states,
                        "actions": actions,
                        "instruction": instruction,
                        "start_block": start_block,
                        "target_block": target_block,
                    }
                    if save_data:
                        if replace:
                            self.custom_dataset["episode_" + str(episode_num)] = episode_dict
                            self.create_videos(frames, episode_num, instruction, BC_VIDEOS_FILENAME)
                        else:
                            self.custom_dataset["episode_" + str(count + start_idx)] = episode_dict
                            self.create_videos(frames, count + start_idx, instruction, BC_VIDEOS_FILENAME)
                    count += 1

            if nkey in keys and keys[nkey]&self.env._pybullet_client.KEY_WAS_TRIGGERED:
                print("resetting the env")
                flag = True
                
        if save_data:
            self.save_trajs()

    def playback_trajs(self, episode_num):
        self.load_custom_dataset()
        self.env.reset()
        episode = list(self.custom_dataset.values())[episode_num]
        init_state = episode['init_state']
        self.env.set_pybullet_state(init_state)

        print("episode: ", episode['instruction'])
        qkey = ord('q')
        num_steps = episode['actions'].shape[0]
        for step in range(num_steps):
            keys = self.env._pybullet_client.getKeyboardEvents()
            if qkey in keys and keys[qkey]&self.env._pybullet_client.KEY_WAS_TRIGGERED:
                break
            action = utils.convert_actions(episode['actions'][step, :])
            self.env.step(action)
        return init_state
    
    def delete_traj(self, episode_indices):
        for idx in episode_indices:
            episode = "episode_" + str(idx)
            if episode in self.custom_dataset:
                del self.custom_dataset[episode]
            video_file = os.path.join(BC_VIDEOS_FILENAME, "demo" + str(idx) + ".mp4")
            if os.path.exists(video_file):
                os.remove(video_file)
        self.save_trajs()
         

if __name__=="__main__":
    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_4,
        reward_factory=block2block.BlockToBlockReward ,
        control_frequency=10.0,
        render_mode="human"
    )
    collect_trajs = CollectBCTrajs(env, noise_injection=True, one_task=True)
    collect_trajs.collect_human_demos(replace=False, save_data=True)
