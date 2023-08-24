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
INST_FILENAME = "../pickle_files/lang_inst.pickle"
BC_DATA_FILENAME = "../pickle_files/bc_data.pickle"
BC_VIDEOS_FILENAME = "../videos/bc_videos"
PAD_SA_LEN = 10
SEED = 42


class CollectBCTrajs:
    def __init__(self, env) -> None:
        self.env = env
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
    
    def save_lang_instructions(self, num_episodes=None):
        if not num_episodes:
            num_episodes = len(self.ds)
    
        instructions_list = []
        for iter, episode in enumerate(self.ds):
            for step in episode['steps'].as_numpy_iterator():
                instructions_list.append(self.decode_inst(step['observation']['instruction']))
                break
            if iter >= num_episodes - 1:
                break

            print("iter: ", iter)
        
        with open(INST_FILENAME, 'wb') as file:
            pickle.dump(instructions_list, file)

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
                delta = np.array([-0.01, -0.01])
                # print("up left")
            elif self.env._pybullet_client.B3G_UP_ARROW in keys and self.env._pybullet_client.B3G_RIGHT_ARROW in keys:
                delta = np.array([-0.01, 0.01])
                # print("up right")
            elif self.env._pybullet_client.B3G_DOWN_ARROW in keys and self.env._pybullet_client.B3G_LEFT_ARROW in keys:
                delta = np.array([0.01, -0.01])
                # print("down left")
            elif self.env._pybullet_client.B3G_DOWN_ARROW in keys and self.env._pybullet_client.B3G_RIGHT_ARROW in keys:
                delta = np.array([0.01, 0.01])
                # print("down right")
            elif self.env._pybullet_client.B3G_UP_ARROW in keys:
                delta = np.array([-0.01, 0])
                # print("up")
            elif self.env._pybullet_client.B3G_DOWN_ARROW in keys:
                delta = np.array([0.01, 0])
                # print("down")
            elif self.env._pybullet_client.B3G_LEFT_ARROW in keys:
                delta = np.array([0, -0.01])
                # print("left")
            elif self.env._pybullet_client.B3G_RIGHT_ARROW in keys:
                delta = np.array([0, 0.01])
                # print("right")
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
                store_data = False
            action = utils.convert_deltas(delta)
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

    def collect_human_demos(self):
        start_idx = len(self.custom_dataset.keys())
        num_trajs = int(input("Enter the number of trajs you want to record: \n"))
        
        ykey = ord('y')
        nkey = ord('n')
        count = 0
        flag = True
        while count < num_trajs:
            keys = self.env._pybullet_client.getKeyboardEvents()
            
            if flag:
                obs = self.env.reset()
                instruction = self.env._instruction_str
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
                        "instruction": instruction 
                    }
                    self.custom_dataset["episode_" + str(count + start_idx)] = episode_dict
                    self.create_videos(frames, count + start_idx, instruction, BC_VIDEOS_FILENAME)
                    count += 1

            if nkey in keys and keys[nkey]&self.env._pybullet_client.KEY_WAS_TRIGGERED:
                print("resetting the env")
                flag = True
        
        self.save_trajs()
    
    def replace_traj(self, episode_num):
        episode = list(self.custom_dataset.values())[episode_num]
        instruction = episode['instruction']
        ykey = ord('y')
        nkey = ord('n')
        count = 0
        flag = True
        while count < 1:
            keys = self.env._pybullet_client.getKeyboardEvents()
        
            if flag:
                obs = self.env.reset()
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
                        "instruction": instruction 
                    }
                    self.custom_dataset["episode_" + str(episode_num)] = episode_dict
                    self.create_videos(frames, episode_num, instruction, BC_VIDEOS_FILENAME)
                    count += 1

            if nkey in keys and keys[nkey]&self.env._pybullet_client.KEY_WAS_TRIGGERED:
                print("resetting the env")
                flag = True
        
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
    
    def test(self):
        print("len: ", len(self.custom_dataset.keys()))
    

if __name__=="__main__":
    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_4,
        reward_factory=block2block.BlockToBlockReward ,
        control_frequency=10.0,
        render_mode="human"
    )
    collect_trajs = CollectBCTrajs(env)
    collect_trajs.test()
    # collect_trajs.clear_trajs()
    # collect_trajs.collect_human_demos()
    # collect_trajs.playback_trajs(0)
    # collect_trajs.replace_traj(100)
    # collect_trajs.test()
