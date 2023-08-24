import pickle
import os
from train_bc_w_lang import BCAgent
import random
import sys

ltable_path = '../'
sys.path.append(ltable_path)
from environments import blocks
from environments import language_table


TC_DATA_FILENAME = "../pickle_files/tc_data.pickle"
INST_FILENAME = "../pickle_files/lang_inst.pickle"
TAU1_VIDEOS_FILENAME = "../videos/tc_videos/tau_1"
TAU2_VIDEOS_FILENAME = "../videos/tc_videos/tau_2"


class CollectTCTrajs:
    def __init__(self, env) -> None:
        self.env = env
        self.bc_agent = BCAgent(self.env)
        self.load_tc_ds()
        self.load_instructions()
    
    def load_tc_ds(self):
        if os.path.isfile(TC_DATA_FILENAME):
            with open(TC_DATA_FILENAME, 'rb') as file:
                self.tc_ds = pickle.load(file)
        else:
            self.tc_ds = {}
        if not self.tc_ds:
            self.tc_ds = {}

    def load_instructions(self):
        with open(INST_FILENAME, 'rb') as file:
            self.instructions_list = pickle.load(file)
    
    def save_trajs(self):
        print("\nSaving Trajectories\n")
        with open(TC_DATA_FILENAME, 'wb') as file:
            pickle.dump(self.tc_ds, file)
    
    def clear_videos(self, file_path):
        video_dir = file_path
        for file in os.listdir(video_dir):
            os.remove(os.path.join(video_dir, file))

    def clear_tc_trajs(self):
        self.tc_ds = {}
        self.save_trajs()
        self.clear_videos(TAU1_VIDEOS_FILENAME)
        self.clear_videos(TAU2_VIDEOS_FILENAME)
    
    def prompt_for_coll_data(self):
        ykey = ord('y')
        nkey = ord('n')
        while True:
            keys = self.env._pybullet_client.getKeyboardEvents()
            if ykey in keys and keys[ykey]&self.env._pybullet_client.KEY_WAS_TRIGGERED:
                return True
            elif nkey in keys and keys[nkey]&self.env._pybullet_client.KEY_WAS_TRIGGERED:
                self.env.reset()
    
    def collect_single_traj(self, start_idx, video_filename, instruction=None, init_state=None):
        success = False
        if init_state:
            self.env.set_pybullet_state(init_state)
            obs = self.env._compute_observation()
            instruction = obs['instruction']
        else:
            obs = self.env._compute_observation()
        while not success:
            success, init_state, states, actions, frames = self.bc_agent.col_trajs.start_teleop(obs)
            if not success:
                if init_state:
                    self.env.set_pybullet_state(init_state)
                    obs = self.env._compute_observation()
                else:
                    obs = self.env.reset()
        self.bc_agent.col_trajs.create_videos(frames, start_idx, instruction, video_filename)
        return init_state, states, actions

    def collect_trajs(self, num_trajs, cloned_tau1=False):
        start_idx = len(self.tc_ds.keys())

        count = 0
        while count < num_trajs:
            if cloned_tau1:
                # sample language statement
                instruction = random.choice(self.instructions_list)
                print("\ninstruction: ", instruction)
                # play cloned trajectory
                init_state, tau1_states, tau1_actions = \
                    self.bc_agent.play_cloned_trajs(instruction=instruction, 
                                                    video_file_path=TAU1_VIDEOS_FILENAME)
            else:
                instruction = "push the blue block to the yellow block"
                print("\ninstruction: ", instruction)
                print("press \"y\" to start recording trajectory or \"n\" to reset env \n")
                if self.prompt_for_coll_data():
                    init_state, tau1_states, tau1_actions = self.collect_single_traj(start_idx + count, 
                                                                                    video_filename=TAU1_VIDEOS_FILENAME,
                                                                                    instruction=instruction)
            if cloned_tau1:
                print("\npress \'y\' to provide corrections or \'n\' if the trajectory is okay")
            else:
                print("\npress \"y\" to start recording trajectory\n")
            if self.prompt_for_coll_data():
                _, tau2_states, tau2_actions = self.collect_single_traj(start_idx + count, 
                                                                video_filename=TAU2_VIDEOS_FILENAME,
                                                                init_state=init_state)
            
            # correction = input("\nWrite how you changed the trajectory:\n")
            correction = "Move blue block closer to yellow block"
            tau1_info = {'tau1_states': tau1_states,
                        'tau1_actions': tau1_actions,}
            tau2_info = {'tau2_states': tau2_states,
                        'tau2_actions': tau2_actions,}
            episode_dict = {'init_state': init_state,
                            'instruction': instruction,
                            'correction': correction,
                            'tau1_info': tau1_info,
                            'tau2_info': tau2_info}

            self.tc_ds["episode_" + str(start_idx + count)] = episode_dict
            count += 1
            print("\nCollected one trajectory\n")
            self.env.reset()
        self.save_trajs()


if __name__=="__main__":
    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_4,
        control_frequency=10.0,
        render_mode="human"
    )

    collect_tc_trajs = CollectTCTrajs(env)
    collect_tc_trajs.clear_tc_trajs()
    collect_tc_trajs.collect_trajs(num_trajs=2)
