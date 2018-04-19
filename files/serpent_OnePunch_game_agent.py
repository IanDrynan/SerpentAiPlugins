from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey
from serpent.frame_grabber import FrameGrabber
from serpent.machine_learning.context_classification.context_classifiers.cnn_inception_v3_context_classifier import CNNInceptionV3ContextClassifier
from serpent.frame_transformation_pipeline import FrameTransformationPipeline
from serpent.frame_transformer import FrameTransformer

from .helpers.ppo import SerpentPPO
from .helpers.terminal_printer import TerminalPrinter

from datetime import datetime

import serpent.cv
import numpy as np
import random
import offshoot
import time
import skimage.color
import os
import pickle
import subprocess
import shlex
import collections


class SerpentOnePunchGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play
        self.frame_handlers["PLAY_BOT"] = self.handle_play_bot
        self.frame_handlers["PLAY_RANDOM"] = self.handle_play_random

        self.frame_handler_setups["PLAY"] = self.setup_play
        self.frame_handler_setups["PLAY_BOT"] = self.setup_play_bot
        self.frame_handler_setups["PLAY_RANDOM"] = self.setup_play_random

        self.prev_time = 0
        self.printer = TerminalPrinter()

    def setup_play(self):

        self.first_run = True

        self.game_inputs = {
            "PUNCH LEFT": [KeyboardKey.KEY_LEFT],
            "PUNCH RIGHT": [KeyboardKey.KEY_RIGHT],
            "WAIT": []
        }

        self.frame_buffer = None

        self.run_count = 0
        self.run_reward = 0

        self.observation_count = 0
        self.episode_observation_count = 0

        self.input_history = collections.deque(list(), maxlen=8)

        self.reward_10 = collections.deque(list(), maxlen=10)
        self.reward_100 = collections.deque(list(), maxlen=100)
        self.reward_1000 = collections.deque(list(), maxlen=1000)

        self.rewards = list()

        self.average_reward_10 = 0
        self.average_reward_100 = 0
        self.average_reward_1000 = 0

        self.top_reward = 0
        self.top_reward_run = 0
        

        self.ppo_agent = SerpentPPO(
            frame_shape=(100, 100, 4),
            game_inputs=self.game_inputs
        )

        try:
            self.ppo_agent.agent.restore_model(directory=os.path.join(os.getcwd(), "datasets", "OnePunchAi"))
            self.restore_metadata()
        except Exception:
            pass

        self.analytics_client.track(event_key="INITIALIZE", data=dict(episode_rewards=[]))

        for reward in self.rewards:
            self.analytics_client.track(event_key="EPISODE_REWARD", data=dict(reward=reward))
            time.sleep(0.01)

        game_frame_buffer = FrameGrabber.get_frames([0, 1, 2, 3], frame_type="PIPELINE")
        self.ppo_agent.generate_action(game_frame_buffer)

        self.started_at = datetime.utcnow().isoformat()
        self.episode_started_at = None

    def handle_play(self, game_frame):

        if self.first_run:
            
            self.first_run = False
            self.run_count += 1

            self.launch()
            self.click_main_menu()
            self.survival_select()
            self.skill_select()

            self.episode_started_at = time.time()
            return None
        
        if self.is_paused(game_frame):
            return None
        
        self.printer.add("One Finger Death Punch - PPO agent")
        self.printer.add(f"Started - {self.started_at}")
        self.printer.add(f"Run count - {self.run_count}")
        self.printer.add("")
        reward = self.calculate_reward(game_frame)

        self.printer.add(f"Total run reward - {self.run_reward}")
        self.printer.add(f"Current run reward - {reward}")
        self.printer.add("")
        self.printer.add(f"Time for this step : {time.time() - self.prev_time}")
        self.prev_time = time.time()
        self.printer.add("")
        if self.frame_buffer is not None:
            self.run_reward += reward

            self.observation_count += 1
            self.episode_observation_count += 1

            self.analytics_client.track(event_key="RUN_REWARD", data=dict(reward=reward))

            # Train agent every batch. Larger batches calculate a more accurate gradient but take longer and requires more memory
            if self.ppo_agent.agent.batch_count == self.ppo_agent.agent.batch_size - 1:
                
                self.printer.add("The batch has been completed and the agent is being updated")
                self.printer.flush()
                self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)
                self.ppo_agent.observe(reward, terminal=(self.is_game_over(game_frame)))
                self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)

                self.frame_buffer = None

                if not self.is_game_over(game_frame):
                    time.sleep(1)
                    return None
            else:
                self.ppo_agent.observe(reward, terminal=(self.is_game_over(game_frame)))
        
        self.printer.add(f"Observation count: {self.observation_count}")
        self.printer.add(f"Episode observation count: {self.episode_observation_count}")
        self.printer.add(f"Current batch count: {self.ppo_agent.agent.batch_count}")
        self.printer.add("")

        if not self.is_game_over(game_frame):
            
            self.printer.add(f"Average Rewards (Last 10 Runs): {round(self.average_reward_10, 2)}")
            self.printer.add(f"Average Rewards (Last 100 Runs): {round(self.average_reward_100, 2)}")
            self.printer.add(f"Average Rewards (Last 1000 Runs): {round(self.average_reward_1000, 2)}")
            self.printer.add("")
            self.printer.add(f"Top Run Reward: {round(self.top_reward, 2)} (Run #{self.top_reward_run})")
            self.printer.add("")

            for inp in self.input_history:
                self.printer.add(inp)
            self.printer.flush()

            self.frame_buffer = FrameGrabber.get_frames([0, 1, 2, 3], frame_type="PIPELINE")

            action, label, action_input = self.ppo_agent.generate_action(self.frame_buffer)

            self.input_history.appendleft(label)
            self.input_controller.handle_keys(action_input)

        else:
            self.printer.flush()
            self.analytics_client.track(event_key="RUN_END", data=dict(run=self.run_count))
            self.run_count += 1

            self.reward_10.appendleft(self.run_reward)
            self.reward_100.appendleft(self.run_reward)
            self.reward_1000.appendleft(self.run_reward)

            self.rewards.append(self.run_reward)

            self.average_reward_10 = float(np.mean(self.reward_10))
            self.average_reward_100 = float(np.mean(self.reward_100))
            self.average_reward_1000 = float(np.mean(self.reward_1000))

            if self.run_reward > self.top_reward:
                self.top_reward = self.run_reward
                self.top_reward_run = self.run_count - 1

                self.analytics_client.track(event_key="NEW_RECORD", data=dict(type="REWARD", value=self.run_reward, run=self.run_count - 1))

            self.analytics_client.track(event_key="EPISODE_REWARD", data=dict(reward=self.run_reward))

            if not self.run_count % 10:
                    self.ppo_agent.agent.save_model(directory=os.path.join(os.getcwd(), "datasets", "OnePunchAi", "ppo_model"), append_timestep=False)
                    self.dump_metadata()

            self.run_reward = 0
            self.input_history.clear()
            self.frame_buffer = None

            time.sleep(10)
            self.input_controller.tap_key(KeyboardKey.KEY_ENTER, duration=1)

            self.survival_select()
            self.skill_select()

            self.episode_started_at = time.time()
            self.episode_observation_count = 0

    def calculate_reward(self, game_frame):
        
        if self.is_game_over(game_frame):

            return 0

        else:
            return .001
    
    def is_paused(self, game_frame):
        
        death_check_region = serpent.cv.extract_region_from_image(game_frame.frame, self.game.screen_regions["DEATH_CHECK"])
        pause_color_difference = skimage.color.deltaE_cie76(death_check_region[0][0], [5, 20, 25])

        if pause_color_difference < 20:
            self.printer.flush()
            self.printer.add("paused")
            self.printer.flush()
            return False

    def is_game_over(self, game_frame):
        
        death_check_region = serpent.cv.extract_region_from_image(game_frame.frame, self.game.screen_regions["DEATH_CHECK"])

        dead_color_difference = skimage.color.deltaE_cie76(death_check_region[0][0], [49, 199, 247])
        
        if dead_color_difference < 50:
            return False
        else:
            return True

    def dump_metadata(self):
        metadata = dict(
            started_at=self.started_at,
            run_count=self.run_count - 1,
            observation_count=self.observation_count,
            reward_10=self.reward_10,
            reward_100=self.reward_100,
            reward_1000=self.reward_1000,
            rewards=self.rewards,
            average_reward_10=self.average_reward_10,
            average_reward_100=self.average_reward_100,
            average_reward_1000=self.average_reward_1000,
            top_reward=self.top_reward,
            top_reward_run=self.top_reward_run
        )

        with open("datasets/OnePunchAi/metadata.json", "wb") as f:
            f.write(pickle.dumps(metadata))

    def restore_metadata(self):
        with open("datasets/OnePunchAi/metadata.json", "rb") as f:
            metadata = pickle.loads(f.read())

        self.started_at = metadata["started_at"]
        self.run_count = metadata["run_count"]
        self.observation_count = metadata["observation_count"]
        self.reward_10 = metadata["reward_10"]
        self.reward_100 = metadata["reward_100"]
        self.reward_1000 = metadata["reward_1000"]
        self.rewards = metadata["rewards"]
        self.average_reward_10 = metadata["average_reward_10"]
        self.average_reward_100 = metadata["average_reward_100"]
        self.average_reward_1000 = metadata["average_reward_1000"]
        self.top_reward = metadata["top_reward"]
        self.top_reward_run = metadata["top_reward_run"]

    def initialize_context_classifier(self):
        plugin_path = offshoot.config["file_paths"]["plugins"]

        context_classifier_path = f"{plugin_path}/SerpentOnePunchGameAgentPlugin/files/ml_models/context_classifier.model"

        context_classifier = CNNInceptionV3ContextClassifier(input_shape=(300, 400, 3))  #Replace with the shape (rows, cols, channels) of your captured context frames

        context_classifier.prepare_generators()
        context_classifier.load_classifier(context_classifier_path)

        self.machine_learning_models["context_classifier"] = context_classifier

    def setup_play_random(self):
        self.initialize_context_classifier()

        self.context_handlers = {"ClickMainMenu": self.click_main_menu,
                                 "HighScore": self.high_score,
                                 "InGame": self.in_game_random,
                                 "Launch": self.launch,
                                 "Score": self.score,
                                 "SKillSelect": self.skill_select,
                                 "SurvivalSelect": self.survival_select
                                 }  

    def handle_play_random(self, game_frame):

        context = self.handle_context(game_frame)

        if context == "InGame":
            return self.context_handlers[context](game_frame)
        if context is not None:
            return self.context_handlers[context]()
        

    def setup_play_bot(self):
        
        self.not_in_game = True
        self.swap = True
        self.initialize_context_classifier()

        self.context_handlers = {"ClickMainMenu": self.click_main_menu,
                                 "HighScore": self.high_score,
                                 "InGame": self.in_game_bot,
                                 "Launch": self.launch,
                                 "Score": self.score,
                                 "SKillSelect": self.skill_select,
                                 "SurvivalSelect": self.survival_select
                                 }
         
    def handle_play_bot(self, game_frame):
        
        if self.not_in_game:   
            context = self.handle_context(game_frame)

            if context == "InGame":

                self.in_game_bot(game_frame)
                self.not_in_game = False

            elif context is not None:
                self.context_handlers[context]()

        else:
            self.in_game_bot(game_frame)


    def resize_frame_for_bots(self, frame_to_resize, size):

        resized_frame = FrameTransformer.resize(frame_to_resize, size)

        return resized_frame
    
    
    def handle_context(self, game_frame):

        resized_frame = self.resize_frame_for_bots(game_frame.frame, "400x300")
        context = self.machine_learning_models["context_classifier"].predict(resized_frame)

        return context

    def in_game_random(self, game_frame):

        time.sleep(random.random()/2)

        if random.randint(0, 1):
            
            self.input_controller.tap_key(KeyboardKey.KEY_RIGHT)
            
        else:

            self.input_controller.tap_key(KeyboardKey.KEY_LEFT)

    def in_game_bot(self, game_frame):    
        resized_frame = self.resize_frame_for_bots(game_frame.frame, "400x300")

        self.swap = not self.swap
        if self.swap:
            if(self.check_left_color_diff(resized_frame)): self.input_controller.tap_key(KeyboardKey.KEY_LEFT)

            if(self.check_right_color_diff(resized_frame)): self.input_controller.tap_key(KeyboardKey.KEY_RIGHT)

        else:
            if(self.check_right_color_diff(resized_frame)): self.input_controller.tap_key(KeyboardKey.KEY_RIGHT)

            if(self.check_left_color_diff(resized_frame)): self.input_controller.tap_key(KeyboardKey.KEY_LEFT)

    def check_left_color_diff(self, resized_frame):
        color_distance_left = skimage.color.deltaE_cie76(resized_frame[147, 189], [24,138,181])

        if  color_distance_left < 100 or color_distance_left > 250:
            return True
        else:
            return False

    def check_right_color_diff(self, resized_frame):
        color_distance_right = skimage.color.deltaE_cie76(resized_frame[147, 210], [24,138,181])

        if color_distance_right < 100 or color_distance_right > 250:
            return True
        else:
            return False  



    def launch(self):
        self.input_controller.click_screen_region(screen_region = "LAUNCH_PLAY")
        time.sleep(20)

    def click_main_menu(self):
        self.input_controller.click_screen_region(screen_region = "CLICK_PLAY_AND_SURVIVAL")
        time.sleep(1)
        self.input_controller.click_screen_region(screen_region = "CLICK_PLAY_AND_SURVIVAL")
        time.sleep(2)
        self.input_controller.click_screen_region(screen_region = "CLICK_PLAY_AND_SURVIVAL")
        time.sleep(2)

    def survival_select(self):
        self.input_controller.click_screen_region(screen_region = "SURVIVAL_SELECT")
        time.sleep(2)

    def skill_select(self):
        self.input_controller.click_screen_region(screen_region = "SKILLS_NEXT")
        time.sleep(1)

    def score(self):
        time.sleep(5)
        self.input_controller.click_screen_region(screen_region = "SCORE_NEXT")
        time.sleep(2)

    def high_score(self):
        time.sleep(5)
        self.input_controller.click_screen_region(screen_region = "HIGHSCORE_NEXT")
        time.sleep(2)