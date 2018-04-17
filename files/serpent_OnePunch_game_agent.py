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
            "PUNCH RIGHT": [KeyboardKey.KEY_RIGHT]
        }

        self.frame_buffer = None

        self.run_count = 0
        self.run_reward = 0

        self.observation_count = 0
        self.episode_observation_count = 0

        self.performed_inputs = collections.deque(list(), maxlen=8)

        self.reward_10 = collections.deque(list(), maxlen=10)
        self.reward_100 = collections.deque(list(), maxlen=100)
        self.reward_1000 = collections.deque(list(), maxlen=1000)

        self.rewards = list()

        self.average_reward_10 = 0
        self.average_reward_100 = 0
        self.average_reward_1000 = 0

        self.top_reward = 0
        self.top_reward_run = 0

        self.death_check = False
        
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

            self.run_count += 1

            self.launch()
            self.click_main_menu()
            self.survival_select()
            self.skill_select()

            self.first_run = False

            self.episode_started_at = time.time()

            return None
        
        reward = self.calculate_reward(game_frame)

        if self.frame_buffer is not None:
            self.run_reward += reward

            self.observation_count += 1
            self.episode_observation_count += 1

            self.analytics_client.track(event_key="RUN_REWARD", data=dict(reward=reward))

            if self.ppo_agent.agent.batch_count == self.ppo_agent.agent.batch_size - 1:

                self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)
                self.ppo_agent.observe(reward, terminal=(self.is_game_over(game_frame)))
                self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)

                self.frame_buffer = None

                if not self.is_game_over(game_frame):
                    time.sleep(1)
                    return None
            else:
                self.ppo_agent.observe(reward, terminal=(self.is_game_over(game_frame)))

        else:
            self.ppo_agent.observe(reward, terminal=(self.is_game_over(game_frame)))

        if not self.is_game_over(game_frame):
            
            self.frame_buffer = FrameGrabber.get_frames([0, 1, 2, 3], frame_type="PIPELINE")

            action, label, action_input = self.ppo_agent.generate_action(self.frame_buffer)

            self.input_controller.handle_keys(action_input)

        else:

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

    def is_game_over(self, game_frame):

        death_check_region = serpent.cv.extract_region_from_image(game_frame.frame, self.game.screen_regions["DEATH_CHECK"])

        color_difference = skimage.color.deltaE_cie76(death_check_region[0][0], [49, 199, 247])

        if color_difference < 50:
            return False
        else:
            return True

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
        time.sleep(5)

    def score(self):
        time.sleep(5)
        self.input_controller.click_screen_region(screen_region = "SCORE_NEXT")
        time.sleep(2)

    def high_score(self):
        time.sleep(5)
        self.input_controller.click_screen_region(screen_region = "HIGHSCORE_NEXT")
        time.sleep(2)
        
    
    
    
    

    

    
    
    
        
