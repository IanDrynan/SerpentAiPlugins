from serpent.game import Game

from .api.api import OnePunchAPI

from serpent.utilities import Singleton


class SerpentOnePunchGame(Game, metaclass=Singleton):

    def __init__(self, **kwargs):
        kwargs["platform"] = "steam"

        kwargs["window_name"] = "One Finger Death Punch"

        kwargs["app_id"] = "264200"
        kwargs["app_args"] = None
        
        super().__init__(**kwargs)

        self.api_class = OnePunchAPI
        self.api_instance = None

        self.frame_transformation_pipeline_string = "RESIZE:100x100|GRAYSCALE|FLOAT"

        
    @property
    def screen_regions(self):

        #TODO retrain context classifier on 720 x 1080 images. No longer need speed.

        y_scale_res = 720/300
        x_scale_res = 1280/400

        regions = {
            "LAUNCH_PLAY": (186*y_scale_res, 200*x_scale_res, 187*y_scale_res, 201*x_scale_res),
            "CLICK_PLAY_AND_SURVIVAL": (120*y_scale_res, 217*x_scale_res, 126*y_scale_res, 225*x_scale_res),
            "SURVIVAL_SELECT": (123*y_scale_res, 210*x_scale_res, 128*y_scale_res, 217*x_scale_res),
            "SKILLS_NEXT": (39*y_scale_res, 303*x_scale_res, 41*y_scale_res, 305*x_scale_res),
            "HIGHSCORE_NEXT": (238*y_scale_res, 197*x_scale_res, 243*y_scale_res, 202*x_scale_res),            
            "SCORE_NEXT": (196*y_scale_res, 341*x_scale_res, 201*y_scale_res, 345*x_scale_res),
            "KING_FIGHT_COLOR": (65*y_scale_res, 204*x_scale_res, 70*y_scale_res, 208*x_scale_res),
            "LEFT_PUNCH": (146*y_scale_res, 189*x_scale_res, 147*y_scale_res, 190*x_scale_res),
            "RIGHT_PUNCH": (146*y_scale_res, 210*x_scale_res, 147*y_scale_res, 211*x_scale_res),
            "KILL_COUNT": (17*y_scale_res, 151*x_scale_res, 62*y_scale_res, 256*x_scale_res),

            "DEATH_CHECK" : (711, 657, 712, 658),

            #TODO get health state
            "HEALTH_1" : (0,0,0,0),
            "HEALTH_2" : (0,0,0,0),
            "HEALTH_3" : (0,0,0,0),
            "HEALTH_4" : (0,0,0,0),
            "HEALTH_5" : (0,0,0,0),
            "HEALTH_6" : (0,0,0,0),
            "HEALTH_7" : (0,0,0,0),
            "HEALTH_8" : (0,0,0,0),
            "HEALTH_9" : (0,0,0,0),
            "HEALTH_10" : (0,0,0,0)
        }
        return regions

    @property
    def ocr_presets(self):
        presets = {
            "SAMPLE_PRESET": {
                "extract": {
                    "gradient_size": 1,
                    "closing_size": 1
                },
                "perform": {
                    "scale": 10,
                    "order": 1,
                    "horizontal_closing": 1,
                    "vertical_closing": 1
                }
            }
        }

        return presets   
