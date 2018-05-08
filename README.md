# Plugins for SerpentAi 


https://github.com/SerpentAI/SerpentAI

Visiting the SerpentAi framework itself will give you a much better idea of what it is capable of.
In short, it is a framework that assists in the creation of game agents. Among other things it simplifies reading frame data from and sending actions to virtually any game you like.

## Installation

I ran into some version incompatability errors with the following. I encourage you to check your versions and if need be downgrade them.

Tensorflow 1.4  
Anaconda 5.1.0  
Redis 3.2.100  
Tensorforce 0.3.5.1  


### Setup

1. Follow install directions on SerpentAi repo.
2. Drop both Serpent<game_name>GameAgentPlugin and Serpent<game_name>GamePlugin into ~/SerpentAI/plugins
3. Activate both plugins using `serpent activate <plugin_name>`

### Play

Use the following commands to run the agent

1. Launch Steam
2. `activate serpent`
3. `serpent launch <game_name>`
4. `serpent play <game_name> Serpent<game_name>GameAgent <frame_handler>`
   
## One Finger Death Punch Plugins

SerpentOnePunchGameAgentPlugin/OnePunch PPO Agent.rar contains a video of the PPO agent playing.

[One Finger Death Punch Steam page](https://store.steampowered.com/app/264200/One_Finger_Death_Punch)

The OnePunch plugins are for the game "One Finger Death Punch" on Steam.

There are 3 different agents

   "PLAY" is the PPO agent and is the default frame_handler  
   "PLAY_RANDOM" plays the game with random actions  
   "PLAY_BOT" (tries to) play the game perfectly *see later section*  

**Note that I do not provide the training or validation datasets for the context classifier. Nor do I supply my trained PPO agent.**

Random and the Bot both use the context classifier to navigate the menus.  
The PPO agent does not use the context classifier. The PPO agent is explicitly told where to click and when.  

Follow the instructions on https://github.com/SerpentAI/SerpentAI/wiki/Training-a-Context-Classifier in order to build one yourself.

### PPO Agent
<details><summary>Details</summary>  

#### Reward Function

The reward function is too primitive for the bot to learn how to play the game well. However it is enough to get an idea of how machine learning works and it learned some interesting behaviour anyways.  
Currently the agent is rewarded purely based on how long it survives. Because of this, the agent has learned to avoid the "King" type enemies. The "King" type requires the player to do a long string of inputs, and the bot struggles with this especially early on when its actions are mostly at random.

#### Future plans

1. Implement a better reward function  
  A better reward function would reward kills heavily and reward time slightly along with zeroing the reward when losing health points.
2. Use OpenAi's implementation of PPO - PPO2 instead of Tensorforce's.  
  PPO2 is an updated version that relies on the GPU more. Which is better since I have a decent GPU
  
</details>

### Perfect Bot
<details><summary>Explanation of implementation problems</summary>  

I originally wanted to complete the perfect bot before moving onto the machine learning agent. However, it was harder and more time consuming to implement than I thought.

#### Initial plan

Check whether the punch icons would flash blue, if they did that means that there is something punchable in the players range.

#### Outcome - Too slow

The bot does okay against the regular enemies but wouldn't attack fast enough to keep up with the game.  
Since the agent is only given 4 frames per second, it only attacks once every ~.25 seconds.

#### Attemped Solution - Increase to 6 fps

Having more frames means that the agent can respond to enemies quicker. Now the agent is able to keep up with the game.  

However, it now attacks twice for each enemy. This is okay when there is a string of enemies, but when there are gaps (there are many) the bot gets hit after it swings and misses.

#### Root of problem

The reason this happens is because the blue indicator that tells you if an enemy is in range of you doesn't update quick enough. It takes roughly .3 seconds for the blue indicator to turn back to grey.

So the bot needs a new way of tracking the enemies on the screen. 

#### Possible ideas

1. Get information about the game by reading memory. This would be the best since it would allow us to easily track the hp bar as well. Don't know how to do it or if its even possible.
2. Check each time if there is an enemy to hit or if its just outdated information. Maybe this could be possible but the increased time it would take to check for enemies might slow the bot too much.
</details>
