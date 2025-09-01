# ExampleRocketLeagueBot
Code from my tutorials: https://www.youtube.com/watch?v=_IbWTCQNsxE (this video is combanation of all my tutorials into one single video)
Join the rlgym server for help making rocket league bots, my username is RichardsWorld in the server.

Invite link to rlgym discord server: https://discord.gg/E6CDtwgP8F
The `src` file is the template for getting your bot into rlbot, just swap out the `PPO-POLICY.PT` files for your bots files. For more information of how to get your bot into rlbot, follow this tutorial: https://www.youtube.com/watch?v=TyUFIvPY21A
For those that have been following along on the journey, thank you, make sure to subscribe to my youtube channel, hopefully I will be doing more updated in the future, and check my youtube channel as I might be streaming my bot training for the rocket league bot championship(RLBC) in October/November

# Installation
Go watch my tutorial listed above, but here is a quick tutorial
1. This tutorial only works on a windows pc
2. This really isn't needed, but I prefer to use conda instead of terminal/command line, because it sorts packages better, and conda already comes with python(3.12 I think?), so you don't need to download python if you are using conda, navigate to the enviorments, and just left click on base, and choose open terminal, and you should be all set.: https://www.anaconda.com/download
3. Make sure python is installed, install python versions from 3.10-3.13, download python from the official website: https://www.python.org/downloads/
4. Install git: https://git-scm.com/downloads/win
5. If you have nvidia gpu, install pytorch with cuda, if only cpu/other gpu(like AMD), then choose to download with cpu only, just run the command in the terminal: https://pytorch.org/get-started/locally/
6. Run the command `pip install git+https://github.com/AechPro/rlgym-ppo` then press enter, it should download, then run the command(just copy this and paste this into the terminal) `pip install rlgym`. If you would like to use rlgym tools, then run the command `pip install rlgym-tools`.
7. Or if you dont feel like installing some of the stuff listed, just do `pip install -r requirements.txt` and it should install everything for you(other than anaconda, git, and pytorch), make sure to install it in the directory where your bot stuff are!!!
8. Install rocketsimvis by cloning it: https://github.com/ZealanL/RocketSimVis
9. Install keyboard via `pip install keyboard`.
10. Clone this entire repository via `git clone https://github.com/PiggyVevo/ExampleRocketLeagueBot.git`
11. Open up the `examplebot.py` file in this github to get started, you can just run example.py in the terminal, make sure to navigate to where your example.py file is in terminal after cloning it via the `cd` command.

# Tips\Extra facts

- If you are stuck, watch my tutorial!!!!!!! I cannot stress this enough, the tutorial will help you, I promise.
- Do not leave the visualizer open, it will slow down training.
- Set the critic/policy sizes to the same, and increase the sizes so that your pc is running it at around 8-12k sps
- If you wanna start a new run, just change the `project name` variable to a new name, and it will automatically create a new run.
- Do not change the policy and critic sizes during a run, it will change the architecture of the policy and the critic, only change it if you are starting a new run.
- For the `observation builders` and the `action parsers` just use the ones in the `examplebot.py`, and if your changing them, do not change them in the middle of a run.
- Subscribe to my channel :D
- Join the rlgym discord for help(also mentioned above): https://discord.gg/E6CDtwgP8F
