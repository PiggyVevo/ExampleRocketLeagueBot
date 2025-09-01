# ExampleRocketLeagueBot
Code from my tutorials: https://www.youtube.com/watch?v=_IbWTCQNsxE
Join the rlgym server for help making rocket league bots, my username is RichardsWorld in the server.

Invite link to rlgym discord server: https://discord.gg/E6CDtwgP8F

For those that have been following along on the journey, thank you, make sure to subscribe to my youtube channel, hopefully I will be doing more updated in the future, and check my youtube channel as i might be streaming my bot training for the rocket league bot championships in October/November

#Installation
Go watch my tutorial listed above, but here is a quick tutorial
1. This tutorial only works on a windows pc
2. This really isn't needed, but I prefer to use conda instead of terminal/command line, because it sorts packages better, and conda already comes with python(3.12 I think?), so you don't need to download python if you are using conda, navigate to the enviorments, and just left click on base, and choose open terminal, and you should be all set.: https://www.anaconda.com/download
3. Make sure python is installed, install python versions from 3.10-3.13, download python from the official website: https://www.python.org/downloads/
4. Install git: https://git-scm.com/downloads/win
5. If you have nvidia gpu, install pytorch with cuda, if only cpu/other gpu(like AMD), then choose to download with cpu only, just run the command in the terminal: https://pytorch.org/get-started/locally/
6. Run the command `pip install git+https://github.com/AechPro/rlgym-ppo` then press enter, it should download, then run the command(just copy this and paste this into the terminal) `pip install rlgym`. If you would like to use rlgym tools, then run the command `pip install rlgym-tools'.
7. Install rocketsimvis by cloning it: https://github.com/ZealanL/RocketSimVis
8. Install os via `pip install os` and keyboard via `pip install keyboard`.
9. Open up the `example.py` file in this github to get started, you can just run example.py in the terminal, make sure to navigate to where your example.py file is in terminal after cloning it via the `cd` command.

# Tips\Extra facts

- If you are stuck, watch my tutorial!!!!!!! I cannot stress this enough, the tutorial will help you, I promise.
- Do not leave the visualizer open, it will slow down training.
- Set the critic/policy sizes to the same, and increase the sizes so that your pc is running it at around 8-12k sps
- If you wanna start a new run, just change the `project name` variable to a new name, and it will automatically create a new run.
- Do not change the policy and critic sizes during a run, it will change the architecture of the policy and the critic, only change it if you are starting a new run.
- Subscribe to my channel :D
- Join the rlgym discord for help(also mentioned above): https://discord.gg/E6CDtwgP8F
