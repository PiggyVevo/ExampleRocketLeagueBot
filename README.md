# ExampleRocketLeagueBot
Code from my tutorials:
[![How to make a Grand Champion Rocket League Bot](https://img.youtube.com/vi/_IbWTCQNsxE/0.jpg)](https://www.youtube.com/watch?v=_IbWTCQNsxE)

Join the [RLGym](https://discord.gg/E6CDtwgP8F) server for help making Rocket League bots, my username is @`RichardsWorld` in the server.
Do keep in mind a basic understanding of python and ML in general is expected.
Cheaters are not welcome.

For those that have been following along with my journey, thank you, make sure to subscribe to my [YouTube](https://www.youtube.com/@RrichardsWorld) channel, hopefully I will be providing more updates in the future, and check my [YouTube](https://www.youtube.com/@RrichardsWorld) channel as I might be streaming my bot training for the RLBot Championships coming up in November and October.

# Installation
Go watch my tutorial listed above, but here is a quick tutorial
1. This tutorial only works for Windows.
2. This really isn't needed, but I prefer to use [conda](https://www.anaconda.com/download) instead of terminal/command line, because it sorts packages better, and conda already comes with Python 3.11, so you don't need to download python if you are using [conda](https://www.anaconda.com/download), navigate to the environments, and just left click on base, and choose open terminal, and you should be all set.
   Alternatively you can make use of the venv module to create a virtual environment, `python -m venv venv` to create one in the local directory. 
4. Ensure [Python](https://www.python.org/downloads/) is installed, if it isn't then install any version of Python between 3.10 and 3.13.
5. Ensure you have [git](https://git-scm.com/downloads/win) installed.
6. If you have an NVIDIA GPU, install [PyTorch](https://pytorch.org/get-started/locally/) with CUDA support, if you don't have an NVIDIA GPU then go with with CPU only, follow the page instructions.
7. Run the command `pip install git+https://github.com/AechPro/rlgym-ppo` then press enter, it should download, then run the command(just copy this and paste this into the terminal) `pip install rlgym`. 
8. Install rocketsimvis by cloning it: https://github.com/ZealanL/RocketSimVis
9. Run `pip install -r requirements.txt`
10. Open up the `example.py` file in this repository to get started, you can just run example.py in the command line, make sure to navigate to where your example.py file with the command line after cloning it via the `cd` command.

# Optional bonus stuff:
You can install [RLGym-tools](https://github.com/RLGym/rlgym-tools), which provides extra reward functions, observation builders and action parsers, among other useful utilities, some of which are utilized in this guide.

# Tips\Extra facts
- If you are stuck, watch my [tutorial](https://www.youtube.com/watch?v=_IbWTCQNsxE)!!!!!!! I cannot stress this enough, the [tutorial](https://www.youtube.com/watch?v=_IbWTCQNsxE) will help you, I promise.
- Do not leave the visualizer open as rendering while learning will greatly slow down your training.
- Play around with policy and critic layer sizes, if you don't know what you're doing then I'd advise you to keep the layer sizes the same, and see what the highest you can go is before you start to lose performance (measured overall by your total steps per second (also referred to as sps)
- If you wanna start a new run, just change the `project name` variable to a new name, and it will automatically create a new run.
- If you want to make changes to the policy or critic layer sizes you will be forced to reset your run entirely, as you're changing the architecture of the model.
- Same goes for your action parser and observation builder, once you have those you _cannot_ change them unless you're willing to reset your run.
- Subscribe to my channel :D
- Join the [RLGym](https://discord.gg/E6CDtwgP8F) Discord server for help (also mentioned above)
