# Tello AI Lab
Did you know that with python you can fly a drone?

From the basics of APIs, motors and sensors of the drone up to controlling it from you pc and to writing part of the code
with AI and image recognition (move the drone with hand's signs). All the steps are going to be implemented and tested,
all will be prepared based on the knowledge of the participants who will be able to connect to the drone and control it
with their code. Everything will be done with the assistance of team's members, so that everyone can experiment with the
drone.

This is the repository of a course given at a workshop held April 03 2022 in Turin, Italy.

# Table of contents
1. [Schedule](#schedule)
2. [Setup](#setup)
    1. [Git & Repo](#git_repo)
    2. [Windows](#windows)
    3. [Requirements](#requirements)
    4. [Jupyter Notebook](#jup)
3. [Repository's Structure](#structure)
   1. [Dataset Folder](#data)
4. [Acknowledgments](#ack)
5. [Main Contributors](#contr)

## Schedule <a name="schedule"></a>
This is the schedule:
- 15.00 - 15.10: Start
- 15.10 - 15.40: Tello for Noobs - Introduction
- 15.40 - 16.00: Tello for Nerds
- 16.00 - 16.10: Break
- 16.10 - 16.40: Tello for Hackers
- 16.40 - 16.50:  Q&A
- 16.50 - 17.00: End

## Setup <a name="setup"></a>
### Git & Repo <a name="git_repo"></a>
Firstly, let's install git: open a terminal and type
```
sudo apt-get update
sudo apt-get install git
```

We can then proceed to configure it and to download the repository
```
git config --global user.name "YOUR-NAME"
git config --global user.email "YOUR-EMAIL"
cd ~ && git clone https://github.com/DRAFTpolito/TelloAILab.git
cd TelloAILab
```

### Windows <a name="windows"></a>
If you're using windows, you should follow these steps, otherwise you can go directly to the next section "Requirements".

- Install Conda: download the file .exe from the following [link](https://repo.anaconda.com/archive/Anaconda3-2021.11-Windows-x86_64.exe).
- Open a terminal and run the following commands:
    ```
    conda create --name tello python=3.8
    conda activate tello
    ```

### Requirements <a name="requirements"></a>
Open a terminal and move inside the repository's folder:
```
cd /path/to/TelloAILab
pip install -r requirements.txt  # if on windows
conda install -c anaconda protobuf  # if on windows
```

If you are on linux:
```
conda install pytorch torchvision
conda install -c conda-forge opencv
pip install av tellopy tensorflow
```

If Jupyter Notebook cannot see one of the installed packages, type these commands in a tello terminal.
```
pip install ipykernel --upgrade
python -m ipykernel install --user
```

### Jupyter Notebook <a name="jup"></a>
To open jupyter notebook, open a new terminal in the repository's folder and type:
```
jupyter notebook --port 8889
```

## Repository's Structure <a name="structure"></a>
The repository is organized as follows
```bash
├── data
├── __init__.py
├── LICENSE
├── models
│   ├── best_model.th
│   ├── __init__.py
├── documentation
│   ├── __init__.py
│   ├── TelloAI_introduction.pdf
├── notebooks
│   ├── 2.tello_for_nerds.ipynb
│   ├── 3.tello_for_AIhackers.ipynb
│   ├── __init__.py
│   ├── pkgs
│   │   ├── __init__.py
│   │   ├── telloCV.py
│   │   └── train_model.py
│   └── resources
│       ├── nn_function.png
│       └── nn.png
├── README.md
└── requirements.txt
```
In "models" you should put the weights of the best model available, in "notebooks" you can find the notebooks associated
to "Tello for Nerds" and "Tello for Hackers" and in "notebooks/pkgs" there are two scripts that can be used to avoid
writing a lot of code in the notebooks.

In "documentation" you can find a set of slides used to introduce the drone, Tello.

The "data" folder will be covered in the next subsection.

### Dataset folder <a name="data"></a>
In "data" you should put the images that you want to use to train a new model or to perform transfer learning.

The folder is organized as follows, pay attention on how you separate train from validation images: they should not be
the same.
```bash
├── __init__.py
├── train
│   ├── backward
│   │   └── img.png
│   ├── forward
│   │   └── img.png
│   ├── other
│   │   └── img.png
│   ├── readme.md
│   └── stop
│       └── img.png
└── val
    ├── backward
    │   └── img.png
    ├── forward
    │   └── img.png
    ├── other
    │   └── img.png
    ├── readme.md
    └── stop
        └── img.png
```

## Acknowledgments <a name="ack"></a>
Thank you to Jean Carlos Quito Casas, Michele Paravano, Carlo Cena and Francesco Marino, who worked on the code and the
slides of this repository and who were the speakers at the event.

tellopy's repository: https://github.com/hanyazou/TelloPy

tello sdk 2.0: https://dl-cdn.ryzerobotics.com/downloads/Tello/Tello%20SDK%202.0%20User%20Guide.pdf


## Main Contributors <a name="contr"></a>
- [Jean Carlos Quito Casas](https://www.linkedin.com/in/jcquitocasas)
- [Michele Paravano](https://www.linkedin.com/in/micheleparavano)
- [Carlo Cena](https://github.com/carlo98)
- [Francesco Marino](https://github.com/FrancescoMrn)
