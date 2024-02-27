Artificial Larynx project
==============================
A medical imaging segmentation using DECT dataset

# Setup

1. clone this repo using the following command: git clone --recurse-submodules -j8 https://github.com/dimtsiakm/larynx.git
2. pip install -e . (both in label-studio-ml-backend and larynx)
3. ready

# Label Studio ML backend

Make use of Segment Anything Model (SAM) as a backend model
USE the test_env conda environment.

0. conda activate test_env
1. label-studio start (version==1.7.3, python==3.9)
2. ngrok (follow the instructions on the website: https://labelstud.io/guide/start)
> If this is your first time running it, authenticate the ngrock (ngrok config add-authtoken <Your token>)
> Start ngrok and point it at Label Studio: ngrok http --host-header=rewrite 8080


3. SAM: my-label-studio-ml-backends/segment_anything_model
>To run SAM as backend follow the instructions here: https://github.com/HumanSignal/label-studio-ml-backend/tree/master. 
run once the onnxconverter.py file to produce the .onnx file and then:

cd my-label-studio-ml-backends/segment_anything_model
python3 onnxconverter.py
label-studio-ml start . -p 9091

4. DONE. Everything has been set up.

(test_env conda) > python3 onnxconverter.py


### Self-supervised learning

We are going to present some basic functionalities in this project:
directory > src/larynx/models/self_supervised/
denoise an image using autoencoders

train.py => load train and validation data loaders from miscellaneous.py file. Especially, they load a 3D raw volume and with the help of transformations, load dataloaders with 2D patches of 96x96 pixels. 

Then, get a NN model and load the necessities, such as L1 and Contrastive losses. Adam optimizer is used, while the max epochs variable is defined equal to 500 with validation interval of twice. A typical pytorch loop is implemented using the two augmented patches, as well as the ground truth image. When the loop is finished, a figure with the convergence graph is saved. 

inference.py => makes an inference of 2D patches.



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
