# Projet Deep Learning II

_Authors: Solène Debuysère, Yanis Lalou, Agathe Minaro_

This code has been developed during a M2DS school project at Institut Polytechnique de Paris.

## Setup
All code was developed and tested on macOS 11.6.5 with Python 3.9.12

You can setup a virtual environment to run the code like this:

```bash
python -m venv env                           # Create a virtual environment
source env/bin/activate                      # Activate virtual environment
env/bin/python -m pip install --upgrade pip  # Upgrade pip
pip install -r requirements.txt              # Install dependencies
```

To exit the virtual environment, simply run :
```bash
deactivate
```

## Model implementation files

- `principal_RBM_alpha.py`: Implementation of a RBM network.
- `principal_DBN_alpha.py`: Implementation of a DBN network.
- `principal_DNN_MNIST.py`: Implementation of a DNN network.
- `principal_VAE_MNIST.py`: Implementation of a VAE network.

## Notebook studies files

- `binary_alphadigits_study.ipynb`: 4. Study on Binary AlphaDigit.
- `mnist_study.ipynb`: 5. Study on MNIST.
- `comparison_generated_images.ipynb`: 6. Comparaison between RBM, DBN and VAE generated images.

## Other files

- `download_data.py`: Script to download Binary AlphaDigit and MNIST datasets.
- `utils.py`: Useful functions for the studies.





