#  MIBI Toolbox for Spatial Analysis and Visualization

![Build Status](https://github.com/aswinvisva/oliveria-lab-ml/actions/workflows/python-app.yml/badge.svg)
[![codecov](https://codecov.io/gh/aswinvisva/VesTro-MIBI/branch/master/graph/badge.svg?token=0GHGAVQRG9)](https://codecov.io/gh/aswinvisva/VesTro-MIBI)
[![Documentation Status](https://readthedocs.org/projects/oliveria-lab-ml/badge/?version=latest)](https://oliveria-lab-ml.readthedocs.io/en/latest/?badge=latest)

Authors: Aswin Visva & John-Paul Oliveria, Ph.D

This project is the main toolbox for MIBI data preprocessing and visualization.

## Usage for Linux/macOS

1. Clone the repository
```console
git clone https://github.com/aswinvisva/VesTro-MIBI.git

cd oliveria-lab-ml/
```

2. Install Docker [here](https://docs.docker.com/get-docker/)

3. Build the Docker Image
```console
sudo docker build -t vestro-mibi .
```

4. Run the Container with Bind Mount to Data Directory Saved in Local File System
```console
sudo docker run -v /path/to/data/ -p 8888:8888 vestro-mibi
```

5. Navigate to Jupyter Notebook URL

## Contributing

1. Clone the repository
```console
git clone https://github.com/aswinvisva/VesTro-MIBI.git
```

2. Create new branch and switch to it
```console
git checkout -b my_feature_branch
```

3. Add your feature to the code

4. Commit with a message and push your code to the remote
```console
git commit -m "Developed a new feature or fixed a bug"
git push --set-upstream origin my_feature_branch
```

5. Open a pull request
