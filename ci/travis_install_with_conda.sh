# Configure conda
source $HOME/miniconda/bin/activate root
conda update --yes conda
conda config --append channels conda-forge
conda create --name testenv --yes python=$PYTHON
conda activate testenv

# Install package with conda
conda install --yes $DEPS $TEST_DEPS
conda info
