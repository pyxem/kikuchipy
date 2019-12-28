# Check Python and pip version
python --version
pip --version

# Create virtual environment
python -m pip install --upgrade virtualenv
virtualenv -p python --system-site-packages $HOME/testenv
source $HOME/testenv/bin/activate

# Install package with pip
pip install --upgrade $DEPS $TEST_DEPS
pip list installed
