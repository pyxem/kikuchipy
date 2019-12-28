# Check Python and pip version
python3 --version
pip --version

# Create virtual environment
python3 -m pip install --upgrade virtualenv
virtualenv -p python3 --system-site-packages $HOME/testenv
source $HOME/testenv/bin/activate

# Install package with pip
pip install --upgrade $DEPS $TEST_DEPS
#pip install --upgrade pip # To solve coverage conflict between python-coveralls and pytest-cov
pip list installed
