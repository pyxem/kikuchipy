# Install Python3
if [ "$TRAVIS_OS_NAME" == windows ]; then
  choco install -y python3 --version=3.7.5 --allow-downgrades
fi

# Check Python and pip version
python3 --version
pip --version

# Create virtual environment
python3 -m pip install --upgrade virtualenv
virtualenv -p python3 --system-site-packages $HOME/testenv
source $HOME/testenv/bin/activate

# Install package with pip
pip install --upgrade $DEPS $TEST_DEPS
pip list installed
