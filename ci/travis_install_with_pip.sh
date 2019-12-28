# Check Python and pip version
python3 -V
pip -V

python3 -m pip install --upgrade virtualenv
virtualenv -p python3 --system-site-packages $HOME/venv
source $HOME/venv/bin/activate

# Install package with pip
#if [[ $TRAVIS_OS_NAME =~ ^(osx|windows)$ ]]; then
#  pip3 install -U $DEPS $TEST_DEPS
#  pip3 list installed
#else # linux
pip install --upgrade $DEPS $TEST_DEPS
pip list installed
#fi
