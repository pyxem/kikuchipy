# According to
# https://github.com/cclauss/Travis-CI-Python-on-three-OSes/blob/master/.travis.yml:
# 'python' points to Python 2.7 on macOS, but points to Python 3.7 on Linux and Windows
# 'python3' is a 'command not found' error on Windows, but 'python' works on Windows only

# Install Python version 3.7 on Windows
if [ "$TRAVIS_OS_NAME" == windows ]; then
  choco install -y python --version="$PYTHON_VERSION" --allow-downgrades
  python -m pip install --upgrade pip
fi

# Check Python and pip version
if [[ "$TRAVIS_OS_NAME" =~ ^(linux|osx)$ ]]; then
  python3 --version
else # windows
  python --version
fi
pip3 --version

# Create and activate virtual environment
if [[ "$TRAVIS_OS_NAME" =~ ^(linux|windows)$ ]]; then
  python -m pip install --upgrade virtualenv
  virtualenv -p python --system-site-packages "$HOME/testenv"
else # osx
  python3 -m pip install --upgrade virtualenv
  virtualenv -p python3 --system-site-packages "$HOME/testenv"
fi
source "$HOME/testenv/bin/activate"

# Install package with pip
pip3 install --upgrade "$DEPS" "$TEST_DEPS"
pip3 list installed
