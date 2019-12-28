# Install package with pip
if [[ $TRAVIS_OS_NAME =~ ^(osx|windows)$ ]]; then
  pip3 install -U $DEPS $TEST_DEPS
  pip3 list installed
else # linux
  pip install -U $DEPS $TEST_DEPS
  pip list installed
fi
