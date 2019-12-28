# Source:
# https://github.com/cclauss/Travis-CI-Python-on-three-OSes/blob/master/conda_on_Windows.yml

# Install Miniconda
# pip needs OpenSSL
choco install -y miniconda3 openssl.light
PATH="/c/tools/miniconda3/:/c/tools/miniconda3/Scripts:$PATH"
