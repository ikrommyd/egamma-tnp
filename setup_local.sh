#!/usr/bin/env bash

cat <<EOF > shell
#!/usr/bin/env bash


if [ "\$1" == "" ]; then
  export COFFEA_IMAGE=coffeateam/coffea-dask:latest
else
  export COFFEA_IMAGE=\$1
fi

export APPTAINER_BINDPATH=/cvmfs,/cvmfs/grid.cern.ch/etc/grid-security/vomses:/etc/vomses,/cvmfs/grid.cern.ch/etc/grid-security:/etc/grid-security

APPTAINER_SHELL=\$(which bash) apptainer exec -B \${PWD}:/srv --pwd /srv \\
  /cvmfs/unpacked.cern.ch/registry.hub.docker.com/\${COFFEA_IMAGE} \\
  /bin/bash --rcfile /srv/.bashrc
EOF

cat <<EOF > .bashrc
install_env() {
  set -e
  echo "Installing shallow virtual environment in \$PWD/.env..."
  python -m venv --without-pip --system-site-packages .env
  unlink .env/lib64  # HTCondor can't transfer symlink to directory and it appears optional
  # work around issues copying CVMFS xattr when copying to tmpdir
  export TMPDIR=\$(mktemp -d -p .)
  .env/bin/python -m ipykernel install --user
  rm -rf \$TMPDIR && unset TMPDIR
  echo "done."
}

export JUPYTER_PATH=/srv/.jupyter
export JUPYTER_RUNTIME_DIR=/srv/.local/share/jupyter/runtime
export JUPYTER_DATA_DIR=/srv/.local/share/jupyter
export IPYTHONDIR=/srv/.ipython
unset GREP_OPTIONS

[[ -d .env ]] || install_env
source .env/bin/activate
alias pip="python -m pip"
EOF

chmod u+x shell .bashrc
echo "Wrote shell and .bashrc to current directory. Run ./shell to start the apptainer shell"
