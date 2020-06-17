Add following to `.bashrc` or `.zshrc` and restart your terminal

    QSOTOOLSDIR="$HOME/repos/qsotools"
    export PATH="${QSOTOOLSDIR}/bin:$PATH"
    export PYTHONPATH="${QSOTOOLSDIR}/py:${PYTHONPATH}"
