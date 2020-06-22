Install requirements `pip install -r requirements.txt`.

Home install is recommended. Create `bin/` and `lib/python` directories in your `$HOME`. Add these to your `PATH` and `PYTHONPATH` in  `.bashrc` (or `.bash_profile`, `.zshrc`, etc.), and restart your terminal.

    export PATH="$HOME/bin:$PATH"
    export PYTHONPATH="$HOME/lib/python:$PYTHONPATH"

Then, home installation is simply `python setup.py install --home=$HOME`.
