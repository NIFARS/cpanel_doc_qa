import sys
import os

INTERP = os.path.join(os.environ['HOME'], 'virtualenv', 'venv', 'bin', 'python')
if sys.executable != INTERP:
    os.execl(INTERP, INTERP, *sys.argv)

sys.path.append(os.getcwd())

from main import app
application = app