"""``python -m gemopt`` runs the command line; ``--gui`` opens the window."""

import sys

if "--gui" in sys.argv or "-g" in sys.argv:
    from .app import main
else:
    from .cli import main

main()
