#!/usr/bin/env python
from qsotools.scripts.generate_mocks import main
import logging

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(e)
        exit(1)
