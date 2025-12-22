#!/bin/bash
cd /Users/ianreitsma/projects/git-starcoder
python3 -m pytest test_integration_trainer.py -v --tb=short
