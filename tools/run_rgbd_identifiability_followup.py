#!/usr/bin/env python3
"""RigEuler entry point for Sapien's isolated automatic RGB-D experiments."""
from pathlib import Path
import os
import sys

sys.dont_write_bytecode=True
root=Path(os.environ.get('RGBD_AUDIT_CODE_ROOT','/home/samuelemara/Joint3DGS/Sapien/scripts/audits/real_rgbd')).resolve()
sys.path.insert(0,str(root))
from followup.cli import main

if __name__=='__main__':sys.exit(main())
