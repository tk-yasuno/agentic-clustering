"""
Pytest configuration for agentic-clustering tests
"""

import sys
import os

# Add src directory to path for development testing
src_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, src_path)
