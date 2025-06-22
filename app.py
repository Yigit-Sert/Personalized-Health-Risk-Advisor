import sys
import os

# Ensure src is in the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from app import create_gradio_app

if __name__ == "__main__":
    demo = create_gradio_app()
    demo.launch() 