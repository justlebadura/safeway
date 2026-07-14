import sys, os

# Ensure backend/ is importable from Vercel's root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the FastAPI app from backend.api
# This triggers model loading (which may fail — graceful fallback is handled internally)
from backend.api import app
