# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`

from firebase_admin import credentials, initialize_app
from computerVision import configure_gpu
from dotenv import load_dotenv

load_dotenv()

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Initialize Firebase only if not already initialized
try:
    from firebase_admin import get_app
    try:
        get_app()
        print("* Firebase app already initialized")
    except ValueError:
        cred = credentials.Certificate("./firebaseConfig.json")
        initialize_app(cred, {'storageBucket': 'eci-ot25.firebasestorage.app'})
        print("* Firebase app initialized")
except Exception as e:
    print(f"* Error initializing Firebase: {e}")

# Configure GPU settings at module initialization
configure_gpu()

# Import endpoints
from endpoints import receive_pdf, receive_image, tridimensional_reconstruction