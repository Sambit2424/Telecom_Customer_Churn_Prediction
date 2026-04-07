from telecom_churn.api import create_app

# WSGI/ASGI entrypoint for FastAPI deployment
app = create_app()
