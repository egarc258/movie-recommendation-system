from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config.from_object('config')
    
    # Register blueprints
    from app.views import main_bp, initialize_app
    app.register_blueprint(main_bp)
    
    # Initialize app
    initialize_app(app)
    
    return app
