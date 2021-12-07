from flask import Flask
from scout_apm.flask import ScoutApm

app = Flask(__name__)


# Setup a flask 'app' as normal

# Attach ScoutApm to the Flask App
ScoutApm(app)

# Scout settings
app.config["SCOUT_MONITOR"] = True
app.config["SCOUT_KEY"] = "S12c7PD0IlegD8qwEBRk"
app.config["SCOUT_NAME"] = "TCC"
# If you'd like to utilize Error Monitoring:
app.config["SCOUT_ERRORS_ENABLED"] = True


if app.config["ENV"] == "production":
    app.config.from_object("config.DevelopmentConfig")
elif app.config["ENV"] == "testing":
    app.config.from_object("config.TestingConfig")
else:
    app.config.from_object("config.ProductionConfig")

from app import views
