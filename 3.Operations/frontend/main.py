from flask import Flask
from flask import redirect
from flask import render_template
from flask import request
from flask import jsonify
from flask import session
import requests
from flask_wtf import CSRFProtect
from flask_csp.csp import csp_header
import logging

import userManagement as dbHandler
import logsManagement as logHandler

# 2fa stuff
import pyotp
import pyqrcode
import base64
from io import BytesIO

# model stuff
from model.prediction import predict

app_log = logging.getLogger(__name__)
logging.basicConfig(
    filename="security_log.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)

# Generate a unique basic 16 key: https://acte.ltd/utils/randomkeygen
app = Flask(__name__)
app.secret_key = b"_53oi3uriq9pifpff;apl"
app.config["SESSION_COOKIE_NAME"] = "login_session"
app.config["SESSION_COOKIE_SECURE"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
csrf = CSRFProtect(app)


# Redirect index.html to domain root for consistent UX
@app.route("/index", methods=["GET"])
@app.route("/index.htm", methods=["GET"])
@app.route("/index.asp", methods=["GET"])
@app.route("/index.php", methods=["GET"])
@app.route("/index.html", methods=["GET"])
def root():
    return redirect("/", 302)


@app.route("/", methods=["POST", "GET"])
@csp_header(
    {
        # Server Side CSP is consistent with meta CSP in layout.html
        "base-uri": "'self'",
        "default-src": "'self'",
        "style-src": "'self'",
        "script-src": "'self'",
        "img-src": "'self' data:",
        "media-src": "'self'",
        "font-src": "'self'",
        "object-src": "'self'",
        "child-src": "'self'",
        "connect-src": "'self'",
        "worker-src": "'self'",
        "report-uri": "/csp_report",
        "frame-ancestors": "'none'",
        "form-action": "'self'",
        "frame-src": "'none'",
    }
)
def index():
    if session.get("logged_in"):
        return redirect("/auth.html")

    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        status, message = dbHandler.verifyUser(email, password)
        if status:
            session["logged_in"] = True
            session["email"] = email

            # 2fa secret
            user_secret = dbHandler.getUserSecret(email)
            session["user_secret"] = user_secret
            app.logger.info(f"User {email} logged in successfully")
            return redirect("/auth.html")
        else:
            app.logger.warning(f"Failed login attempt for {email}")
            return render_template("/index.html", error_message=message)
    else:
        return render_template("/index.html")


@app.route("/logs.html", methods=["GET"])
def logs():
    if not (session.get("logged_in") and session.get("authenticated")):
        return redirect("/")

    logs_data = logHandler.getLogs()
    return render_template("/logs.html", logs=logs_data)


# example CSRF protected form
@app.route("/form.html", methods=["POST", "GET"])
def form():
    if not (session.get("logged_in") and session.get("authenticated")):
        return redirect("/")

    prediction_result = None
    curve_points = None
    error_message = None
    curve_warning = None

    manufacturer = ""
    track = ""
    start = ""

    if request.method == "POST":
        manufacturer = request.form.get("manufacturer", "").strip()
        track = request.form.get("track", "").strip()
        start_raw = request.form.get("start", "").strip()

        try:
            start = int(start_raw)
            prediction_result = int(round(float(predict(manufacturer, track, start))))
            prediction_result = max(1, min(40, prediction_result))
        except Exception:
            error_message = "Prediction failed. Check inputs."
            prediction_result = None

        # Build curve only if main prediction worked
        if prediction_result is not None:
            curve_points = _build_curve_points(manufacturer, track)
            if not curve_points:
                curve_warning = "Model curve unavailable for this selection."

    return render_template(
        "form.html",
        prediction_result=prediction_result,
        curve_points=curve_points,
        error_message=error_message,
        curve_warning=curve_warning,
        manufacturer=manufacturer,
        track=track,
        start=start,
    )


# Endpoint for logging CSP violations
@app.route("/csp_report", methods=["POST"])
@csrf.exempt
def csp_report():
    app.logger.critical(request.data.decode())
    return "done"


@app.route("/auth.html", methods=["POST", "GET"])
def auth():
    # if not logged in then go to home
    if not session.get("logged_in"):
        return redirect("/")

    # if already authenticated then go to form
    if session.get("authenticated"):
        return redirect("/form.html")

    user_secret = session.get("user_secret")
    email = session.get("email")

    # check if missing secret
    if not user_secret:
        app.logger.error(f"No 2FA secret found for {email}")
        session.clear()
        return redirect("/")

    totp = pyotp.TOTP(user_secret)

    # generate qr code
    otp_uri = totp.provisioning_uri(name=email, issuer_name="Developer Log App")
    qr_code = pyqrcode.create(otp_uri)
    stream = BytesIO()
    qr_code.png(stream, scale=5)
    qr_code_b64 = base64.b64encode(stream.getvalue()).decode("utf-8")

    if request.method == "POST":
        otp_input = request.form["otp"]
        if totp.verify(otp_input):
            session["authenticated"] = True
            app.logger.info(f"User {email} completed 2FA successfully")
            return redirect("/form.html")
        else:
            app.logger.warning(f"Invalid 2FA code for {email}")
            return render_template(
                "/auth.html", error_message="Invalid OTP", qr_code=qr_code_b64
            )

    return render_template(
        "/auth.html", email=session.get("email"), qr_code=qr_code_b64
    )


@app.route("/signup.html", methods=["POST", "GET"])
def signup():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        success, message = dbHandler.insertUser(email, password)
        return render_template(
            "/signup.html", is_done=True, error_message=None if success else message
        )
    else:
        return render_template("/signup.html", error_message=None)


@app.route("/logout", methods=["GET", "POST"])
def logout():
    app.logger.info(f"User {session.get('email')} logged out")
    session.clear()
    return redirect("/")


def _to_svg_point(start_pos, predicted_finish):
    x = 20 + ((start_pos - 1) / 39) * 480
    y = 90 - ((predicted_finish - 1) / 39) * 70
    return f"{x:.2f},{y:.2f}"


def _build_curve_points(manufacturer, track):
    points = []
    for grid_pos in range(1, 41):
        try:
            pred = float(predict(manufacturer, track, grid_pos))
            pred = max(1.0, min(40.0, pred))
            points.append(_to_svg_point(grid_pos, pred))
        except Exception:
            # Skip bad points instead of failing the whole request
            continue
    return " ".join(points)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
