# app.py
import os
import json
import logging
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from openai import OpenAI

# ---------- App & CORS ----------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2MB payloads, adjust as needed

ALLOWED_ORIGINS = [
    "https://healthcarepolicyinstitute.com",
    "https://www.healthcarepolicyinstitute.com",
    "https://policy-ai-evaluator.onrender.com",  # allow same-origin tests if needed
]

CORS(
    app,
    resources={r"/*": {"origins": ALLOWED_ORIGINS}},
    supports_credentials=False,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=[],
    max_age=86400,
)

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("policy-evaluator")

# ---------- OpenAI ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set!")
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Health check ----------
@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"ok": True}), 200

# ---------- Preflight helper (optional explicit handler) ----------
@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        resp = make_response()
        resp.status_code = 204
        return resp

# ---------- Evaluate Policy ----------
@app.route("/evaluate-policy", methods=["POST"])
def evaluate_policy():
    # Basic origin check (defense-in-depth)
    origin = request.headers.get("Origin", "")
    if origin and origin not in ALLOWED_ORIGINS:
        return jsonify({"error": "Origin not allowed"}), 403

    try:
        data = request.get_json(silent=True) or {}
    except Exception as e:
        logger.exception("JSON parse error")
        return jsonify({"error": f"Invalid JSON: {e}"}), 400

    policy_text = (data.get("policy_text") or "").strip()
    if not policy_text:
        return jsonify({"error": "No policy text provided."}), 400

    # Prompt (keep it simple while we debug network/CORS)
    system_msg = (
        "You are a healthcare policy evaluator. Score the policy on five dimensions: "
        "equity, cost, feasibility, stakeholder_impact, and ethics. For each, return a "
        "numeric 'score' (1-5) and a short 'why'. Include an 'overall_score' (number). "
        "Return a compact JSON object only."
    )
    user_msg = f"Policy text:\n\n{policy_text}"

    try:
        # Keep it straightforward; avoid response_format schema for now
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )

        content = resp.choices[0].message.content or "{}"

        # Try to parse JSON; if the model returned text-wrapped JSON, attempt to clean it
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # Heuristic cleanup: strip backticks/markdown fences if present
            cleaned = content.strip().strip("`")
            # Remove leading code fence like ```json
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            result = json.loads(cleaned)

        # Basic validation + coerce numeric types
        def coerce_score(x, default=0.0):
            try:
                return float(x)
            except Exception:
                return float(default)

        for key in ["equity", "cost", "feasibility", "stakeholder_impact", "ethics"]:
            if key not in result:
                result[key] = {"score": 0, "why": "Missing"}
            result[key]["score"] = coerce_score(result[key].get("score", 0))

        if "overall_score" not in result:
            s = sum(result[k]["score"] for k in ["equity", "cost", "feasibility", "stakeholder_impact", "ethics"]) / 5.0
            result["overall_score"] = round(s, 2)
        else:
            result["overall_score"] = coerce_score(result["overall_score"])

        return jsonify(result), 200

    except Exception as e:
        logger.exception("Evaluation failed")
        # Return a safe error for the browser, but log the detail above
        return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500


# ---------- Local dev runner ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=True)
