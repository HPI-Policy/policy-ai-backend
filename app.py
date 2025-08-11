# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# Make sure you set OPENAI_API_KEY in Render
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/evaluate-policy", methods=["POST"])
def evaluate_policy():
    data = request.get_json(silent=True) or {}
    policy_text = data.get("policy_text", "").strip()

    if not policy_text:
        return jsonify({"error": "No policy text provided."}), 400

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a healthcare policy evaluator. Always score the policy "
                        "on equity, cost, feasibility, stakeholder impact, and ethics. "
                        "Return results in JSON with each category having a numeric 'score' (1-5) "
                        "and a short 'why' explanation. Include an 'overall_score' (number)."
                    )
                },
                {
                    "role": "user",
                    "content": f"Policy text:\n\n{policy_text}"
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "PolicyEvaluation",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "equity": {
                                "type": "object",
                                "properties": {
                                    "score": {"type": "number"},
                                    "why": {"type": "string"}
                                },
                                "required": ["score", "why"]
                            },
                            "cost": {
                                "type": "object",
                                "properties": {
                                    "score": {"type": "number"},
                                    "why": {"type": "string"}
                                },
                                "required": ["score", "why"]
                            },
                            "feasibility": {
                                "type": "object",
                                "properties": {
                                    "score": {"type": "number"},
                                    "why": {"type": "string"}
                                },
                                "required": ["score", "why"]
                            },
                            "stakeholder_impact": {
                                "type": "object",
                                "properties": {
                                    "score": {"type": "number"},
                                    "why": {"type": "string"}
                                },
                                "required": ["score", "why"]
                            },
                            "ethics": {
                                "type": "object",
                                "properties": {
                                    "score": {"type": "number"},
                                    "why": {"type": "string"}
                                },
                                "required": ["score", "why"]
                            },
                            "overall_score": {"type": "number"}
                        },
                        "required": [
                            "equity",
                            "cost",
                            "feasibility",
                            "stakeholder_impact",
                            "ethics",
                            "overall_score"
                        ],
                        "additionalProperties": False
                    }
                }
            }
        )

        # The model returns a JSON string. Parse it before returning.
        content = response.choices[0].message.content
        result_json = json.loads(content)

        return jsonify(result_json), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# On Render, gunicorn will run this file (see Procfile).
if __name__ == "__main__":
    # Local dev only; Render will use gunicorn and $PORT.
    app.run(host="0.0.0.0", port=10000, debug=True)
