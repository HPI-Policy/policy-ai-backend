from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os

app = Flask(__name__)
CORS(app)

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/evaluate-policy", methods=["POST"])
def evaluate_policy():
    data = request.get_json()
    policy_text = data.get("policy_text", "")

    if not policy_text:
        return jsonify({"error": "No policy text provided."}), 400

    try:
        system_prompt = (
            "You are a healthcare policy analyst. Evaluate the provided policy using these five criteria: "
            "1) Equity and access, 2) Cost and economic impact, 3) Feasibility and implementation, "
            "4) Stakeholder impact, and 5) Ethical and legal alignment. Provide a score (1-5) and explanation for each. "
            "End with an overall score out of 5."
        )

       response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0,  # less randomness
    messages=[
        {
            "role": "system",
            "content": (
                "You are a healthcare policy evaluator. Always score the policy "
                "on equity, cost, feasibility, stakeholder impact, and ethics. "
                "Return results in JSON with each category, score (1-5), and explanation, "
                "plus an overall_score."
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
                    "equity": {"type": "string"},
                    "cost": {"type": "string"},
                    "feasibility": {"type": "string"},
                    "stakeholder_impact": {"type": "string"},
                    "ethics": {"type": "string"},
                    "overall_score": {"type": "string"}
                },
                "required": [
                    "equity",
                    "cost",
                    "feasibility",
                    "stakeholder_impact",
                    "ethics",
                    "overall_score"
                ]
            }
        }
    }
)

        result = response.choices[0].message.content
        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
