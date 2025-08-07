from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY")

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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": policy_text}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=1000,
            temperature=0.3
        )

        result = response.choices[0].message["content"]
        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
