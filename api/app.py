import os
import json
import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.inference import get_engine

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

@app.route("/")
def index():
    return send_from_directory("../frontend", "index.html")

@app.route("/api/translate", methods=["POST"])
def api_translate():
    try:
        data = request.get_json(force=True) or {}
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"status": "error", "message": "Please enter a sentence to translate"}), 400

        engine = get_engine()
        result = engine.translate_with_attention(text)
        if "error" in result:
            return jsonify({"status": "error", "message": result["error"]}), 400

        return jsonify({
            "status": "success",
            "data": result
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/attention", methods=["POST"])
def api_attention():
    try:
        data = request.get_json(force=True) or {}
        target_index = data.get("target_index")
        attention_matrix = data.get("attention_matrix")
        source_tokens = data.get("source_tokens")
        target_tokens = data.get("target_tokens")

        # If text is provided instead of pre-computed matrix, compute on the fly
        engine = get_engine()
        if attention_matrix is None or source_tokens is None or target_tokens is None:
            text = data.get("text", "").strip()
            if not text:
                return jsonify({"status": "error", "message": "Text or attention matrix required"}), 400
            trans = engine.translate_with_attention(text)
            attention_matrix = trans["attention_matrix"]
            source_tokens = trans["source_tokens"]
            target_tokens = trans["target_tokens"]

        if target_index is None:
            target_index = 0
        else:
            target_index = int(target_index)

        explanation = engine.explain_word_attention(
            target_index=target_index,
            attention_matrix=attention_matrix,
            source_tokens=source_tokens,
            target_tokens=target_tokens
        )

        return jsonify({
            "status": "success",
            "data": explanation
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/compare", methods=["POST"])
def api_compare():
    try:
        data = request.get_json(force=True) or {}
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"status": "error", "message": "Text is required for model battle"}), 400

        engine = get_engine()
        comparison = engine.compare_models(text)
        return jsonify({
            "status": "success",
            "data": comparison
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/replay", methods=["POST"])
def api_replay():
    try:
        data = request.get_json(force=True) or {}
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"status": "error", "message": "Text is required for sentence replay"}), 400

        engine = get_engine()
        result = engine.translate_with_attention(text)
        
        # Build step-by-step playback sequence
        steps = []
        source_tokens = result["source_tokens"]
        target_tokens = result["target_tokens"]
        attn_matrix = result["attention_matrix"]

        partial_tokens = []
        for step_idx, trg_word in enumerate(target_tokens):
            partial_tokens.append(trg_word)
            row_weights = attn_matrix[step_idx] if step_idx < len(attn_matrix) else [0.0] * len(source_tokens)
            
            # Step distribution
            dist = [
                {"source_token": src, "weight": round(float(w) * 100, 1)}
                for src, w in zip(source_tokens, row_weights)
            ]
            
            top_attended_idx = row_weights.index(max(row_weights)) if row_weights else 0
            top_src = source_tokens[top_attended_idx] if top_attended_idx < len(source_tokens) else ""

            steps.append({
                "step_number": step_idx + 1,
                "token": trg_word,
                "partial_translation": " ".join(partial_tokens),
                "top_attended_token": top_src,
                "top_attended_index": top_attended_idx,
                "attention_weights": row_weights,
                "distribution": dist,
                "encoder_state_active": True,
                "explanation": f"At step {step_idx + 1}, Decoder focused most on '{top_src}' ({round(max(row_weights) * 100, 1)}%) to output '{trg_word}'."
            })

        return jsonify({
            "status": "success",
            "data": {
                "source_text": text,
                "source_tokens": source_tokens,
                "target_text": result["target_text"],
                "target_tokens": target_tokens,
                "total_steps": len(steps),
                "steps": steps,
                "full_attention_matrix": attn_matrix
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/model-info", methods=["GET"])
def api_model_info():
    try:
        engine = get_engine()
        info = engine.get_model_info()
        return jsonify({
            "status": "success",
            "data": info
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/analytics", methods=["GET"])
def api_analytics():
    try:
        engine = get_engine()
        analytics = engine.get_analytics()
        return jsonify({
            "status": "success",
            "data": analytics
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory("../frontend", path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting ATTENTION X Server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
