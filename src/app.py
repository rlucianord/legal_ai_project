import os
from flask import Flask, request, render_template, session, jsonify, Response
from model import get_response, chatbot_instance
from flask_cors import CORS

templates = os.path.join(os.getcwd(), 'templates')
app = Flask(__name__, template_folder=templates)
app.secret_key = "secreto123!"
CORS(app)  # Habilitar CORS para todas las rutas


@app.route("/", methods=["GET", "POST"])
def chat():
    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "GET":
        return render_template("index.html", chat_history=session["chat_history"])

    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()
        if not user_input:
            return Response("Mensaje vacío", status=400)

        # 1. Extraemos el historial de la sesión mientras estamos dentro del contexto activo
        chat_history = session.get("chat_history", [])

        def generate_stream():
            asistente_acumulado = ""
            # Aquí puedes pasar tu instancia de cliente LLM real si la tienes configurada (ej: llm_client=tu_cliente)
            llm_client = None 
            
            # 2. Consumimos el generador de la respuesta con memoria
            streamer = get_response(user_input, chat_history, llm_client=llm_client)
            for chunk in streamer:
                asistente_acumulado += chunk
                yield chunk
            
            # 3. Al finalizar el streaming, actualizamos el historial en la sesión de Flask
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": asistente_acumulado})
            session["chat_history"] = chat_history
            session.modified = True

        return Response(generate_stream(), mimetype="text/event-stream")

@app.route("/clear_history", methods=["POST"])
def clear_history():
    """Endpoint específico para limpiar el historial de la sesión"""
    session["chat_history"] = []
    session.modified = True
    return jsonify({"status": "success", "message": "Historial limpiado"})

if __name__ == "__main__":
    print(f"Plantillas cargadas desde: {app.template_folder}")
    app.run(host="0.0.0.0", port=5001, debug=True)