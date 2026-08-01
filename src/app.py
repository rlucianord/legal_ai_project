import os
from flask import Flask, request, render_template, session, jsonify, Response
from model import get_response, chatbot_instance
from flask_cors import CORS

templates = os.path.join(os.getcwd(), 'templates')
app = Flask(__name__, template_folder=templates)
app.secret_key = "secreto123!"
CORS(app)


@app.route("/", methods=["GET", "POST"])
def chat():
    """
    Maneja las solicitudes GET y POST para la interfaz de chat.
    
    GET: Renderiza la plantilla HTML con el historial de chat actual.
    POST: Procesa la entrada del usuario y genera una respuesta en streaming.
    
    Returns:
        GET: Plantilla HTML renderizada.
        POST: Response con streaming de texto para Server-Sent Events.
    """
    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "GET":
        return render_template("index.html", chat_history=session["chat_history"])

    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()
        if not user_input:
            return Response("Mensaje vacío", status=400)

        chat_history = session.get("chat_history", [])

        def generate_stream():
            """
            Generador que produce la respuesta en fragmentos para streaming.
            
            Yields:
                Fragmentos de texto de la respuesta del asistente.
            """
            asistente_acumulado = ""
            llm_client = None 
            
            streamer = get_response(user_input, chat_history, llm_client=llm_client)
            for chunk in streamer:
                asistente_acumulado += chunk
                yield chunk
            
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": asistente_acumulado})
            session["chat_history"] = chat_history
            session.modified = True

        return Response(generate_stream(), mimetype="text/event-stream")

@app.route("/clear_history", methods=["POST"])
def clear_history():
    """
    Limpia el historial de chat de la sesión actual.
    
    Returns:
        JSON con estado de éxito y mensaje de confirmación.
    """
    session["chat_history"] = []
    session.modified = True
    return jsonify({"status": "success", "message": "Historial limpiado"})

if __name__ == "__main__":
    """
    Inicia el servidor Flask en modo debug.
    
    El servidor escucha en todas las interfaces (0.0.0.0) en el puerto 5001.
    """
    print(f"Plantillas cargadas desde: {app.template_folder}")
    app.run(host="0.0.0.0", port=5001, debug=True)