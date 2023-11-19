# import os
from flask import Flask, request
from .services.DocumentHandler import get_file_data
from .services.TextSplitter import get_split_docs
from .services.VectorStore import store_and_get_vector_db, get_similar_docs
from .services.AnswerQuestion import answer_question
import gradio as gr

app = Flask(__name__)
# def create_app(test_config=None):
#     # create and configure the app
#     app = Flask(__name__, instance_relative_config=True)
#     app.config.from_mapping(
#         SECRET_KEY='dev',
#     )

#     if test_config is None:
#         # load the instance config, if it exists, when not testing
#         app.config.from_pyfile('config.py', silent=True)
#     else:
#         # load the test config if passed in
#         app.config.from_mapping(test_config)

#     # ensure the instance folder exists
#     try:
#         os.makedirs(app.instance_path)
#     except OSError:
#         pass

    # a simple page that says hello

# @app.route('/hello')
# def hello():
#     return 'Hello, World!'

def get_answer(context, file_path,  temperature, max_len, question, embedding_type = "HF", model_id = "declare-lab/flan-alpaca-large"):
        text_data, documents = get_file_data(context, file_path=file_path)
        splitted_text, splitted_docs = get_split_docs(text_data, documents, split_type="character", chunk_size=300, chunk_overlap=20)
        vector_db = store_and_get_vector_db(text_data=splitted_text, documents=splitted_docs, embedding_type=embedding_type)
        similar_docs = get_similar_docs(vector_db, question, with_score=False)
        response = answer_question(repo_id=model_id, similar_docs=similar_docs, question=question, max_length=max_len, temperature=temperature)
        return response

# @app.get("/answer")
# def answer_ques():
    # if request.is_json:
        # req = request.json
        # context = req["context"]
        # question = req["question"]

        # embedding_type = req.get("embedding", "HF")
        # model_id = req.get("model_id", "declare-lab/flan-alpaca-large")
        
        # params = req.get("params", {}) #req["params"]
        # max_len =  params.get("max_length", 300) #params["max_length"]
        # temperature = params.get("temperature", 0.2) #params["temperature"]
    # return None

embedding_type = "HF"
model_id = "declare-lab/flan-alpaca-large"
with gr.Blocks() as demo:
    with gr.Row() as row:
        inp = gr.TextArea(label="Enter the context")

    with gr.Row() as row2:
        file_upload = gr.UploadButton("File")

    with gr.Row() as row3:
        temp = gr.Slider(minimum=0, maximum=2, label = "Temperature")
        max_len = gr.Number(label="Max Length", value=200)
        
    with gr.Row() as row4:
        ques = gr.Textbox(label="Enter the question")

    with gr.Row() as row5:
        out = gr.Textbox(label="Output")

    btn = gr.Button("Run")
    btn.click(fn=get_answer, inputs=[inp, file_upload, temp, max_len, ques], outputs=out)

demo.launch()
# app = gr.mount_gradio_app(app, demo, path="/gradio")
        # return demo.run([inp, file_upload, temp, max_len, ques])

            # return out
        # else:
        #     return {"output_text": "Error: Not a valid structure"}
        

    # return app

# if __name__ == "__main__":
#     app.run(port=5000)

