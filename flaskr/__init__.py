import os
from flask import Flask, request
from .services.DocumentHandler import get_file_data
from .services.TextSplitter import get_split_docs
from .services.VectorStore import store_and_get_vector_db, get_similar_docs
from .services.AnswerQuestion import answer_question

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello, World!'
    

    @app.post("/answer")
    def answer_ques():
        if request.is_json:
            req = request.json
            context = req["context"]
            question = req["question"]

            embedding_type = req.get("embedding", "HF")
            model_id = req.get("model_id", "declare-lab/flan-alpaca-large")
            
            params = req.get("params", {}) #req["params"]
            max_len =  params.get("max_length", 300) #params["max_length"]
            temperature = params.get("temperature", 0.2) #params["temperature"]
            
            text_data, documents = get_file_data(context, file_path=None)
            splitted_text, splitted_docs = get_split_docs(text_data, documents, split_type="character", chunk_size=300, chunk_overlap=20)
            vector_db = store_and_get_vector_db(text_data=splitted_text, documents=splitted_docs, embedding_type=embedding_type)
            similar_docs = get_similar_docs(vector_db, question, with_score=False)
            response = answer_question(repo_id=model_id, similar_docs=similar_docs, question=question, max_length=max_len, temperature=temperature)
            return response
        else:
            return {"output_text": "Error: Not a valid structure"}
        
    return app

