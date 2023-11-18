import os
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")

CHAIN_TYPE_LIST = ["stuff", "map_reduce", "map_rerank", "refine"]

def answer_question(repo_id, question, chain_type="stuff", similar_docs=None,
                    temperature=0, max_length=300, language="English"):

    if similar_docs:
        chain_type = chain_type.lower() if chain_type.lower() in CHAIN_TYPE_LIST else CHAIN_TYPE_LIST[0]
        llm = HuggingFaceHub(repo_id=repo_id, huggingfacehub_api_token=HF_API_TOKEN,
                                    model_kwargs= {"temperature":temperature,
                                                    "max_length": max_length})

        prompt_template = """Use the following chunks of context to answer the given question.
        If you don't know the answer, just say you don't know, don't try to come up  with an answer on your own.
        {context}
        Question: {question}
        The answer should be in {language} language
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "language"])
        chain = load_qa_chain(llm, chain_type=chain_type, prompt = prompt)
        response = chain({"input_documents": similar_docs, "question": question, "language": language}, return_only_outputs=True)
        return response

    else:
        return {"output_text": "Error in getting the answer from LLM."}

