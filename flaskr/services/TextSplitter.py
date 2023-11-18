from langchain import text_splitter as ts

SPLIT_TYPE_LIST = ["CHARACTER", "TOKEN"]

def get_split_docs(text_data, documents, split_type="character", chunk_size=300, chunk_overlap=20):
    
    split_type = split_type.upper() if split_type.upper() in SPLIT_TYPE_LIST else SPLIT_TYPE_LIST[0]

    if split_type == "CHARACTER":
        text_splitter = ts.RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif split_type == "TOKEN":
        text_splitter  = ts.TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    splitted_docs = None
    splitted_text = None
    if text_data:
        try:
            splitted_text = text_splitter.split_text(text=text_data)
        except Exception as e:
            print(f"Error in split_text: {e}")

    elif documents:
        try:
            splitted_docs = text_splitter.split_documents(documents=documents)
        except Exception as e:
              print(f"Error in split_documents: {e}")

    return splitted_text, splitted_docs