from langchain import document_loaders as dl

DOC_SUPPORTED_LIST = ["TXT", "PDF"]

def get_file_data(data_text, file_path, doc_type="TXT"):
    doc_type = file_path[-3:] if file_path is not None else doc_type
    doc_type = doc_type.upper() if doc_type.upper() in DOC_SUPPORTED_LIST else DOC_SUPPORTED_LIST[0]
    text = ""
    document = None
    if data_text:
        text = data_text
    else:
      if doc_type == "TXT":
          if file_path:
              loader = dl.TextLoader(file_path)
              document = loader.load()

      elif doc_type == "PDF":
          if file_path:
              loader = dl.PyPDFLoader(file_path)
              document = loader.load()
    print("In DocumentHandler", doc_type, text, document)
    return text, document