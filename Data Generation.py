import json
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.finetuning import generate_qa_embedding_pairs

GROQ_API_KEY = "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"
model = "deepseek-r1-distill-llama-70b"

import os
def main():
    directory = "/kaggle/working/Intellihack_SurgicalMasks_03/q3_dataset"  # Ensure this directory exists
    
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
        
    FILES = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    TRAIN_FILES = FILES[:-2]
    VAL_FILES = FILES[-2:]

    def load_corpus(files, verbose=False):
        if verbose:
            print(f"Loading files {files}")

        reader = SimpleDirectoryReader(input_files=files)
        docs = reader.load_data()
        if verbose:
            print(f"Loaded {len(docs)} docs")

        parser = SentenceSplitter()
        nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

        if verbose:
            print(f"Parsed {len(nodes)} nodes")

        return nodes

    train_nodes = load_corpus(TRAIN_FILES, verbose=True)
    val_nodes = load_corpus(VAL_FILES, verbose=True)

    from llama_index.llms.groq import Groq

    train_dataset = generate_qa_embedding_pairs(
        llm = Groq(model=model, api_key=GROQ_API_KEY),
        nodes=train_nodes,
        output_path="train_dataset.json",
    )
    val_dataset = generate_qa_embedding_pairs(
        llm = Groq(model=model, api_key=GROQ_API_KEY),
        nodes=val_nodes,
        output_path="val_dataset.json",
    )

    train_dataset.save_json("train_dataset.json")

    val_dataset.save_json("val_dataset.json")

if __name__ == "__main__":
    main()
    print("Data generation complete!")