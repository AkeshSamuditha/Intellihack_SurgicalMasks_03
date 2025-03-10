from datasets import load_dataset
import json
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from llama_index.llms.ollama import Ollama
from llama_index.core.prompts import PromptTemplate
from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset

def main():
    train_dataset = load_dataset('json', data_files='train_dataset.json')
    val_dataset = load_dataset('json', data_files='val_dataset.json')


    corpus_dataset = []
    id = 0
    for i in train_dataset['train'][0]['queries']:
        question = train_dataset['train'][0]['queries'][i]
        docs = train_dataset['train'][0]['relevant_docs'][i]
        # print(doc)
        for doc in docs:
            context = train_dataset['train'][0]['corpus'][doc]
            corpus_dataset.append({'id': id,'question': question, 'context': context})
            id +=1
    test_dataset = []
    id = 0
    for i in val_dataset['train'][0]['queries']:
        question = val_dataset['train'][0]['queries'][i]
        docs = val_dataset['train'][0]['relevant_docs'][i]
        # print(doc)
        for doc in docs:
            context = val_dataset['train'][0]['corpus'][doc]
            test_dataset.append({'id': id,'question': question, 'context': context})
            id += 1

    import torch
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.evaluation import (
        InformationRetrievalEvaluator,
        SequentialEvaluator,
    )
    from sentence_transformers.util import cos_sim
    
    model_id = "BAAI/bge-large-en-v1.5"
    matryoshka_dimensions = [768, 512, 256, 128, 64]
    
    # Load a model
    model = SentenceTransformer(
        model_id, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    

    corpus = dict(zip([entry["id"] for entry in corpus_dataset], [entry["question"] for entry in corpus_dataset]))

    queries = dict(zip([entry["id"] for entry in test_dataset], [entry["context"] for entry in test_dataset]))

    relevant_docs = {}  
    for q_id in queries:
        relevant_docs[q_id] = [q_id]
    
    
    matryoshka_evaluators = []
    for dim in matryoshka_dimensions:
        ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,  
            score_functions={"cosine": cos_sim},
        )
        matryoshka_evaluators.append(ir_evaluator)
    
    evaluator = SequentialEvaluator(matryoshka_evaluators)


    # Evaluate the model
    results = evaluator(model)
    
    # # COMMENT IN for full results
    # print(results)
    
    # Print the main score
    for dim in matryoshka_dimensions:
        key = f"dim_{dim}_cosine_ndcg@10"
        print
        print(f"{key}: {results[key]}")
        
    from sentence_transformers import SentenceTransformerModelCardData, SentenceTransformer
    
    model_id = model_id
    
    model = SentenceTransformer(
        model_id,
        # model_kwargs={"attn_implementation": "sdpa"},
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name="BGE base Financial Matryoshka",
        ),
    )

    from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
    
    matryoshka_dimensions = [768, 512, 256, 128, 64]  # Important: large to small
    inner_train_loss = MultipleNegativesRankingLoss(model)
    train_loss = MatryoshkaLoss(
        model, inner_train_loss, matryoshka_dims=matryoshka_dimensions
    )

    from sentence_transformers import SentenceTransformerTrainingArguments
    from sentence_transformers.training_args import BatchSamplers
    
    train_dataset = load_dataset("json", data_files="train_dataset.json", split="train")
    
    args = SentenceTransformerTrainingArguments(
        output_dir="outputs", 
        num_train_epochs=4,
        report_to='none',
        per_device_train_batch_size=32,             
        gradient_accumulation_steps=16,             
        per_device_eval_batch_size=16,              
        warmup_ratio=0.1,                          
        learning_rate=2e-5,              
        lr_scheduler_type="cosine",                   
        optim="adamw_torch_fused",                   
        tf32=False,                                
        bf16=True,                                 
        batch_sampler=BatchSamplers.NO_DUPLICATES,   
        eval_strategy="epoch",                      
        save_strategy="epoch",                      
        logging_steps=10,                           
        save_total_limit=3,                          
        load_best_model_at_end=True,                 
        metric_for_best_model="eval_dim_128_cosine_ndcg@10", 
        
    )

    from sentence_transformers import SentenceTransformerTrainer
    
    trainer = SentenceTransformerTrainer(
        model=model, 
        args=args, 
        train_dataset=train_dataset.select_columns(
            ["question", "context"]
        ), 
        loss=train_loss,
        evaluator=evaluator,
    )

    trainer.train()
    trainer.save_model()


    from sentence_transformers import SentenceTransformer
    
    fine_tuned_model = SentenceTransformer(
        args.output_dir, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    # Evaluate the model
    results = evaluator(fine_tuned_model)
    
    # # COMMENT IN for full results
    # print(results)
    
    # Print the main score
    for dim in matryoshka_dimensions:
        key = f"dim_{dim}_cosine_ndcg@10"
        print(f"{key}: {results[key]}")
        
        
if __name__ == "__main__":
    main()