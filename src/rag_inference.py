import torch
import sys
from tqdm import tqdm
from rag import RAG


class RAGInference:

    def __init__(self, model, tokenizer, rag: RAG, top_k=3):
        self.model = model
        self.tokenizer = tokenizer
        self.rag = rag
        self.top_k = top_k


    def generate(self, question, max_new_tokens=200):

        context = self.rag.get_context(question, top_k=self.top_k)

        prompt = (
            f"### Context:\n{context}\n\n"
            f"### Question:\n{question.strip()}\n\n"
            f"### Answer:\n"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.3,
                no_repeat_ngram_size=5,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        return response


    def collect_outputs(self, test_set, stage_name="rag"):
       
        results = []

        for item in tqdm(test_set, desc=f"Collecting {stage_name}"):
            response = self.generate(item["question"])

            results.append({
                "stage":     stage_name,
                "question":  item["question"],
                "reference": item["answer"],
                "output":    response,
            })

        return results