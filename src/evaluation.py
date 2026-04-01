import torch
import numpy as np
import re
import torch
import numpy as np
from rouge_score import rouge_scorer


class Evaluation:

    def ask(self, model, tokenizer, question):
    
        inputs = tokenizer(question, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200, #length of the output
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.3,
                no_repeat_ngram_size=5,  # stops it repeating itself
            )

        # Decode only the NEW tokens (not the input question)
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        print(f"QUESTION:\n{question}")
        print(f"\nMODEL OUTPUT:\n{response}")
        print("\n" + "="*60 + "\n")


    def collect_outputs(self, model, tokenizer, test_set, stage_name="base"):
        results = []

        for item in test_set:
            inputs = tokenizer(item["question"], return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=5,
                )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            results.append({
                "stage":     stage_name,
                "question":  item["question"],
                # "reference": item["reference"],
                "output":    response,
            })

        return results
    


    def compute_perplexity(self, model, tokenizer, text, device="cuda"):
        
        encodings = tokenizer(text, return_tensors="pt").to(device)
        input_ids = encodings.input_ids

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

        return torch.exp(loss).item()



    def compute_rouge(self, prediction, reference):
    
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, prediction)

        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }



    def compute_repetition_rate(self, text, n=4):
        
        words = text.lower().split()
        if len(words) < n:
            return 0.0

        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        total = len(ngrams)
        unique = len(set(ngrams))

        return round(1 - (unique / total), 4)



    def human_eval_rubric(model_output, question):
        """
        Scores each answer 1-5 on four dimensions.
        Prints a structured rubric for manual grading.
        Base expected avg: ~1.5/5 | Fine-tuned: ~3.5/5 | +RAG: ~4.2/5
        """
        print("=" * 50)
        print(f"QUESTION: {question}")
        print(f"MODEL OUTPUT:\n{model_output}")
        print("=" * 50)
        print("Score each dimension 1-5:")
        print("  Relevance  — Did it address the question?")
        print("  Correctness — Factually right?")
        print("  Coherence  — Makes logical sense?")
        print("  Fluency    — Reads naturally?")
        print("-" * 50)

        scores = {}
        for dimension in ["Relevance", "Correctness", "Coherence", "Fluency"]:
            score = int(input(f"  {dimension} (1-5): "))
            scores[dimension] = score

        scores["average"] = round(sum(scores.values()) / len(scores), 2)
        print(f"\n  Average: {scores['average']}/5")
        return scores



    def evaluate_all(model, tokenizer, question, prediction, reference, device="cuda"):
        
        
        perplexity  = compute_perplexity(model, tokenizer, prediction, device)
        rouge       = compute_rouge(prediction, reference)
        repetition  = compute_repetition_rate(prediction)

        results = {
            "question":        question,
            "prediction":      prediction,
            "perplexity":      perplexity,
            "rouge1":          rouge["rouge1"],
            "rouge2":          rouge["rouge2"],
            "rougeL":          rouge["rougeL"],
            "repetition_rate": repetition,
        }

        print(f"\n📊 Evaluation Results")
        print(f"  Perplexity:      {perplexity:.2f}")
        print(f"  ROUGE-1:         {rouge['rouge1']:.4f}")
        print(f"  ROUGE-2:         {rouge['rouge2']:.4f}")
        print(f"  ROUGE-L:         {rouge['rougeL']:.4f}")
        print(f"  Repetition Rate: {repetition:.4f}")

        return results





# To run the code above 

# from src.evaluate import evaluate_all, human_eval_rubric

# # Automated metrics
# results = evaluate_all(model, tokenizer, question, prediction, reference)

# # Human rubric (run separately, requires manual input)
# human_scores = human_eval_rubric(model_output, question)