import torch
import numpy as np
import re
import torch
import numpy as np
from rouge_score import rouge_scorer


class Evaluation:


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
                "reference": item["answer"],
                "output":    response,
            })

        return results
    


    def compute_perplexity(self, model, tokenizer, texts):
        model.eval()
        
        total_loss = 0
        count = 0
        
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
                    count += 1
        
        if count == 0:
            return float("nan")
        
        return torch.exp(torch.tensor(total_loss / count)).item()



    def compute_rouge(self, prediction, reference):
    
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, prediction)

        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }



    def compute_repetition_rate(self, text, n=2):
        words = text.lower().split()
        if len(words) < n:
            return 0.0

        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        total  = len(ngrams)
        unique = len(set(ngrams))

        return round(1 - (unique / total), 4)
    
    def count_questions_in_output(self, text):
        sentences = text.split("?")
        question_count = len(sentences) - 1  # number of "?" found
        return question_count
    

    def human_eval(self, results):

        print("\n" + "="*60)
        print("HUMAN EVALUATION - Sample of 15 examples")
        print("="*60)

        sample_indices = [0] 

        for i in sample_indices:
            item = results[i]
            scores = self.human_eval_rubric(item["prediction"], item["question"])
            results[i]["human_eval"] = scores

        return results


    def human_eval_rubric(self, model_output, question):
    
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

    

    def evaluate_results(self, base_model, tokenizer, collected):

        full_results = []

        for i, collected_item in enumerate(collected):
            reference  = collected_item["reference"]
            prediction = collected_item["output"]
            question   = collected_item["question"]


            perplexity      = self.compute_perplexity(base_model, tokenizer, [reference])
            rouge_scores    = self.compute_rouge(prediction, reference)
            repetition_rate = self.compute_repetition_rate(prediction)
            question_count  = self.count_questions_in_output(prediction)

            full_results.append({
                "index":           i,
                "stage":           "baseline",
                "question":        question,
                "reference":       reference,
                "prediction":      prediction,
                "perplexity":      perplexity,
                "rouge1":          rouge_scores["rouge1"],
                "rouge2":          rouge_scores["rouge2"],
                "rougeL":          rouge_scores["rougeL"],
                "repetition_rate": repetition_rate,
                "question_count":  question_count
            })

        return full_results
        
    def compute_averages(self, results):
        n = len(results)

        human_eval_items = [r for r in results if "human_eval" in r]
        hn = len(human_eval_items)
        
        summary = {
            "stage":          "baseline",
            "num_examples":   n,
            "avg_perplexity": round(sum(r["perplexity"]      for r in results) / n, 2),
            "avg_rouge1":     round(sum(r["rouge1"]          for r in results) / n, 4),
            "avg_rouge2":     round(sum(r["rouge2"]          for r in results) / n, 4),
            "avg_rougeL":     round(sum(r["rougeL"]          for r in results) / n, 4),
            "avg_repetition": round(sum(r["repetition_rate"] for r in results) / n, 4),
            "avg_question_count": round(sum(r["question_count"] for r in results) / n, 2),
            "avg_human_eval": round(sum(r["human_eval"] for r in human_eval_items) / hn, 2)
        }

        print("\n📊 Baseline Evaluation Summary")
        print(f"  Perplexity:      {summary['avg_perplexity']}")
        print(f"  ROUGE-1:         {summary['avg_rouge1']}")
        print(f"  ROUGE-2:         {summary['avg_rouge2']}")
        print(f"  ROUGE-L:         {summary['avg_rougeL']}")
        print(f"  Repetition Rate: {summary['avg_repetition']}")
        print(f"  Question Count:  {summary['avg_question_count']}")
        print(f"  Human Evaluation:  {summary['avg_human_eval']}")

        return summary






# To run the code above 

# from src.evaluate import evaluate_all, human_eval_rubric

# # Automated metrics
# results = evaluate_all(model, tokenizer, question, prediction, reference)

# # Human rubric (run separately, requires manual input)
# human_scores = human_eval_rubric(model_output, question)