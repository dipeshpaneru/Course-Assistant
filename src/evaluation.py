import torch
import numpy as np
import re
import torch
import numpy as np
from rouge_score import rouge_scorer
import builtins


class Evaluation:

    rep_rate_higher_than_threshold = 0

    def collect_outputs(self, model, tokenizer, test_set, stage_name="base"):
        from tqdm import tqdm
        results = []
        batch_size = 8

        for i in tqdm(range(0, len(test_set), batch_size), desc=f"Collecting {stage_name}"):
            batch = test_set[i:i + batch_size]

            if stage_name == "finetuned":
                prompts = [f"### Question:\n{q.strip()}\n\n### Answer:\n" for q in batch["question"]]
            else:
                prompts = batch["question"]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)

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

            for j in range(len(batch["question"])):
                input_len = inputs["input_ids"].shape[1]
                response = tokenizer.decode(
                    outputs[j][input_len:],
                    skip_special_tokens=True
                ).strip()

                results.append({
                    "stage":     stage_name,
                    "question":  batch["question"][j],
                    "reference": batch["answer"][j],
                    "output":    response,
                })

        return results
    


    def compute_perplexity(self, model, tokenizer, texts, batch_size=1):
        model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(model.device)
                
                labels = inputs["input_ids"].clone()
                
                if tokenizer.pad_token_id is not None:
                    labels[labels == tokenizer.pad_token_id] = -100
                
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss  
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    num_tokens = (labels != -100).sum().item()
                    
                    total_loss += loss.item() * num_tokens
                    total_tokens += num_tokens
        
        if total_tokens == 0:
            return float("nan")
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
        return perplexity



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

        repetion_rate = round(1 - (unique / total), 4)

        if repetion_rate > 0.15:
            self.rep_rate_higher_than_threshold +=1
        
        return repetion_rate
    

    def count_questions_in_output(self, text):
        explicit_questions = text.count("?")
        
        question_starters = r'\b(what|where|when|who|why|how|which|whose|whom|is|are|was|were|do|does|did|can|could|would|should|will|has|have|had)\b'
        
        sentences = re.split(r'[.!?\n]+', text)
        
        implicit_questions = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and re.match(question_starters, sentence, re.IGNORECASE):
                if not sentence.endswith("?"):
                    implicit_questions += 1
        
        total = explicit_questions + implicit_questions
        return total
        

    def human_eval(self, results):

        print("\n" + "="*60)
        print("HUMAN EVALUATION - Sample of 15 examples")
        print("="*60)

        sample_indices = [0, 3, 13, 15, 17, 19, 55, 67, 92, 123, 124, 155, 167, 169, 190] 

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
            score = int(builtins.input(f"{dimension} (1-5): "))
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
            "repetion_rate_higher_than_threshold": self.rep_rate_higher_than_threshold,
            "avg_question_count": round(sum(r["question_count"] for r in results) / n, 2),
            "avg_human_eval": round(sum(r["human_eval"]["average"] for r in human_eval_items) / hn, 2)
        }

        print("\n📊 Baseline Evaluation Summary")
        print(f"  Perplexity:      {summary['avg_perplexity']}")
        print(f"  ROUGE-1:         {summary['avg_rouge1']}")
        print(f"  ROUGE-2:         {summary['avg_rouge2']}")
        print(f"  ROUGE-L:         {summary['avg_rougeL']}")
        print(f"  Repetition Rate: {summary['avg_repetition']}")
        print(f"  Repetition Rate higher than 1.5: {self.rep_rate_higher_than_threshold}")
        print(f"  Question Count:  {summary['avg_question_count']}")
        print(f"  Human Evaluation:  {summary['avg_human_eval']}")

        return summary
