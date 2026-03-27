import torch
import numpy as np


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