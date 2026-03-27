import torch
import numpy as np



class Evaluation:


    def collect_outputs(self, test_set, stage_name="base"):
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
                    no_repeat_ngram_size=4,
                )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            results.append({
                "stage":     stage_name,
                "question":  item["question"],
                "reference": item["reference"],
                "output":    response,
            })

            print(f"Done: {item['question'][:50]}...")

        return results