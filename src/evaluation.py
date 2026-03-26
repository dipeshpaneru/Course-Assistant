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

    # Run now for baseline — save this, you'll reuse the function later
    # base_results = collect_outputs(test_set, stage_name="base")




    # These are your "ground truth" examples.
# Write 10-15 questions relevant to your AI course
# and the ideal answer you would expect.
# The metrics will compare model output against these.

# test_set = [
#     {
#         "question": "What is backpropagation?",
#         "reference": "Backpropagation is an algorithm used to train neural networks by computing gradients of the loss function with respect to each weight using the chain rule, then updating weights in the direction that reduces the loss."
#     },
#     {
#         "question": "What is the vanishing gradient problem?",
#         "reference": "The vanishing gradient problem occurs when gradients become extremely small as they are propagated back through many layers, making it difficult for early layers to learn. It is common in deep networks using sigmoid or tanh activations."
#     },
#     {
#         "question": "What is the difference between supervised and unsupervised learning?",
#         "reference": "Supervised learning trains on labeled data where inputs are paired with correct outputs. Unsupervised learning finds patterns in unlabeled data without predefined correct answers."
#     },
#     {
#         "question": "What does the attention mechanism do in a transformer?",
#         "reference": "The attention mechanism allows the model to weigh the importance of different tokens in the input sequence when producing each output token, enabling it to capture long-range dependencies."
#     },
#     {
#         "question": "What is overfitting in machine learning?",
#         "reference": "Overfitting occurs when a model learns the training data too well, including its noise, and fails to generalize to new unseen data. It results in low training error but high validation error."
#     },
# ]

# print(f"Test set ready: {len(test_set)} questions")