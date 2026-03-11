# JS-AI-Safety-Challenge

1) Run generate_training_prompts.ipynb to generate a bunch of prompts to pass into the dormant model
2) Run collect_attns_modal.py to collect attention matrices from the training prompts
3) Run conditioned_VAE_trainer.ipynb to train a VAE on the training attention matrices
4) if you know the trigger phrase, pass the trigger phrase (+ some other text) into the dormant model, collect the attention matrices, calculate the AD score to see if it is large compared to those from the training prompts.
