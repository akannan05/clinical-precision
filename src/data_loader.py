from datasets import load_dataset

def load_medqa():
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
    return dataset['train'], dataset['test']

