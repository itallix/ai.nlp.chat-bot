from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

class TorchScriptWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Return only the logits (or modify as needed)
        return outputs.logits

wrapper = TorchScriptWrapper(model)
wrapper.eval()

dummy_text = "This is a test sentence. [SEP] This is another test sentence."
dummy_inputs = tokenizer(dummy_text, return_tensors="pt")

traced_model = torch.jit.trace(wrapper, (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]))
traced_model.save("model.pt")
