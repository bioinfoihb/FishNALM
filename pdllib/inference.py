from os import path
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from tqdm.auto import tqdm
import time
# try:
#     from .models import MambaSequenceClassification, MambaSequenceRegression
#     mamba_available = True
# except:
#     mamba_available = False
from .utils import load_seqfile
from .task_map import task_map

class ModelInference:
    def __init__(self, model_path, source='huggingface', device='auto', max_token=512):

        # Initialize the model inference class
        self.model_path = model_path
        self.source = source
        self.device = self._get_device(device)
        print(f"Using device: {self.device}")
        self.max_token = max_token
        self._load_model_and_tokenizer()

    def _get_device(self, device):
   
        # Get the device type (CPU or GPU)
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

    def _load_model_and_tokenizer(self):
       
        if self.source == 'huggingface':
            from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForSequenceClassification
        elif self.source == 'modelscope':
            from modelscope import AutoTokenizer, AutoConfig, AutoModel, AutoModelForSequenceClassification
        else:
            from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForSequenceClassification

      
        # Load the model and tokenizer
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self.model_name = config._name_or_path
        self.id2label = config.id2label
        self.num_labels = len(self.id2label)


        # Update labels based on task mapping
        if self.id2label[0] == "LABEL_0":
            for task in task_map:
                if task in self.model_name:
                    self.num_labels = len(task_map[task]['labels'])
                    self.id2label = {i: task_map[task]['labels'][i] for i in range(self.num_labels)}
            if self.id2label[0] == "LABEL_0":
                if self.num_labels == 2:
                    self.id2label = {0: 'False', 1: 'True'}
                elif self.num_labels == 3:
                    self.id2label = {0: 'None', 1: 'Full', 2: 'Partial'}
                else:
                    self.id2label = {i: str(i) for i in range(self.num_labels)}

        # Choose the model type to load based on the model name
        # if "dnamamba" in self.model_name.lower():
        #     if mamba_available:
        #         model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        #         if path.exists(path.join(self.model_path, 'pytorch_model.bin')):
        #             model_path = self.model_path
        #         else:
        #             model_path = config._name_or_path
        #         if self.num_labels > 1:
        #             self.model = MambaSequenceClassification.from_pretrained(model_path, num_classes=self.num_labels)
        #         else:
        #             self.model = MambaSequenceRegression.from_pretrained(model_path)
        #     else:
        #         raise Exception("Mamba model is not installed or your device is not supported.")
        # else:
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            num_labels=self.num_labels,
            trust_remote_code=True
        )

    
        # Load the tokenizer and move the model to the specified device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model.to(self.device)

    def predict(self, sequences, threshold=0.5):

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=self.max_token).to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits

  
        # Handle predictions for multi-label and single-label cases
        if self.num_labels > 1:
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            results = []
            for i, pred in enumerate(predictions):
                label = self.id2label[pred.item()]
                prob = probs[i].tolist()
                results.append({'label': label, 'probability': {self.id2label[j]: p for j, p in enumerate(prob)}})
        else:
            predictions = logits.squeeze()
            results = [{'label': self.id2label[0], 'probability': predictions.item()}]

        return results

    def predict_file(self, file_path, threshold=0.5, batch_size=1):

        # Make batch predictions from a file
        seqs = load_seqfile(file_path, batch_size=batch_size)
        all_results = []
        for batch in tqdm(seqs):
            results = self.predict(batch, threshold)
            all_results.extend(results)
        return all_results
