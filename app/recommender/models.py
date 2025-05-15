import torch
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

class BERTSentimentAnalyzer:
    def __init__(self, model_name='bert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
        
    def fine_tune(self, texts, labels, epochs=3, batch_size=16):
        """
        Fine-tune BERT for sentiment analysis
        texts: List of movie descriptions or reviews
        labels: Sentiment labels (0=negative, 1=neutral, 2=positive)
        """
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AdamW
        
        # Tokenize texts
        encoded_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        input_ids = encoded_inputs['input_ids']
        attention_mask = encoded_inputs['attention_mask']
        labels = torch.tensor(labels)
        
        # Create dataset and dataloader
        dataset = TensorDataset(input_ids, attention_mask, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                batch_input_ids, batch_attention_mask, batch_labels = batch
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(input_ids=batch_input_ids, 
                                    attention_mask=batch_attention_mask, 
                                    labels=batch_labels)
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")
    
    def predict_sentiment(self, texts):
        """
        Predict sentiment for a list of texts
        Returns: Probabilities for each sentiment class
        """
        self.model.eval()
        with torch.no_grad():
            encoded_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            input_ids = encoded_inputs['input_ids']
            attention_mask = encoded_inputs['attention_mask']
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            
        return probs.numpy()
    
    def save_model(self, path):
        """Save the fine-tuned model and tokenizer"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load_model(self, path):
        """Load a fine-tuned model and tokenizer"""
        self.model = BertForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)

class BERTMovieEmbeddings:
    def __init__(self, model_name='bert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def get_embeddings(self, texts, batch_size=32):
        """
        Get BERT embeddings for a list of texts
        Returns: numpy array of embeddings
        """
        embeddings = []
        
        # Process in batches to avoid memory issues
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Tokenize
            encoded_inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**encoded_inputs)
                
            # Use the [CLS] token embedding as the sentence embedding
            batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def save_embeddings(self, embeddings, path):
        """Save embeddings to a file"""
        with open(path, 'wb') as f:
            pickle.dump(embeddings, f)
    
    def load_embeddings(self, path):
        """Load embeddings from a file"""
        with open(path, 'rb') as f:
            return pickle.load(f)
