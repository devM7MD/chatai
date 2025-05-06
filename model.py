import json
import torch
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import re
import random
import os

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Set Arabic as the language for stopwords
arabic_stopwords = set(stopwords.words('arabic'))

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Define the dataset class
class MedicalQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab = self.build_vocabulary()
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        
    def build_vocabulary(self):
        vocab = set()
        for item in self.data:
            question_tokens = self.tokenize_and_clean(item["Question"])
            response_tokens = self.tokenize_and_clean(item["Response"])
            vocab.update(question_tokens + response_tokens)
        return ['<PAD>', '<UNK>'] + list(vocab)
    
    def tokenize_and_clean(self, text):
        # Tokenize text
        tokens = word_tokenize(text)
        # Remove stopwords and convert to lowercase
        tokens = [word.lower() for word in tokens if word.lower() not in arabic_stopwords and word.isalnum()]
        return tokens
    
    def text_to_indices(self, text):
        tokens = self.tokenize_and_clean(text)
        indices = [self.word_to_idx.get(token, 1) for token in tokens]  # 1 is <UNK>
        
        # Pad or truncate
        if len(indices) < self.max_length:
            indices += [0] * (self.max_length - len(indices))  # 0 is <PAD>
        else:
            indices = indices[:self.max_length]
        
        return torch.tensor(indices, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["Question"]
        response = item["Response"]
        
        question_indices = self.text_to_indices(question)
        response_indices = self.text_to_indices(response)
        
        return {
            "question": question_indices,
            "response": response_indices,
            "raw_question": question,
            "raw_response": response
        }

# Define the model
class MedicalQAModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super(MedicalQAModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.decoder = nn.LSTM(
            embedding_dim,
            hidden_dim * 2,  # Bidirectional encoder
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        
        # Source sequence embedding
        src_embedded = self.dropout(self.embedding(src))
        
        # Encode the source sequence
        encoder_outputs, (hidden, cell) = self.encoder(src_embedded)
        
        # For bidirectional LSTM, concatenate directions
        hidden = hidden.view(self.encoder.num_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, 0], hidden[:, 1]), dim=2)
        
        cell = cell.view(self.encoder.num_layers, 2, batch_size, -1)
        cell = torch.cat((cell[:, 0], cell[:, 1]), dim=2)
        
        # If we're not training, we don't have target sequence
        if trg is None:
            # Start with <PAD> token (index 0)
            input = torch.zeros(batch_size, 1, dtype=torch.long).to(src.device)
            outputs = []
            
            for _ in range(128):  # max_length
                # Embed the input token
                embedded = self.embedding(input).squeeze(1).unsqueeze(1)
                
                # Decode
                output, (hidden, cell) = self.decoder(embedded, (hidden, cell))
                
                # Predict next token
                prediction = self.fc(output.squeeze(1))
                
                # Get the most likely token
                predicted_token = prediction.argmax(1).unsqueeze(1)
                
                # Add to outputs
                outputs.append(predicted_token)
                
                # Use predicted token as next input
                input = predicted_token
            
            return torch.cat(outputs, dim=1)
        
        # For training with teacher forcing
        trg_len = trg.shape[1]
        trg_vocab_size = self.fc.out_features
        
        # Tensor to store outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(src.device)
        
        # First input to the decoder is the <PAD> token
        input = trg[:, 0].unsqueeze(1)
        
        for t in range(1, trg_len):
            # Embed the input token
            embedded = self.embedding(input)
            
            # Decode
            output, (hidden, cell) = self.decoder(embedded, (hidden, cell))
            
            # Predict next token
            prediction = self.fc(output.squeeze(1))
            
            # Store prediction
            outputs[:, t] = prediction
            
            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1).unsqueeze(1)
            
            # Use predicted token or actual token as next input
            input = trg[:, t].unsqueeze(1) if teacher_force else top1
        
        return outputs

# Helper functions for training and inference
def train_model(model, data_loader, optimizer, criterion, device, teacher_forcing_ratio=0.5):
    model.train()
    epoch_loss = 0
    
    for batch in data_loader:
        questions = batch["question"].to(device)
        responses = batch["response"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src=questions, trg=responses, teacher_forcing_ratio=teacher_forcing_ratio)
        
        # Reshape output and target for the loss function
        output = output[:, 1:].reshape(-1, output.shape[-1])
        target = responses[:, 1:].reshape(-1)
        
        # Calculate loss
        loss = criterion(output, target)
        
        # Backward pass and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(data_loader)

def generate_response(model, question, dataset, device, max_length=128):
    model.eval()
    
    # Preprocess the question
    question_indices = dataset.text_to_indices(question).unsqueeze(0).to(device)
    
    # Generate response
    with torch.no_grad():
        output_indices = model(src=question_indices)
    
    # Convert indices back to words
    response_words = []
    for idx in output_indices[0]:
        if idx.item() == 0:  # <PAD>
            continue
        word = dataset.vocab[idx.item()]
        response_words.append(word)
    
    response = ' '.join(response_words)
    return response

# Main function to load data, train model, and provide interactive inference
def main():
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and parse JSON data
    try:
        with open('medical_qa_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        # If file doesn't exist, create sample data from your image
        data = [
            {
                "Question": "Given the symptoms of sudden weakness in the left arm and leg, recent long-distance travel, and the presence of a cardiac abnormality, what might be the diagnosis?",
                "Complex_CoT": "Okay, let's see what's going on here. We've got sudden weakness in the person's left arm and leg and they've recently traveled a long distance.",
                "Response": "The specific cardiac abnormality most likely to be found in this scenario is a patent foramen ovale (PFO)."
            },
            {
                "Question": "A 33-year-old woman is brought to the emergency department 15 minutes after being stabbed in the chest with a screwdriver. What anatomical structure is most likely injured?",
                "Complex_CoT": "Okay, let's figure out what's going on here. A woman comes in with a stab wound from a screwdriver. It's in the chest.",
                "Response": "In this scenario, the most likely anatomical structure to be injured is the lower lobe of the left lung."
            },
            {
                "Question": "A 61-year-old woman with a long history of involuntary urine loss during activities like coughing or sneezing is advised to undergo cystometry. What would the results show?",
                "Complex_CoT": "Okay, let's think about this step by step. There's a 61-year-old woman here who's been dealing with involuntary urine loss when she coughs or sneezes.",
                "Response": "Cystometry in this case of stress urinary incontinence would most likely reveal a normal post-void residual volume with increased pressures during provocative maneuvers."
            }
        ]
        
        # Save sample data
        with open('medical_qa_data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    
    # Create a dataset
    dataset = MedicalQADataset(data, word_tokenize)
    
    # Split into train and validation sets (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    # Initialize model
    vocab_size = len(dataset.vocab)
    model = MedicalQAModel(vocab_size).to(device)
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    # Training loop
    num_epochs = 1
    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(num_epochs):
        # Train
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                questions = batch["question"].to(device)
                responses = batch["response"].to(device)
                
                output = model(questions, responses)
                
                output = output[:, 1:].reshape(-1, output.shape[-1])
                target = responses[:, 1:].reshape(-1)
                
                loss = criterion(output, target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_medical_qa_model.pt')
    
    # Load best model for inference
    model.load_state_dict(torch.load('best_medical_qa_model.pt'))
    
    # Interactive loop
    print("\nModel training complete! Enter your medical questions (or 'quit' to exit):")
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == 'quit':
            break
        
        response = generate_response(model, user_input, dataset, device)
        print(f"\nAI Response: {response}")

if __name__ == "__main__":
    main()