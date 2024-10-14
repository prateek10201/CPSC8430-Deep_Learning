import torch.optim as optim
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from scipy.special import expit
import sys
import os
import json
import re
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp

# Set multiprocessing start method
mp.set_start_method('spawn', force=True)

# Determine the device (GPU or CPU) for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to preprocess the data and create word mappings
def preprocess_training_data():
    with open('training_label.json', 'r') as f:
        data = json.load(f)

    word_count = {}
    for item in data:
        for caption in item['caption']:
            sentence = re.sub('[.!,;?]', ' ', caption).split()
            for word in sentence:
                word = word.replace('.', '') if '.' in word else word
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1

    word_dict = {word: count for word, count in word_count.items() if count > 4}
    special_tokens = [('<PAD>', 0), ('<BOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    indexToWord_mapping = {i + len(special_tokens): word for i, word in enumerate(word_dict)}
    word_to_index = {word: i + len(special_tokens) for i, word in enumerate(word_dict)}

    for token, index in special_tokens:
        indexToWord_mapping[index] = token
        word_to_index[token] = index

    return indexToWord_mapping, word_to_index, word_dict

# Convert sentences into sequences of word indices
def convert_sentence_to_indices(sentence, word_dict, word_to_index):
    sentence = re.sub(r'[.!,;?]', ' ', sentence).split()
    for i in range(len(sentence)):
        if sentence[i] not in word_dict:
            sentence[i] = 3  # <UNK> token
        else:
            sentence[i] = word_to_index[sentence[i]]
    sentence.insert(0, 1)  # <BOS> token
    sentence.append(2)  # <EOS> token
    return sentence

# Annotate the dataset with word indices for each caption
def annotate_dataset(label_file_path, word_dict, word_to_index):
    with open(label_file_path, 'r') as f:
        dataset = json.load(f)
    
    annotated_data = []
    for item in dataset:
        for caption in item['caption']:
            caption_indices = convert_sentence_to_indices(caption, word_dict, word_to_index)
            annotated_data.append((item['id'], caption_indices))
    
    return annotated_data

# Load video features from the specified directory
def load_video_features(features_dir):
    video_features = {}
    feature_files = os.listdir(features_dir)
    for i, file in enumerate(feature_files):
        print(f"Loading feature file {i+1}/{len(feature_files)}")
        feature_data = np.load(os.path.join(features_dir, file))
        video_features[file.split('.npy')[0]] = feature_data
    return video_features

# Batch processing to align video features and captions
def batch_data(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    video_features, captions = zip(*data)
    video_features = torch.stack(video_features, 0)

    lengths = [len(cap) for cap in captions]
    target_sequences = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        target_sequences[i, :end] = cap[:end]
    
    return video_features.to(device), target_sequences.to(device), lengths

# Dataset class for training
class VideoCaptionDataset(Dataset):
    def __init__(self, label_file_path, features_dir, word_dict, word_to_index):
        self.label_file_path = label_file_path
        self.features_dir = features_dir
        self.word_dict = word_dict
        self.video_features = load_video_features(label_file_path)
        self.word_to_index = word_to_index
        self.annotated_data = annotate_dataset(features_dir, word_dict, word_to_index)
        
    def __len__(self):
        return len(self.annotated_data)
    
    def __getitem__(self, idx):
        assert idx < self.__len__()
        video_id, sentence = self.annotated_data[idx]
        video_data = torch.Tensor(self.video_features[video_id]).to(device)
        video_data += torch.Tensor(video_data.size()).random_(0, 2000).to(device) / 10000.
        return video_data, torch.Tensor(sentence).to(device)

# Dataset class for testing
class VideoTestDataset(Dataset):
    def __init__(self, test_data_dir):
        self.video_data = []
        feature_files = os.listdir(test_data_dir)
        for file in feature_files:
            video_id = file.split('.npy')[0]
            feature_data = np.load(os.path.join(test_data_dir, file))
            self.video_data.append([video_id, feature_data])
    
    def __len__(self):
        return len(self.video_data)
    
    def __getitem__(self, idx):
        return self.video_data[idx]

# Attention mechanism for decoder
class AttentionMechanism(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionMechanism, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(2 * hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.attention_weights_fc = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden_state, encoder_outputs):
        batch_size, seq_len, feat_dim = encoder_outputs.size()
        decoder_hidden_state = decoder_hidden_state.view(batch_size, 1, feat_dim).repeat(1, seq_len, 1)
        concat_input = torch.cat((encoder_outputs, decoder_hidden_state), 2).view(-1, 2 * self.hidden_size)

        x = self.fc1(concat_input)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        attention_scores = self.attention_weights_fc(x)
        attention_scores = attention_scores.view(batch_size, seq_len)
        attention_weights = F.softmax(attention_scores, dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context_vector

# Encoder class to process video features
class VideoEncoder(nn.Module):
    def __init__(self):
        super(VideoEncoder, self).__init__()
        self.fc = nn.Linear(4096, 512)
        self.dropout = nn.Dropout(0.35)
        self.lstm = nn.LSTM(512, 512, batch_first=True)

    def forward(self, input_features):
        batch_size, seq_len, feat_dim = input_features.size()    
        input_features = input_features.view(-1, feat_dim)
        embedded_features = self.fc(input_features)
        embedded_features = self.dropout(embedded_features)
        embedded_features = embedded_features.view(batch_size, seq_len, 512)

        lstm_output, (hidden_state, context) = self.lstm(embedded_features)
        return lstm_output.to(device), hidden_state.to(device)

# Decoder class to generate captions
class CaptionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, word_dim, dropout_percentage=0.35):
        super(CaptionDecoder, self).__init__()

        self.hidden_size = 512
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.word_dim = word_dim

        self.embedding_layer = nn.Embedding(output_size, 1024).to(device)
        self.dropout = nn.Dropout(0.35)
        self.lstm = nn.LSTM(hidden_size + word_dim, hidden_size, batch_first=True).to(device)
        self.attention_mechanism = AttentionMechanism(hidden_size).to(device)
        self.fc_output = nn.Linear(hidden_size, output_size).to(device)

    def forward(self, encoder_hidden_state, encoder_outputs, target_sentences=None, mode='train', training_steps=None):
        _, batch_size, _ = encoder_hidden_state.size()
        
        decoder_hidden_state = encoder_hidden_state.to(device) if encoder_hidden_state is not None else None
        decoder_context = torch.zeros(decoder_hidden_state.size()).to(device)
        decoder_input_word = Variable(torch.ones(batch_size, 1)).long().to(device)

        sequence_log_probabilities = []
        predicted_sequences = []

        target_sentences = self.embedding_layer(target_sentences.to(device))
        _, sequence_len, _ = target_sentences.size()

        for i in range(sequence_len-1):
            teacher_forcing_threshold = self.calculate_teacher_forcing_threshold(training_steps)
            if random.uniform(0.05, 0.995) > teacher_forcing_threshold:  
                current_input_word = target_sentences[:, i]
            else:
                current_input_word = self.embedding_layer(decoder_input_word).squeeze(1)

            context_vector = self.attention_mechanism(decoder_hidden_state, encoder_outputs.to(device))
            lstm_input = torch.cat([current_input_word, context_vector], dim=1).unsqueeze(1)
            lstm_output, (decoder_hidden_state, decoder_context) = self.lstm(lstm_input, (decoder_hidden_state, decoder_context))

            log_probabilities = self.fc_output(lstm_output.squeeze(1))
            sequence_log_probabilities.append(log_probabilities.unsqueeze(1))
            decoder_input_word = log_probabilities.unsqueeze(1).max(2)[1]

        sequence_log_probabilities = torch.cat(sequence_log_probabilities, dim=1)
        predicted_sequences = sequence_log_probabilities.max(2)[1]
        return sequence_log_probabilities, predicted_sequences

    def calculate_teacher_forcing_threshold(self, training_steps):
        return expit(training_steps / 20 + 0.85)

    def inference_mode(self, encoder_hidden_state, encoder_outputs):
        _, batch_size, _ = encoder_hidden_state.size()
        decoder_hidden_state = encoder_hidden_state.to(device)
        decoder_input_word = Variable(torch.ones(batch_size, 1)).long().to(device)
        decoder_context = torch.zeros(decoder_hidden_state.size()).to(device)

        sequence_log_probabilities = []
        predicted_sequences = []
        max_sequence_len = 28
        
        for i in range(max_sequence_len-1):
            current_input_word = self.embedding_layer(decoder_input_word).squeeze(1)
            context_vector = self.attention_mechanism(decoder_hidden_state, encoder_outputs.to(device))
            lstm_input = torch.cat([current_input_word, context_vector], dim=1).unsqueeze(1)
            lstm_output, (decoder_hidden_state, decoder_context) = self.lstm(lstm_input, (decoder_hidden_state, decoder_context))

            log_probabilities = self.fc_output(lstm_output.squeeze(1))
            sequence_log_probabilities.append(log_probabilities.unsqueeze(1))
            decoder_input_word = log_probabilities.unsqueeze(1).max(2)[1]

        sequence_log_probabilities = torch.cat(sequence_log_probabilities, dim=1)
        predicted_sequences = sequence_log_probabilities.max(2)[1]
        return sequence_log_probabilities, predicted_sequences

# Complete model combining the encoder and decoder
class VideoCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(VideoCaptioningModel, self).__init__()
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

    def forward(self, video_features, mode, target_sentences=None, training_steps=None):
        encoder_outputs, encoder_hidden_state = self.encoder(video_features.to(device))
        if mode == 'train':
            sequence_log_probabilities, predicted_sequences = self.decoder(
                encoder_hidden_state=encoder_hidden_state, 
                encoder_outputs=encoder_outputs, 
                target_sentences=target_sentences, 
                mode=mode, 
                training_steps=training_steps
            )
        elif mode == 'inference':
            sequence_log_probabilities, predicted_sequences = self.decoder.inference_mode(
                encoder_hidden_state=encoder_hidden_state, 
                encoder_outputs=encoder_outputs
            )
        return sequence_log_probabilities, predicted_sequences

# Calculate loss for the training loop
def calculate_loss(loss_function, predictions, targets, sequence_lengths):
    batch_size = len(predictions)
    accumulated_predictions = None
    accumulated_targets = None
    initial_flag = True

    for batch in range(batch_size):
        prediction = predictions[batch]
        ground_truth = targets[batch]
        sequence_len = sequence_lengths[batch] - 1

        prediction = prediction[:sequence_len]
        ground_truth = ground_truth[:sequence_len]

        if initial_flag:
            accumulated_predictions = prediction
            accumulated_targets = ground_truth
            initial_flag = False
        else:
            accumulated_predictions = torch.cat((accumulated_predictions, prediction), dim=0)
            accumulated_targets = torch.cat((accumulated_targets, ground_truth), dim=0)

    loss = loss_function(accumulated_predictions, accumulated_targets)
    average_loss = loss / batch_size
    return loss

# Train the model for each epoch
def train_model(model, current_epoch, loss_function, optimizer, training_loader):
    model.train()
    print(f"Starting Epoch {current_epoch}...")

    for batch_idx, batch in enumerate(training_loader):
        video_features, ground_truths, sequence_lengths = batch
        video_features, ground_truths = Variable(video_features).to(device), Variable(ground_truths).to(device)

        optimizer.zero_grad()
        sequence_log_probabilities, predicted_sequences = model(
            video_features, target_sentences=ground_truths, mode='train', training_steps=current_epoch
        )
        ground_truths = ground_truths[:, 1:]  # Remove the <BOS> token
        
        loss = calculate_loss(loss_function, sequence_log_probabilities, ground_truths, sequence_lengths)
        print(f'Epoch {current_epoch}, Batch {batch_idx}, Loss: {loss}')
        loss.backward()
        optimizer.step()

    return loss.item()

# Test the model and generate captions
def generate_test_results(test_loader, model, indexToWord_mapping):
    model.eval()
    results = []

    for batch_idx, batch in enumerate(test_loader):
        video_ids, video_features = batch
        video_features = video_features.to(device)
        video_ids, video_features = video_ids, Variable(video_features).float().to(device)
        
        sequence_log_probabilities, predicted_sequences = model(video_features, mode='inference')
        test_predictions = predicted_sequences
        
        generated_captions = [
            [indexToWord_mapping[token.item()] if indexToWord_mapping[token.item()] != '<UNK>' else 'something' for token in sequence]
            for sequence in test_predictions
        ]
        cleaned_captions = [' '.join(sequence).split('<EOS>')[0] for sequence in generated_captions]
        batch_results = zip(video_ids, cleaned_captions)
        for result in batch_results:
            results.append(result)
    
    return results

# Main function for training and saving the model
def main():
    # Preprocess the data and get word mappings
    indexToWord_mapping, word_to_index, word_dict = preprocess_training_data()

    # Save the word mapping to a pickle file
    with open('indexToWord_mapping.pickle', 'wb') as file:
        pickle.dump(indexToWord_mapping, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Load the training data
    features_dir = 'training_data/feat'
    label_file_path = 'training_label.json'
    training_dataset = VideoCaptionDataset(features_dir, label_file_path, word_dict, word_to_index)
    training_loader = DataLoader(
        dataset=training_dataset, 
        batch_size=64, 
        shuffle=True, 
        num_workers=4, 
        collate_fn=batch_data
    )

    # Define the model and training parameters
    epochs = 100
    encoder = VideoEncoder()
    decoder = CaptionDecoder(512, len(indexToWord_mapping) + 4, len(indexToWord_mapping) + 4, 1024, 0.35)
    model = VideoCaptioningModel(encoder=encoder, decoder=decoder).to(device)
    
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_history = []

    # Train the model for the specified number of epochs
    for epoch in range(epochs):
        loss = train_model(model, epoch + 1, loss_function, optimizer, training_loader)
        loss_history.append(loss)
        
    # Save the loss values to a file
    with open('loss_values.txt', 'w') as file:
        for loss in loss_history:
            file.write(f"{loss}\n")
    
    # Save the trained model
    torch.save(model, "Prateek_Choudavarpu_seq2seq_model.h5")
    print("Model training completed and saved successfully!")

    # Plot the loss values and save as a JPEG file
    plt.figure()
    plt.plot(range(1, epochs+1), loss_history, marker='o', linestyle='-', color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Across Epochs')
    plt.savefig('training_loss_plot.jpeg')
    plt.show()

if __name__ == "__main__":
    main()
