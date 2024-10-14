import sys
import torch
import json
from torch.utils.data import DataLoader
from bleu_eval import BLEU
import pickle
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
import os
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Load the model and move to the correct device
trained_model = torch.load('Prateek_Choudavarpu_seq2seq_model.h5', map_location=device)

# Load test data
test_dataset = VideoTestDataset('{}'.format(sys.argv[1]))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Loading the index to word mapping
with open('indexToWord_mapping.pickle', 'rb') as handle:
    index_to_word_mapping = pickle.load(handle)

# Move the model to the appropriate device (if needed)
trained_model = trained_model.to(device)

# Run the test
test_results = generate_test_results(test_loader, trained_model, index_to_word_mapping)

# Write results to output file
with open(sys.argv[2], 'w') as output_file:
    for video_id, description in test_results:
        output_file.write('{},{}\n'.format(video_id, description))

# BLEU Score Evaluation
testing_labels = json.load(open("testing_label.json"))
output_file_name = sys.argv[2]
predicted_results = {}
with open(output_file_name, 'r') as output_file:
    for line in output_file:
        line = line.rstrip()
        split_index = line.index(',')
        video_id = line[:split_index]
        caption = line[split_index + 1:]
        predicted_results[video_id] = caption

bleu_scores = []
for item in testing_labels:
    candidate_captions = [x.rstrip('.') for x in item['caption']]
    bleu_score = BLEU(predicted_results[item['id']], candidate_captions, True)
    bleu_scores.append(bleu_score)

average_bleu_score = sum(bleu_scores) / len(bleu_scores)
print("Average BLEU score thus obtained is -> " + str(average_bleu_score))