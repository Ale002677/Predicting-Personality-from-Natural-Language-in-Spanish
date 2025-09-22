# This script implements a deep learning model for predicting the
# Big Five Personality Traits from Spanish text.
# It uses a combined architecture that processes both text sequences
# via a Long Short-Term Memory (LSTM) Recurrent Neural Network
# and features extracted using TF-IDF and Singular Value Decomposition (SVD).
# Hyperparameter optimization is performed using Optuna.
# -----------------------------------------------------------------------------

# --- Standard and Third-Party Library Imports ---
import os
import re
import random
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from langdetect import detect, LangDetectException
import optuna

# NEW: Imports for visualization
import matplotlib.pyplot as plt
from torchviz import make_dot # You will need: pip install torchviz graphviz
                             # And have Graphviz installed on your system: https://graphviz.org/download/

# --- Configuration for Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Path Definitions for Datasets ---
PATH_TRAIN_DATASET = r'C:\Path\To\Your\pan15-author-profiling-training-dataset-spanish' # MODIFY PATH
PATH_TEST_DATASET = r'C:\Path\To\Your\pan15-author-profiling-test-dataset-spanish'       # MODIFY PATH
FILE_TRAIN_TRUTH = os.path.join(PATH_TRAIN_DATASET, 'truth.txt')
FILE_TEST_TRUTH = os.path.join(PATH_TEST_DATASET, 'truth.txt')

# NEW: Paths for saving final model artifacts
OUTPUT_DIR = "final_model_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PATH_FINAL_MODEL_CHECKPOINT = os.path.join(OUTPUT_DIR, 'final_personality_model.pth')
PATH_ARCHITECTURE_DIAGRAM = os.path.join(OUTPUT_DIR, 'model_architecture.pdf')
PATH_TRAINING_PLOT = os.path.join(OUTPUT_DIR, 'training_evolution.png')


# --- Path Existence Verification ---
for path, name in zip([PATH_TRAIN_DATASET, PATH_TEST_DATASET, FILE_TRAIN_TRUTH, FILE_TEST_TRUTH],
                      ['Training Directory', 'Test Directory',
                       'Labels File (Training)', 'Labels File (Test)']):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Resource not found: {name} at path '{path}'. "
                                "Please verify the data paths.")

# --- NLTK Resource Download ---
print("Verifying and downloading NLTK resources if necessary...")
resources_to_download = ['punkt', 'punkt_tab']
all_resources_found_nltk = True

for resource in resources_to_download:
    try:
        if resource == 'punkt':
            nltk.data.find('tokenizers/punkt')
        elif resource == 'punkt_tab':
            nltk.data.find('tokenizers/punkt_tab')
        print(f"NLTK resource '{resource}' found.")
    except nltk.downloader.DownloadError:
        print(f"NLTK resource '{resource}' NOT found. Attempting to download...")
        all_resources_found_nltk = False
        try:
            nltk.download(resource, quiet=True)
            print(f"Resource '{resource}' downloaded successfully.")
            if resource == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif resource == 'punkt_tab':
                nltk.data.find('tokenizers/punkt_tab')
        except Exception as download_exc:
            print(f"FAILED to download or verify NLTK resource '{resource}': {download_exc}")
            print("Please try downloading it manually by running in a Python console:")
            print("import nltk")
            print(f"nltk.download('{resource}')")
            raise RuntimeError(f"Could not download required NLTK resource '{resource}'.") from download_exc

if all_resources_found_nltk:
    print("All necessary NLTK resources ('punkt', 'punkt_tab') are present.")
print("--- End of NLTK resource check ---\n")

# -----------------------------------------------------------------------------
# Section 1: Data Loading and Preprocessing
# -----------------------------------------------------------------------------
def load_personality_labels(truth_file_path):
    personality_labels = {}
    with open(truth_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(':::')
            if len(parts) < 8:
                print(f"Warning: Malformed line in {truth_file_path}: {line.strip()}")
                continue
            user_id = parts[0]
            try:
                personality_labels[user_id] = {
                    'extroversion': float(parts[3]),
                    'neuroticism': float(parts[4]),
                    'agreeableness': float(parts[5]),
                    'conscientiousness': float(parts[6]),
                    'openness': float(parts[7])
                }
            except ValueError as e:
                print(f"Warning: Error converting value to float for user {user_id} in {truth_file_path}: {e}")
                continue
    return personality_labels

train_labels_dict = load_personality_labels(FILE_TRAIN_TRUTH)
test_labels_dict = load_personality_labels(FILE_TEST_TRUTH)
all_user_labels = {**train_labels_dict, **test_labels_dict}

def get_xml_file_paths(directory_path):
    return [os.path.join(directory_path, f) for f in os.listdir(directory_path)
            if f.endswith('.xml') and os.path.isfile(os.path.join(directory_path, f))]

train_xml_files = get_xml_file_paths(PATH_TRAIN_DATASET)
test_xml_files = get_xml_file_paths(PATH_TEST_DATASET)

if not train_xml_files or not test_xml_files:
    raise ValueError("No XML files were found in the data directories.")

print(f"Number of training XML files loaded: {len(train_xml_files)}")
print(f"Number of test XML files loaded: {len(test_xml_files)}")

def preprocess_text_content(text_content):
    if not isinstance(text_content, str): return ""
    text_content = text_content.lower()
    text_content = re.sub(r'<.*?>', ' ', text_content)
    text_content = re.sub(r'[^a-záéíóúñü\s]', ' ', text_content)
    text_content = re.sub(r'\s+', ' ', text_content).strip()
    return text_content

def is_spanish_language(text_sample, min_length=20):
    if not text_sample or len(text_sample) < min_length: return False
    try: return detect(text_sample) == 'es'
    except LangDetectException: return False

def extract_text_data_from_xml(xml_file_list, user_label_dict,
                               raw_text_list_output, ordered_user_id_list_output):
    texts_for_vocabulary_construction = []
    processed_user_ids = set()
    for xml_file_path in xml_file_list:
        try:
            xml_tree = ET.parse(xml_file_path)
            root_element = xml_tree.getroot()
            user_id = root_element.attrib.get('id')
            if user_id is None or user_id not in user_label_dict or user_id in processed_user_ids:
                continue
            document_elements = root_element.findall('document')
            user_texts = [preprocess_text_content(doc.text) for doc in document_elements if doc.text is not None]
            combined_user_text = ' '.join(filter(None, user_texts)).strip()
            if not combined_user_text or not is_spanish_language(combined_user_text):
                continue
            texts_for_vocabulary_construction.append(combined_user_text)
            raw_text_list_output.append(combined_user_text)
            ordered_user_id_list_output.append(user_id)
            processed_user_ids.add(user_id)
        except (ET.ParseError, AttributeError) as e:
            print(f"Error processing XML file {xml_file_path}: {e}")
    return texts_for_vocabulary_construction

all_texts_for_vocab_build = []
train_raw_texts, test_raw_texts = [], []
train_ordered_ids, test_ordered_ids = [], []

all_texts_for_vocab_build.extend(extract_text_data_from_xml(train_xml_files, train_labels_dict, train_raw_texts, train_ordered_ids))
all_texts_for_vocab_build.extend(extract_text_data_from_xml(test_xml_files, test_labels_dict, test_raw_texts, test_ordered_ids))

if not all_texts_for_vocab_build:
    raise ValueError("No valid texts were extracted for vocabulary construction.")

MIN_WORD_FREQUENCY = 2
MAX_SEQUENCE_LENGTH = 256

def build_vocabulary_from_texts(text_corpus, min_frequency=1):
    word_counter = Counter()
    for text_sample in text_corpus:
        word_counter.update(word_tokenize(text_sample))
    vocabulary = {'<PAD>': 0, '<UNK>': 1}
    current_idx = 2
    for word, count in word_counter.items():
        if count >= min_frequency:
            vocabulary[word] = current_idx
            current_idx += 1
    return vocabulary

text_vocabulary = build_vocabulary_from_texts(all_texts_for_vocab_build, MIN_WORD_FREQUENCY)
VOCABULARY_SIZE = len(text_vocabulary)
print(f"Constructed vocabulary size: {VOCABULARY_SIZE}")

def convert_text_to_sequence(text_sample, vocabulary, max_len):
    tokens = word_tokenize(text_sample)
    sequence = [vocabulary.get(token, vocabulary['<UNK>']) for token in tokens]
    return sequence[:max_len]

train_ordered_labels_np = np.array([
    [all_user_labels[uid]['extroversion'], all_user_labels[uid]['neuroticism'],
     all_user_labels[uid]['agreeableness'], all_user_labels[uid]['conscientiousness'],
     all_user_labels[uid]['openness']]
    for uid in train_ordered_ids
], dtype=np.float32)

test_ordered_labels_np = np.array([
    [all_user_labels[uid]['extroversion'], all_user_labels[uid]['neuroticism'],
     all_user_labels[uid]['agreeableness'], all_user_labels[uid]['conscientiousness'],
     all_user_labels[uid]['openness']]
    for uid in test_ordered_ids
], dtype=np.float32)

TFIDF_NGRAM_RANGE = (1, 1)
TFIDF_MAX_FEATURES = 5000
SVD_N_COMPONENTS = 200

feature_extraction_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=TFIDF_NGRAM_RANGE, max_features=TFIDF_MAX_FEATURES)),
    ('svd', TruncatedSVD(n_components=SVD_N_COMPONENTS, random_state=SEED)),
    ('scaler', StandardScaler())
])

train_tfidf_svd_features_np = feature_extraction_pipeline.fit_transform(train_raw_texts)
test_tfidf_svd_features_np = feature_extraction_pipeline.transform(test_raw_texts)
TFIDF_SVD_FEATURE_DIM = train_tfidf_svd_features_np.shape[1]
print(f"Dimension of TF-IDF+SVD features: {TFIDF_SVD_FEATURE_DIM}")


# -----------------------------------------------------------------------------
# Section 2: PyTorch Dataset and DataLoader Definition
# -----------------------------------------------------------------------------
class MultimodalAuthorProfilingDataset(Dataset):
    def __init__(self, raw_text_samples, tfidf_svd_feature_matrix, personality_label_matrix,
                 vocabulary, max_sequence_len):
        self.raw_text_samples = raw_text_samples
        if hasattr(tfidf_svd_feature_matrix, "toarray"):
            self.tfidf_svd_features = torch.tensor(tfidf_svd_feature_matrix.toarray(), dtype=torch.float32)
        else:
            self.tfidf_svd_features = torch.tensor(tfidf_svd_feature_matrix, dtype=torch.float32)
        self.personality_labels = torch.tensor(personality_label_matrix, dtype=torch.float32)
        self.vocabulary = vocabulary
        self.max_sequence_len = max_sequence_len

    def __len__(self):
        return len(self.raw_text_samples)

    def __getitem__(self, index):
        text_sample = self.raw_text_samples[index]
        numerical_sequence = convert_text_to_sequence(text_sample, self.vocabulary, self.max_sequence_len)
        sequence_tensor = torch.tensor(numerical_sequence, dtype=torch.long)
        tfidf_feature_vector = self.tfidf_svd_features[index]
        personality_label_vector = self.personality_labels[index]
        return sequence_tensor, tfidf_feature_vector, personality_label_vector

train_torch_dataset = MultimodalAuthorProfilingDataset(
    train_raw_texts, train_tfidf_svd_features_np, train_ordered_labels_np,
    text_vocabulary, MAX_SEQUENCE_LENGTH
)
test_torch_dataset = MultimodalAuthorProfilingDataset(
    test_raw_texts, test_tfidf_svd_features_np, test_ordered_labels_np,
    text_vocabulary, MAX_SEQUENCE_LENGTH
)

if len(train_torch_dataset) == 0 or len(test_torch_dataset) == 0:
    raise ValueError("Error: One or both PyTorch datasets are empty.")

def collate_batch_elements(batch_data):
    text_sequences, tfidf_vectors, label_vectors = zip(*batch_data)
    padded_text_sequences = pad_sequence(text_sequences, batch_first=True,
                                         padding_value=text_vocabulary['<PAD>'])
    tfidf_tensor = torch.stack(tfidf_vectors)
    labels_tensor = torch.stack(label_vectors)
    return padded_text_sequences, tfidf_tensor, labels_tensor

BATCH_SIZE = 32
train_dataloader = DataLoader(train_torch_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch_elements)
test_dataloader = DataLoader(test_torch_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=collate_batch_elements)

# -----------------------------------------------------------------------------
# Section 3: Combined Neural Network Model Definition
# -----------------------------------------------------------------------------
class CombinedPersonalityPredictionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dimension, lstm_hidden_dimension,
                 tfidf_input_dimension, combined_layer_hidden_dimension, output_dimension,
                 num_lstm_layers=1, lstm_dropout=0.0, combined_dropout=0.0,
                 embedding_padding_idx=0):
        super(CombinedPersonalityPredictionModel, self).__init__()
        self.text_embedding_layer = nn.Embedding(vocab_size, embedding_dimension,
                                                 padding_idx=embedding_padding_idx)
        self.lstm_layer = nn.LSTM(embedding_dimension, lstm_hidden_dimension,
                                  num_layers=num_lstm_layers, batch_first=True,
                                  dropout=lstm_dropout if num_lstm_layers > 1 else 0.0)
        self.fc_tfidf_1 = nn.Linear(tfidf_input_dimension, tfidf_input_dimension // 2)
        self.tfidf_branch_dropout = nn.Dropout(combined_dropout)
        combined_input_dim = lstm_hidden_dimension + (tfidf_input_dimension // 2)
        self.fc_combined_1 = nn.Linear(combined_input_dim, combined_layer_hidden_dimension)
        self.final_dropout_layer = nn.Dropout(combined_dropout)
        self.output_layer = nn.Linear(combined_layer_hidden_dimension, output_dimension)

    def forward(self, text_input_sequences, tfidf_input_features):
        embedded_text = self.text_embedding_layer(text_input_sequences)
        _, (last_hidden_state, _) = self.lstm_layer(embedded_text)
        text_features_representation = last_hidden_state[-1, :, :]
        processed_tfidf_features = F.relu(self.fc_tfidf_1(tfidf_input_features))
        processed_tfidf_features = self.tfidf_branch_dropout(processed_tfidf_features)
        combined_features = torch.cat((text_features_representation, processed_tfidf_features), dim=1)
        x = F.relu(self.fc_combined_1(combined_features))
        x = self.final_dropout_layer(x)
        predictions = self.output_layer(x)
        return predictions

# -----------------------------------------------------------------------------
# Section 4: Model Training and Evaluation Functions
# -----------------------------------------------------------------------------
def evaluate_model_performance(model_instance, data_loader, loss_criterion, computation_device):
    model_instance.eval()
    accumulated_loss = 0.0
    all_model_predictions = []
    all_ground_truth_labels = []
    with torch.no_grad():
        for text_seq, tfidf_feat, true_labels in data_loader:
            text_seq, tfidf_feat, true_labels = (data.to(computation_device) for data in [text_seq, tfidf_feat, true_labels])
            model_outputs = model_instance(text_seq, tfidf_feat)
            current_loss = loss_criterion(model_outputs, true_labels)
            accumulated_loss += current_loss.item()
            all_model_predictions.extend(model_outputs.cpu().numpy())
            all_ground_truth_labels.extend(true_labels.cpu().numpy())
    avg_loss = accumulated_loss / len(data_loader)
    predictions_np = np.array(all_model_predictions)
    true_labels_np = np.array(all_ground_truth_labels)
    global_mse = mean_squared_error(true_labels_np, predictions_np)
    global_r2 = r2_score(true_labels_np, predictions_np)
    num_traits = true_labels_np.shape[1]
    mse_per_trait_list = [mean_squared_error(true_labels_np[:, i], predictions_np[:, i]) for i in range(num_traits)]
    r2_per_trait_list = [r2_score(true_labels_np[:, i], predictions_np[:, i]) for i in range(num_traits)]
    return avg_loss, global_mse, global_r2, mse_per_trait_list, r2_per_trait_list

global_best_r2_score_optuna = -float('inf')
PATH_OPTUNA_BEST_MODEL_CHECKPOINT = os.path.join(OUTPUT_DIR, 'optuna_best_model_checkpoint.pth')

def optuna_objective_function(optuna_trial):
    global global_best_r2_score_optuna

    # Define hyperparameter search space
    optuna_lr = optuna_trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    optuna_emb_dim = optuna_trial.suggest_categorical('embedding_dim_optuna', [50, 100, 150, 200])
    optuna_lstm_hidden = optuna_trial.suggest_int('lstm_hidden_dim_optuna', 64, 256, step=32)
    optuna_num_lstm_layers = optuna_trial.suggest_categorical('num_lstm_layers', [1, 2])
    optuna_lstm_dropout = optuna_trial.suggest_float('lstm_dropout', 0.1, 0.5) if optuna_num_lstm_layers > 1 else 0.0
    optuna_combined_hidden = optuna_trial.suggest_int('combined_hidden_dim_optuna', 64, 256, step=32)
    optuna_combined_dropout = optuna_trial.suggest_float('combined_dropout', 0.2, 0.6)
    optuna_loss_type = optuna_trial.suggest_categorical('loss_function_type', ['MSELoss', 'HuberLoss'])
    optuna_weight_decay = optuna_trial.suggest_float('optimizer_weight_decay', 1e-6, 1e-3, log=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_hyperparams_dict = {
        'vocab_size': VOCABULARY_SIZE,
        'embedding_dimension': optuna_emb_dim,
        'lstm_hidden_dimension': optuna_lstm_hidden,
        'tfidf_input_dimension': TFIDF_SVD_FEATURE_DIM,
        'combined_layer_hidden_dimension': optuna_combined_hidden,
        'output_dimension': train_ordered_labels_np.shape[1],
        'num_lstm_layers': optuna_num_lstm_layers,
        'lstm_dropout': optuna_lstm_dropout,
        'combined_dropout': optuna_combined_dropout,
        'embedding_padding_idx': text_vocabulary['<PAD>']
    }
    model = CombinedPersonalityPredictionModel(**model_hyperparams_dict).to(device)

    criterion = nn.MSELoss() if optuna_loss_type == 'MSELoss' else nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=optuna_lr, weight_decay=optuna_weight_decay)
    
    NUM_EPOCHS_OPTUNA = 50
    EARLY_STOPPING_PATIENCE = 7
    epochs_without_improvement = 0
    best_r2_for_this_trial = -float('inf')

    for epoch_num in range(NUM_EPOCHS_OPTUNA):
        model.train()
        for text_seq_batch, tfidf_feat_batch, labels_batch in train_dataloader:
            text_seq_batch, tfidf_feat_batch, labels_batch = (
                data.to(device) for data in [text_seq_batch, tfidf_feat_batch, labels_batch]
            )
            optimizer.zero_grad()
            predictions_batch = model(text_seq_batch, tfidf_feat_batch)
            loss_batch = criterion(predictions_batch, labels_batch)
            loss_batch.backward()
            optimizer.step()
        
        val_loss, _, val_r2_global, _, _ = evaluate_model_performance(model, test_dataloader, criterion, device)
        
        if val_r2_global > best_r2_for_this_trial:
            best_r2_for_this_trial = val_r2_global
            epochs_without_improvement = 0
            if val_r2_global > global_best_r2_score_optuna:
                global_best_r2_score_optuna = val_r2_global
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'vocabulary': text_vocabulary,
                    'tfidf_svd_pipeline': feature_extraction_pipeline,
                    'model_hyperparameters': model_hyperparams_dict,
                    'best_r2_score': global_best_r2_score_optuna,
                    'optuna_trial_params': optuna_trial.params
                }, PATH_OPTUNA_BEST_MODEL_CHECKPOINT)
                print(f"Optuna Trial {optuna_trial.number} - New best Optuna model saved. "
                      f"R^2: {global_best_r2_score_optuna:.4f}, Epoch: {epoch_num + 1}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Optuna Trial {optuna_trial.number} - Early stopping at epoch {epoch_num + 1}.")
                break
        
        optuna_trial.report(val_r2_global, epoch_num)
        if optuna_trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return best_r2_for_this_trial

# -----------------------------------------------------------------------------
# Section 4.5: Functions for Final Training and Visualization
# -----------------------------------------------------------------------------
def train_final_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, model_path_to_save):
    best_val_r2 = -float('inf')
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_mse': [], 'val_r2': []}
    temp_best_model_path = model_path_to_save.replace(".pth", "_temp_best.pth")

    print(f"\n--- Starting Final Model Training for {num_epochs} epochs ---")
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for text_seq, tfidf_feat, labels in train_loader:
            text_seq, tfidf_feat, labels = (d.to(device) for d in [text_seq, tfidf_feat, labels])
            optimizer.zero_grad()
            outputs = model(text_seq, tfidf_feat)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        val_loss, val_mse, val_r2, _, _ = evaluate_model_performance(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MSE: {val_mse:.4f}, Val R2: {val_r2:.4f}")

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_mse'].append(val_mse)
        history['val_r2'].append(val_r2)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save(model.state_dict(), temp_best_model_path)
            print(f"  Best model updated at epoch {epoch+1} with R2: {best_val_r2:.4f}")

    if os.path.exists(temp_best_model_path):
        model.load_state_dict(torch.load(temp_best_model_path))
        print(f"Final model loaded from the best training state (R2: {best_val_r2:.4f})")
        os.remove(temp_best_model_path)
    else:
        print("Warning: No temporary best model state found to load.")
    return model, history

def plot_training_history(history, save_path):
    epochs = history['epoch']
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['val_mse'], 'b-', label='MSE (Validation)')
    if 'train_loss' in history:
        plt.plot(epochs, history['train_loss'], 'r--', label='Loss (Training)')
    plt.title('Evolution of Mean Squared Error')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_r2'], 'g-', label='R² (Validation)')
    plt.title('Evolution of the Coefficient of Determination (R²)')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training evolution plot saved to: {save_path}")

def generate_architecture_diagram(model, sample_input_text, sample_input_tfidf, save_path):
    model.eval()
    try:
        device = next(model.parameters()).device
        sample_input_text = sample_input_text.to(device)
        sample_input_tfidf = sample_input_tfidf.to(device)
        output = model(sample_input_text, sample_input_tfidf)
        dot = make_dot(output, params=dict(model.named_parameters()))
        base_name_for_render = os.path.splitext(save_path)[0]
        dot.render(base_name_for_render, format='pdf', cleanup=True)
        print(f"Model architecture diagram saved to: {base_name_for_render}.pdf")
    except Exception as e:
        print(f"Error generating architecture diagram with torchviz: {e}")
        print("Ensure Graphviz is installed and in the system's PATH.")
        print("Also, verify that torchviz is installed (`pip install torchviz`).")

# -----------------------------------------------------------------------------
# Section 5: Execution of Optimization and Final Evaluation
# -----------------------------------------------------------------------------
if os.path.exists(PATH_OPTUNA_BEST_MODEL_CHECKPOINT):
    os.remove(PATH_OPTUNA_BEST_MODEL_CHECKPOINT)
    print(f"Previous Optuna checkpoint '{PATH_OPTUNA_BEST_MODEL_CHECKPOINT}' deleted.")

NUM_OPTUNA_TRIALS = 300
optuna_study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
optuna_study.optimize(optuna_objective_function, n_trials=NUM_OPTUNA_TRIALS, timeout=3600*0.5)

print("\nHyperparameter Optimization Completed.")
if optuna_study.best_trial:
    print(f"Best global R^2 found during optimization (Optuna): {optuna_study.best_value:.4f}")
    print(f"Best hyperparameters found by Optuna: {optuna_study.best_params}")

    if os.path.exists(PATH_OPTUNA_BEST_MODEL_CHECKPOINT):
        optuna_checkpoint_data = torch.load(PATH_OPTUNA_BEST_MODEL_CHECKPOINT, weights_only=False)
        best_hyperparams_from_optuna_checkpoint = optuna_checkpoint_data['model_hyperparameters']
        
        print(f"\nHyperparameters of the best Optuna trial (saved R^2: {optuna_checkpoint_data['best_r2_score']:.4f}):")
        for key, value in best_hyperparams_from_optuna_checkpoint.items():
            print(f"  {key}: {value}")

        device_final = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        final_model = CombinedPersonalityPredictionModel(**best_hyperparams_from_optuna_checkpoint).to(device_final)
        
        optuna_trial_params_for_final = optuna_checkpoint_data.get('optuna_trial_params', optuna_study.best_params)
        final_learning_rate = optuna_trial_params_for_final.get('learning_rate', 1e-3)
        final_weight_decay = optuna_trial_params_for_final.get('optimizer_weight_decay', 1e-5)
        final_loss_type = optuna_trial_params_for_final.get('loss_function_type', 'MSELoss')

        final_optimizer = torch.optim.Adam(final_model.parameters(), lr=final_learning_rate, weight_decay=final_weight_decay)
        final_criterion = nn.MSELoss() if final_loss_type == 'MSELoss' else nn.HuberLoss()

        NUM_EPOCHS_FINAL_TRAINING = 85
        final_model, training_history = train_final_model(
            final_model, train_dataloader, test_dataloader,
            final_optimizer, final_criterion, NUM_EPOCHS_FINAL_TRAINING,
            device_final, PATH_FINAL_MODEL_CHECKPOINT
        )

        torch.save({
            'model_state_dict': final_model.state_dict(),
            'vocabulary': text_vocabulary,
            'tfidf_svd_pipeline': feature_extraction_pipeline,
            'model_hyperparameters': best_hyperparams_from_optuna_checkpoint,
            'training_history': training_history
        }, PATH_FINAL_MODEL_CHECKPOINT)
        print(f"Final trained model saved to: {PATH_FINAL_MODEL_CHECKPOINT}")

        try:
            sample_text_seq, sample_tfidf_feat, _ = next(iter(test_dataloader))
            generate_architecture_diagram(final_model, sample_text_seq, sample_tfidf_feat, PATH_ARCHITECTURE_DIAGRAM)
        except Exception as e:
            print(f"Could not generate architecture diagram: {e}")

        plot_training_history(training_history, PATH_TRAINING_PLOT)

        print("\n--- Final Evaluation of the Trained Model on the Test Set ---")
        final_loss, final_mse, final_r2, mse_by_trait, r2_by_trait = evaluate_model_performance(
            final_model, test_dataloader, final_criterion, device_final
        )
        print(f"Average Loss: {final_loss:.4f}")
        print(f"Global Mean Squared Error (MSE): {final_mse:.4f}")
        print(f"Global Coefficient of Determination (R^2): {final_r2:.4f}")
        trait_names_list = ['Extraversion', 'Neuroticism (Stability Inverted)', 'Agreeableness',
                            'Conscientiousness', 'Openness to Experience']
        print("\nMetrics per Personality Trait:")
        for i, trait_name in enumerate(trait_names_list):
            print(f"  {trait_name}:")
            print(f"    MSE: {mse_by_trait[i]:.4f}, R^2: {r2_by_trait[i]:.4f}")
    else:
        print("\nNo checkpoint for the best Optuna model was found. "
              "The optimization may have failed to save a model or completed unsuccessfully.")
else:
    print("\nOptuna found no successful trials. Cannot proceed with final training.")

print("\n--- End of Script ---")