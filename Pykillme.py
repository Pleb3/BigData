import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import string
from sklearn.model_selection import train_test_split
import pymongo
import torch.nn as nn

# Load data from MongoDB
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["hotel"]
collection = db["reviews_clean"]

data = list(collection.find({}, {"Review": 1, "Label": 1}).limit(25000))
df = pd.DataFrame(data)
df.drop(columns=["_id"], inplace=True)

# Preprocess data
punctuation = string.punctuation
all_reviews = []
for text in df['Review']:
    if text is not None:
        text = text.lower()
        text = "".join([ch for ch in text if ch not in punctuation])
        all_reviews.append(text)
    else:
        all_reviews.append("")  # or some other default value

# Tokenize the reviews
tokenized_reviews = []
for review in all_reviews:
    tokens = review.split()
    tokenized_reviews.append(tokens)

# Create a vocabulary from the tokenized reviews
vocab_to_int = {}
word_count = 0
for review in tokenized_reviews:
    for word in review:
        if word not in vocab_to_int:
            vocab_to_int[word] = word_count
            word_count += 1

# Encode the tokenized reviews using the vocabulary
encoded_reviews = []
for review in tokenized_reviews:
    encoded_review = []
    for word in review:
        encoded_review.append(vocab_to_int[word])
    encoded_reviews.append(encoded_review)

sequence_length = 50
features = np.zeros((len(encoded_reviews), sequence_length), dtype=int)

for i, review in enumerate(encoded_reviews):
    review_len = len(review)
    if review_len <= sequence_length:
        zeros = list(np.zeros(sequence_length - review_len))
        new = zeros + review
    else:
        new = review[:sequence_length]
    features[i, :] = np.array(new)

y = df['Label']

# Split dataset into 80% training, 10% test, and 10% validation
train_x, test_x, train_y, test_y = train_test_split(features, y, test_size=0.2, random_state=42)
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# Convert to tensors
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y.to_numpy()))
valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y.to_numpy()))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y.to_numpy()))

batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

# Define the model
class SentimentalLSTM(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super().__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size()

        embedd = self.embedding(x)
        lstm_out, hidden = self.lstm(embedd, hidden)

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        sig_out = self.sigmoid(out)

        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]

        return sig_out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        if torch.cuda.is_available():
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden

vocab_size = len(vocab_to_int)


# Initialize the model, loss function, and optimizer
vocab_size = len(vocab_to_int)
output_size = 1
embedding_dim = 256
hidden_dim = 256
n_layers = 2
drop_prob = 0.5

model = SentimentalLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

## Train the model
for epoch in range(5):
    for x, y in train_loader:
        hidden = model.init_hidden(batch_size)
        optimizer.zero_grad()
        # Clip x to the correct range
        x = torch.clamp(x, min=0, max=vocab_size - 1)
        output, hidden = model(x, hidden)
        loss = criterion(output.squeeze(), y.float())
        loss.backward()
        optimizer.step()

# Test the model
model.eval()
test_loss = 0
correct = 0
last_10_results = []
with torch.no_grad():
    for i, (x, y) in enumerate(test_loader):
        hidden = model.init_hidden(batch_size)
        output, hidden = model(x, hidden)
        loss = criterion(output.squeeze(), y.float())
        test_loss += loss.item()
        pred = torch.round(output)
        correct += (pred == y.float()).sum().item()
        if i >= len(test_loader) - 10:
            for j in range(batch_size):
                last_10_results.append((y[j].item(), pred[j].item()))

accuracy = correct / len(test_loader.dataset)
print(f'Test Loss: {test_loss / len(test_loader)}')
print(f'Test Accuracy: {accuracy:.3f}')

print("Last 10 results:")
for i, (label, pred) in enumerate(last_10_results):
    print(f"Review {i+1}: Label={label}, Predicted={pred} ({'Positive' if pred == 1 else 'Negative'})")



#####TEST
# Assume you have a function to tokenize and convert text to indices
def tokenize_text(text, vocab):
    # Tokenize the text
    tokens = [word for word in text.split()]
    # Convert tokens to indices
    indices = [vocab[word] for word in tokens if word in vocab]
    return indices
# Assume you have a function to pad sequences to a fixed length
def pad_sequence(sequence, max_length):
    sequence = sequence[:max_length]
    padded_sequence = sequence + [0] * (max_length - len(sequence))
    return padded_sequence

# Load your vocabulary
vocab = vocab_to_int  # Load your vocabulary

# Define multiple reviews
reviews = [
    "What a delightful stay! The hotel staff went above and beyond to ensure our comfort. From the warm welcome at check-in to the personalized service throughout our stay, we felt truly pampered. The room was immaculate, with stunning views of the city skyline. The amenities were top-notch, especially the spa facilities. We left feeling rejuvenated and already planning our next visit!",
    "Disappointing experience overall. Despite booking a 'deluxe' room, we found the accommodations to be lackluster and outdated. The furnishings were worn, and the carpet appeared dirty. Additionally, the noise from nearby construction made it difficult to relax during the day. We were also underwhelmed by the breakfast offerings, which lacked variety and freshness. Overall, not worth the high price tag.",
    "Exceptional experience from start to finish! The location of this hotel is unbeatable, situated in the heart of the bustling downtown area. Despite the central location, the room was surprisingly quiet, allowing for a restful night's sleep. The decor was chic and modern, creating a relaxing atmosphere. The breakfast buffet was a highlight, offering a wide variety of delicious options. Highly recommend this hotel for both business and leisure travelers.",
    "Unpleasant stay from start to finish. The check-in process was chaotic, with long lines and disorganized staff. Our room was not ready upon arrival, despite arriving well after the designated check-in time. When we finally received our keys, we were dismayed to find that the room was not as advertised, with limited amenities and a musty odor. The hotel's location was also inconvenient, requiring a lengthy commute to reach popular attractions. Would not recommend.",
    "Our family had an amazing time at this hotel! The kids loved the pool area, which was clean and well-maintained. The staff were incredibly friendly and attentive, making sure we had everything we needed throughout our stay. The suite we stayed in was spacious and comfortable, providing plenty of room for our family of four. Plus, the hotel's proximity to local attractions made exploring the city a breeze. We can't wait to come back!",
    "Absolutely fantastic stay! The attention to detail at this hotel is truly remarkable. From the luxurious linens to the gourmet dining options, every aspect exceeded our expectations. We especially enjoyed the evening turndown service, which included complimentary chocolates and bottled water. The concierge was also incredibly helpful in recommending activities and restaurants in the area. Overall, a five-star experience that we won't soon forget!",
    "Poor customer service ruined what could have been a pleasant stay. Despite notifying the front desk of issues with our room, including a malfunctioning air conditioning unit and a broken shower head, no action was taken to address these issues. We were left feeling frustrated and uncomfortable throughout our stay. Additionally, the hotel's Wi-Fi was unreliable, making it difficult to stay connected. Overall, a disappointing experience that fell far short of expectations.",
    "Impeccable service and stunning views! We were greeted with a warm welcome upon arrival and were immediately impressed by the elegant lobby. Our room overlooked the ocean, providing breathtaking sunset views each evening. The hotel's restaurant offered delicious cuisine with attentive service. We appreciated the little touches, such as the nightly live music in the lounge and the complimentary shuttle service to nearby attractions. Highly recommend this hotel for a luxurious getaway.",
    "Overpriced and underwhelming. While the hotel's website boasted luxurious accommodations and world-class amenities, the reality was far from it. Our room was cramped and poorly laid out, with outdated furnishings and subpar bedding. The noise from neighboring rooms and the street below made it difficult to sleep at night. The hotel's restaurant was equally disappointing, with lackluster food and slow service. Save your money and stay elsewhere.",
    "Total waste of money, never go there again."
]

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Instance maken voor WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Stopwoorden in het Engels ophalen
stop_words = set(stopwords.words('english'))

# Lowercase, punten en comma's, lemmatizer toepassen.
def lemmatize_text(reviews):
    reviews = reviews.lower()
    reviews = reviews.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    tokens = reviews.split()
    reviews = ' '.join([lemmatizer.lemmatize(word) for word in tokens if word not in stop_words])
    return reviews

# Process each review
predictions = []
for review in reviews:
    # Preprocess the review
    review = lemmatize_text(review)
    review_indices = tokenize_text(review, vocab)
    review_indices = pad_sequence(review_indices, max_length=100)

    # Convert the review to a tensor
    review_tensor = torch.tensor(review_indices).unsqueeze(0)  # Add a batch dimension

    # Initialize the hidden state
    hidden = model.init_hidden(1)

    # Pass the review through the model
    output, _ = model(review_tensor, hidden)

    # Get the prediction
    prediction = torch.sigmoid(output).squeeze().item()

    # Append the prediction to the list
    predictions.append(prediction)

    # Debugging checks
    print(f"Review: {review}")
    print(f"Tokenized review: {review_indices}")
    print(f"Prediction: {prediction:.4f}")
    print()

# Print the predictions
for i, prediction in enumerate(predictions):
    print(f"Review {i+1}: {prediction:.4f}")
    if prediction > 0.5:
        print(f"Review {i+1} is positive!")
    else:
        print(f"Review {i+1} is negative!")