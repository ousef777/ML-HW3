import torch
import torch.nn as nn
import torch.optim as optim


from shakespeareLoader import train_loader, test_loader


X_train, y_train = next(iter(train_loader))
X_val, y_val = next(iter(test_loader))

# Defining the RNN model
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        #This line takes the input tensor x, which contains indices of characters, and passes it through an embedding layer (self.embedding). 
        #The embedding layer converts these indices into dense vectors of fixed size. 
        #These vectors are learned during training and can capture semantic similarities between characters. 
        #The result is a higher-dimensional representation of the input sequence, where each character index is replaced by its corresponding embedding vector. 
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        #The RNN layer returns two outputs: 
        #1- the output tensor containing the output of the RNN at each time step for each sequence in the batch, 
        #2-the hidden state (_) of the last time step (which is not used in this line, hence the underscore).
        output, _ = self.rnn(embedded)
        #The RNN's output contains the outputs for every time step, 
        #but for this task, we're only interested in the output of the last time step because we're predicting the next character after the sequence. 
        #output[:, -1, :] selects the last time step's output for every sequence in the batch (-1 indexes the last item in Python).
        output = self.fc(output[:, -1, :])  # Get the output of the last RNN cell
        return output

# Hyperparameters
hidden_size = 128
learning_rate = 0.001
epochs = 10

# Model, loss, and optimizer
model = CharRNN(len(X_train), hidden_size, len(y_train)).to(torch.device("cuda:0"))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
for epoch in range(epochs):
    #loss_train = 0.0
    model.train()
    for X_train, y_train in train_loader:
        X_train, y_train = X_train.to(torch.device("cuda:0")), y_train.to(torch.device("cuda:0"))
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        for X_val, y_val in test_loader:
            X_val, y_val = X_val.to(torch.device("cuda:0")), y_val.to(torch.device("cuda:0"))
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)
            #The use of the underscore _ is a common Python convention to indicate that the actual maximum values returned by torch.max are not needed and can be disregarded. 
            #What we are interested in is the indices of these maximum values, which are captured by the variable predicted. These indices represent the model's predictions for each example in the validation set.
            _, predicted = torch.max(val_output, 1)
            val_accuracy = (predicted == y_val).float().mean()
    
    #if (epoch) % 10 == 0:
    print(f'Epoch {epoch+1}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}, Validation Accuracy: {val_accuracy.item()}')