d_model = 512
d_ff = d_model * 4 # From attention is all you need (4x the model dimension)
vocab_size=10000

max_seq_length = 512
batch_size = 32
learning_rate = 1e-3
num_epochs = 3
device = "cpu"