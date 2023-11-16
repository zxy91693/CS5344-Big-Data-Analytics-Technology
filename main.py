import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from pytorch_pretrained_bert import BertConfig
import time
import copy

class fakenews_text_dataset(Dataset):
    def __init__(self,x_y_list, transform=None):

        self.x_y_list = x_y_list
        self.transform = transform

    def __getitem__(self,index):
        
        # task feature
        task_feature = self.x_y_list[0][index]
        task_feature = torch.from_numpy(np.array(task_feature))
        
        # Tokenize statements
        tokenized_statements = tokenizer.tokenize(self.x_y_list[1][index])
        if len(tokenized_statements) > max_seq_length_stat:
            tokenized_statements = tokenized_statements[:max_seq_length_stat]
        ids_statements  = tokenizer.convert_tokens_to_ids(tokenized_statements)
        padding = [0] * (max_seq_length_stat - len(ids_statements))
        ids_statements += padding
        assert len(ids_statements) == max_seq_length_stat
        ids_statements = torch.tensor(ids_statements)

        # Tokenize contextual
        tokenized_contextual = tokenizer.tokenize(self.x_y_list[2][index])
        if len(tokenized_contextual) > max_seq_length_stat:
            tokenized_contextual = tokenized_contextual[:max_seq_length_stat]
        ids_contextual  = tokenizer.convert_tokens_to_ids(tokenized_contextual)
        padding = [0] * (max_seq_length_stat - len(ids_contextual))
        ids_contextual += padding
        assert len(ids_contextual) == max_seq_length_stat
        ids_contextual = torch.tensor(ids_contextual)

        fakeness = self.x_y_list[3][index]
        list_of_labels = [torch.from_numpy(np.array(fakeness))]

        return [task_feature, ids_statements, ids_contextual], list_of_labels[0]

    def __len__(self):
        return len(self.x_y_list[0])


class sentimental_text_dataset(Dataset):
    def __init__(self,x_y_list, transform=None):

        self.x_y_list = x_y_list
        self.transform = transform

    def __getitem__(self,index):
        
        # task feature
        task_feature = self.x_y_list[0][index]
        task_feature = torch.from_numpy(np.array(task_feature))
        
        # Tokenize statements
        tokenized_statements = tokenizer.tokenize(self.x_y_list[1][index])
        if len(tokenized_statements) > max_seq_length_stat:
            tokenized_statements = tokenized_statements[:max_seq_length_stat]
        ids_statements  = tokenizer.convert_tokens_to_ids(tokenized_statements)
        padding = [0] * (max_seq_length_stat - len(ids_statements))
        ids_statements += padding
        assert len(ids_statements) == max_seq_length_stat
        ids_statements = torch.tensor(ids_statements)

        sentimental = self.x_y_list[2][index]
        list_of_labels = [torch.from_numpy(np.array(sentimental))]

        return [task_feature, ids_statements], list_of_labels[0]

    def __len__(self):
        return len(self.x_y_list[0])

class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class BertForSequenceClassification(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, num_labels=[2,3]): # Change number of labels here.
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_gate = nn.Sequential(
                    nn.Linear(1, config.hidden_size),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.Sigmoid(),
                )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier0 = nn.Linear(config.hidden_size*2, num_labels[0])
        self.classifier1 = nn.Linear(config.hidden_size, num_labels[1])
        #self.fc1 = nn.Linear(config.hidden_size*2, 512)
        nn.init.xavier_normal_(self.bert_gate[0].weight)
        nn.init.xavier_normal_(self.bert_gate[2].weight)
        nn.init.xavier_normal_(self.classifier0.weight)
        nn.init.xavier_normal_(self.classifier1.weight)

    '''def forward_once(self, x):
        # Forward pass
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output'''

    def forward_once(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        #logits = self.classifier(pooled_output)

        return pooled_output

    def forward(self, task, task_features, input_ids1, input_ids2): 
        if task == 'fakenews':
            # forward pass of input 1
            
            output1 = 2*self.bert_gate(task_features) * self.forward_once(input_ids1, token_type_ids=None, attention_mask=None, labels=None)
            # forward pass of input 2
            output2 = 2*self.bert_gate(task_features) * self.forward_once(input_ids2, token_type_ids=None, attention_mask=None, labels=None)

            out = torch.cat((output1, output2), 1)
            #print(out.shape)

            logits = self.classifier0(out)
        elif task == 'sentimental':
            # forward pass of input 1
            output1 = 2*self.bert_gate(task_features) * self.forward_once(input_ids1, token_type_ids=None, attention_mask=None, labels=None)
            
            #print(out.shape)
            logits = self.classifier1(output1)

        return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True
    
    
max_seq_length_stat = 64
max_seq_length_just = 256
max_seq_length_meta = 32

fakenews_batch_size = 128
sentimental_batch_size = 1024

config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

fakenews_data_path = './fakenews/kaggle_fake_train.csv'
fakenews_data_df = pd.read_csv(fakenews_data_path, header=[0]).drop(columns=['id'])
fakenews_data_df = fakenews_data_df.fillna('None')
data_df = shuffle(fakenews_data_df,random_state=2023).reset_index(drop=True)

len_data = len(data_df)

train_df = data_df.loc[0:int(len_data*0.8)-1,]
valid_df = data_df.loc[int(len_data*0.8):int(len_data*0.9)-1,]
test_df = data_df.loc[int(len_data*0.9):,]

train = train_df.values
valid = valid_df.values
test = test_df.values

def fakenews_to_onehot(a):
        a_cat = [0]*len(a)
        for i in range(len(a)):
            if a[i]== 1:
                a_cat[i] = [1,0]
            else:
                a_cat[i] = [0,1]
        return a_cat

fakenews_task_features = {'train':[[0.] for i in range(len(train))], 'valid':[[0.] for i in range(len(valid))], 'test':[[0.] for i in range(len(test))]}
fakenews_statements = {'train':[train[i][2] for i in range(len(train))], 'valid':[valid[i][2] for i in range(len(valid))], 'test':[test[i][2] for i in range(len(test))]}
fakenews_contextual = {'train':[train[i][0] + '. Written by ' + train[i][1] for i in range(len(train))], 'valid':[valid[i][0] + valid[i][1] for i in range(len(valid))], 'test':[test[i][0] + test[i][1] for i in range(len(test))]}
fakenews_labels = {'train':[train[i][3] for i in range(len(train))], 'valid':[valid[i][3] for i in range(len(valid))], 'test':[test[i][3] for i in range(len(test))]}
fakenews_labels_onehot = {'train':fakenews_to_onehot(fakenews_labels['train']), 'valid':fakenews_to_onehot(fakenews_labels['valid']), 'test':fakenews_to_onehot(fakenews_labels['test'])}


fakenews_train_lists = [fakenews_task_features['train'], fakenews_statements['train'], fakenews_contextual['train'], fakenews_labels_onehot['train']]
fakenews_valid_lists = [fakenews_task_features['valid'], fakenews_statements['valid'], fakenews_contextual['valid'], fakenews_labels_onehot['valid']]
fakenews_test_lists = [fakenews_task_features['test'], fakenews_statements['test'], fakenews_contextual['test'], fakenews_labels_onehot['test']]

fakenews_train_dataset = fakenews_text_dataset(x_y_list = fakenews_train_lists)
fakenews_valid_dataset = fakenews_text_dataset(x_y_list = fakenews_valid_lists)
fakenews_test_dataset = fakenews_text_dataset(x_y_list = fakenews_test_lists)

fakenews_dataloaders_dict = {'train': torch.utils.data.DataLoader(fakenews_train_dataset, batch_size=fakenews_batch_size, shuffle=False, num_workers=0),
                   'valid':torch.utils.data.DataLoader(fakenews_valid_dataset, batch_size=fakenews_batch_size, shuffle=False, num_workers=0),
                   'test':torch.utils.data.DataLoader(fakenews_test_dataset, batch_size=fakenews_batch_size, shuffle=False, num_workers=0)
                   }
fakenews_dataset_sizes = {'train':len(fakenews_train_lists[0]),
                'valid':len(fakenews_valid_lists[0]),
                'test':len(fakenews_test_lists[0])}
print(fakenews_dataset_sizes)

# Read and preprocess data
def read_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
        df = df.drop(columns=['id of the tweet', 'date of the tweet', 'query', 'user']).fillna('None')
        return shuffle(df, random_state=2023).reset_index(drop=True)
    except pd.errors.ParserError as e:
        print(f"An error occurred while reading the file: {e}")

sentimental_data_df = read_and_preprocess_data('./sentimental/sentimental_data.csv')

# Split data
def split_data(df, train_ratio=0.8, valid_ratio=0.1):
    len_data = len(df)
    train_end = int(len_data * train_ratio)
    valid_end = train_end + int(len_data * valid_ratio)
    return df.iloc[:train_end], df.iloc[train_end:valid_end], df.iloc[valid_end:]

train_df, valid_df, test_df = split_data(sentimental_data_df)

# Convert sentiment to one-hot encoding
def sentimental_to_onehot(sentiments):
    mapping = {0: [1, 0, 0], 2: [0, 1, 0], 4: [0, 0, 1]}
    return [mapping.get(sentiment, [0, 0, 1]) for sentiment in sentiments]

# Build datasets
sentimental_datasets = {}
for name, df in zip(['train', 'valid', 'test'], [train_df, valid_df, test_df]):
    features = [[1.]] * len(df)
    statements = df.iloc[:, 1].tolist()
    labels = df.iloc[:, 0].tolist()
    labels_onehot = sentimental_to_onehot(labels)
    sentimental_datasets[name] = {
        'features': features,
        'statements': statements,
        'labels_onehot': labels_onehot
    }


def create_dataloaders(datasets, batch_size):
    loaders = {}
    for name, data in datasets.items():
        dataset = sentimental_text_dataset(x_y_list=[data['features'], data['statements'], data['labels_onehot']])
        shuffle = False
        loaders[name] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loaders

sentimental_dataloaders_dict = create_dataloaders(sentimental_datasets, sentimental_batch_size)

# Dataset sizes
sentimental_dataset_sizes = {name: len(data['features']) for name, data in sentimental_datasets.items()}
print(sentimental_dataset_sizes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = BertForSequenceClassification()
model.freeze_bert_encoder()

log_file = "log.txt"

with open(log_file, "w") as f:  # 打开文件
     f.write("Begin！")

train_acc = []
valid_acc = []
train_loss = []
valid_loss = []

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    print('starting')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100
    best_acc = 0
    best_acc_save = None

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                #scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            fakeness_running_loss = 0.0

            fakeness_corrects = 0
            
            fakeness_predict=[]
            fakeness_label=[]
            # Iterate over data.
            pbar = tqdm(enumerate(fakenews_dataloaders_dict[phase]), total=len(fakenews_dataloaders_dict[phase]))
            for i, batchi in pbar:
                inputs, fakeness = batchi
                task_features = inputs[0].type(torch.float32).to(device) # task features

                inputs1 = inputs[1] # News statement input
                inputs2 = inputs[2] # Justification input

                inputs1 = inputs1.to(device)
                inputs2 = inputs2.to(device)

                fakeness = fakeness.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #print(inputs)
                    outputs = model('fakenews', task_features, inputs1, inputs2)

                    outputs = F.softmax(outputs,dim=1)

                    loss = criterion(outputs, torch.max(fakeness.float(), 1)[1])
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                fakeness_running_loss += loss.item() * inputs1.size(0)


                fakeness_corrects += torch.sum(torch.max(outputs, 1)[1] == torch.max(fakeness, 1)[1]) 
                
                if phase == 'valid':
                    fakeness_predict.extend((1-torch.max(outputs, 1)[1].cpu().numpy()).tolist())
                    fakeness_label.extend((1-torch.max(fakeness, 1)[1].cpu().numpy()).tolist())
                    # fakeness_predict=np.concatenate((fakeness_predict, 1-torch.max(outputs, 1)[1].cpu().numpy()),axis=0)
                    # fakeness_label=np.concatenate((fakeness_label,1-torch.max(fakeness, 1)[1].cpu().numpy()),axis=0)
                # else:
                #     break
            sentimental_running_loss = 0.0
            
            sentimental_corrects = 0
            
            sentimental_predict=[]
            sentimental_label=[]
            pbar = tqdm(enumerate(sentimental_dataloaders_dict[phase]), total=len(sentimental_dataloaders_dict[phase]))
            for i, batchi in pbar:
                inputs, sentimental = batchi
                task_features = inputs[0].type(torch.float32).to(device) # task features

                inputs1 = inputs[1] # News statement input
                inputs2 = None # Justification input

                inputs1 = inputs1.to(device)

                sentimental = sentimental.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #print(inputs)
                    outputs = model('sentimental', task_features, inputs1, inputs2)

                    outputs = F.softmax(outputs,dim=1)

                    loss = criterion(outputs, torch.max(sentimental.float(), 1)[1])
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                sentimental_running_loss += loss.item() * inputs1.size(0)


                sentimental_corrects += torch.sum(torch.max(outputs, 1)[1] == torch.max(sentimental, 1)[1]) 
                
                if phase == 'valid':
                    sentimental_predict.extend((1-torch.max(outputs, 1)[1].cpu().numpy()).tolist())
                    sentimental_label.extend((1-torch.max(sentimental, 1)[1].cpu().numpy()).tolist())
                    # sentimental_predict=np.concatenate((sentimental_predict, 1-torch.max(outputs, 1)[1].cpu().numpy()),axis=0)
                    # sentimental_label=np.concatenate((sentimental_label,1-torch.max(sentimental, 1)[1].cpu().numpy()),axis=0)
                # else:
                #     break
            
            if phase == 'train':
                scheduler.step()
            fakeness_epoch_loss = fakeness_running_loss / fakenews_dataset_sizes[phase]
            sentimental_epoch_loss = sentimental_running_loss / sentimental_dataset_sizes[phase]
            
            fakeness_acc = fakeness_corrects.double() / fakenews_dataset_sizes[phase]
            sentimental_acc = sentimental_corrects.double() / sentimental_dataset_sizes[phase]
            with open(log_file, "a+") as f:  # 打开文件
                f.write('{} total loss: {:.4f}, fakeness loss: {:.4f}, sentimental loss: {:.4f}'.format(phase,fakeness_epoch_loss + sentimental_epoch_loss, fakeness_epoch_loss, sentimental_epoch_loss ))
                f.write('{} fakeness_acc: {:.4f}, sentimental_acc: {:.4f}'.format(phase, fakeness_acc, sentimental_acc))
            print('{} total loss: {:.4f}, fakeness loss: {:.4f}, sentimental loss: {:.4f}'.format(phase,fakeness_epoch_loss + sentimental_epoch_loss, fakeness_epoch_loss, sentimental_epoch_loss ))
            print('{} fakeness_acc: {:.4f}, sentimental_acc: {:.4f}'.format(phase, fakeness_acc, sentimental_acc))

            # Saving training acc and loss for each epoch
            fakeness_acc1 = fakeness_acc.data
            fakeness_acc1 = fakeness_acc1.cpu()
            fakeness_acc1 = fakeness_acc1.numpy()
            
            sentimental_acc1 = sentimental_acc.data
            sentimental_acc1 = sentimental_acc1.cpu()
            sentimental_acc1 = sentimental_acc1.numpy()
            train_acc.append((fakeness_acc1, sentimental_acc1))
            #epoch_loss1 = epoch_loss.data
            #epoch_loss1 = epoch_loss1.cpu()
            #epoch_loss1 = epoch_loss1.numpy()
            train_loss.append((fakeness_epoch_loss,sentimental_epoch_loss))

            if phase == 'valid' and (fakeness_acc + sentimental_acc)/2 > best_acc:
                print('fakeness:',len(fakeness_predict), len(fakeness_label), len(fakenews_statements['valid']))
                print('sentimental:',len(sentimental_predict), len(sentimental_label), len(sentimental_datasets['valid']['statements']))
                
                df= pd.DataFrame({'predict':fakeness_predict, 'label':fakeness_label,'statement':fakenews_statements['valid']})
                #change path
                df.to_csv("./fakeness_Outputs.csv")
                
                df= pd.DataFrame({'predict':sentimental_predict, 'label':sentimental_label,'statement':sentimental_datasets['valid']['statements']})
                #change path
                df.to_csv("./sentimental_Outputs.csv") 
                
                with open(log_file, "a+") as f:  # 打开文件
                    f.write('Saving with accuracy of {} '.format((fakeness_acc + sentimental_acc)/2) + 'improved over previous {}'.format(best_acc))

                print('Saving with accuracy of {}'.format((fakeness_acc + sentimental_acc)/2),'improved over previous {}'.format(best_acc))
                
                best_acc = (fakeness_acc + sentimental_acc)/2
                
                best_acc_save = (fakeness_acc, sentimental_acc)

                # Saving val acc and loss for each epoch
                fakeness_acc1 = fakeness_acc.data
                fakeness_acc1 = fakeness_acc1.cpu()
                fakeness_acc1 = fakeness_acc1.numpy()
                
                sentimental_acc1 = sentimental_acc.data
                sentimental_acc1 = sentimental_acc1.cpu()
                sentimental_acc1 = sentimental_acc1.numpy()
                valid_acc.append((fakeness_acc1, sentimental_acc1))

                valid_loss.append((fakeness_epoch_loss,sentimental_epoch_loss))

                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'bert_model_test_noFC1_triBERT_binary_focalloss.pth')
                

        print('Time taken for epoch'+ str(epoch+1)+ ' is ' + str((time.time() - epoch_start)/60) + ' minutes')
        print()

    time_elapsed = time.time() - since
    with open(log_file, "a+") as f:  # 打开文件
        f.write('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        f.write('Best valid Acc: {:4f} fakeness: {:4f} sentimental: {:4f}'.format(float(best_acc),float(best_acc_save[0]),float(best_acc_save[1])))
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:4f} fakeness: {:4f} sentimental: {:4f}'.format(float(best_acc),float(best_acc_save[0]),float(best_acc_save[1])))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_acc, valid_acc, train_loss, valid_loss

model.to(device)


lrlast = .0001
lrmain = .0001
optim1 = torch.optim.Adam(
    [
        {"params":model.bert_gate.parameters(),"lr": lrmain},
        {"params":model.classifier0.parameters(), "lr": lrlast},
        {"params":model.classifier1.parameters(), "lr": lrlast}

   ])

#optim1 = optim.Adam(model.parameters(), lr=0.001)#,momentum=.9)
# Observe that all parameters are being optimized
optimizer_ft = optim1
criterion = nn.CrossEntropyLoss()

'''import focal_loss
loss_args = {"alpha": 0.5, "gamma": 2.0}
criterion = focal_loss.FocalLoss(*loss_args)'''

# Decay LR by a factor of 0.1 every 3 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)


model_ft1, train_acc, valid_acc, train_loss, valid_loss = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=20)