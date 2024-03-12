import torch
from torch import nn
import csv
from torch.nn import functional as F
import torch.utils.data as Data
import numpy as np
from d2l import torch as d2l
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

iris = loadmat('data1.mat')
data = iris['data']
X, y = data[:, 0:167], data[:, 167]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)
# y_train =y_train.reshape((6254, 1))
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class Mydataset(Dataset):
    def __init__(self,
                 mode='train', ):
        self.mode = mode

        if mode == 'test':
            mydata = X_test[:, 0:2048]
            self.mydata = torch.FloatTensor(mydata)
        elif mode == 'val':
            mydata = test
            self.mydata = torch.FloatTensor(mydata)
        else:
            y_train1 = y_train.reshape((6254, 1))
            target = y_train1[:, 0]
            mydata = X_train[:, 0:2048]

            if mode =='train':
                indices = [i for i in range(len(X_train)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(X_train)) if i % 10 == 0]
            self.mydata = torch.FloatTensor(mydata[indices])
            self.target = torch.FloatTensor(target[indices])

        self.dim = self.mydata.shape[1]

    def __getitem__(self, index):
        if self.mode in ['train', 'dev']:
            return self.mydata[index], self.target[index]
        else:
            return self.mydata[index]

    def __len__(self):
        return len(self.mydata)


def prep_dataloader(mode, batch_size, n_job=0):
    dataset = Mydataset(mode=mode)
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_job, pin_memory=True)
    return dataloader

class Dnn(nn.Module):
    def __init__(self):
        super(Dnn, self).__init__()
        self.model = nn.Sequential(
            # nn.Linear(2048, 1024),
            # nn.ReLU(),
            # nn.Linear(1024,512),
            # nn.ReLU(),
            # nn.Linear(512,128),
            # nn.ReLU(),
            # nn.Linear(128,2)

            nn.Linear(167, 48),
            nn.ReLU(),
            nn.Linear(48,24),
            nn.ReLU(),
            nn.Linear(24,2)

            # nn.Linear(334, 96),
            # nn.ReLU(),
            # nn.Linear(96,48),
            # nn.ReLU(),
            # nn.Linear(48,12),
            # nn.ReLU(),
            # nn.Linear(12,2)

            # nn.Linear(4096, 1024),
            # nn.ReLU(),
            # nn.Linear(1024,512),
            # nn.ReLU(),
            # nn.Linear(512,128),
            # nn.ReLU(),
            # nn.Linear(128,2)
        )

    def forward(self, x):
        x = self.model(x)
        return x

loss_fn = nn.CrossEntropyLoss()

def train(tr_set, dv_set,model, config, device):
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hparas'])
    n_epochs = config['n_epochs']
    total_train_acc = 0
    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}
    epoch = 0
    early_stop_cnt =0
    while epoch < n_epochs:
        model.train()
        accuracy = 0
        for x, y in tr_set:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            c = pred.argmax(1)
            accuracy += (c ==y.long()).sum().item()
            # for i in range(len(c)):
            #     if c[i] == y[i]:
            #         accuracy = accuracy + 1
            # total_train_acc = accuracy
            loss = loss_fn(pred, y.long())
            loss.backward()
            optimizer.step()
            loss_record['train'].append(loss.detach().cpu().item())
        print('total_train_acc:{}'.format(accuracy / len(tr_set.dataset)))
        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            min_mse = dev_mse
            print('total mse:{}'.format(min_mse))
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            break

    print('finish {} epochs'.format(epoch))
    return min_mse, loss_record




def dev(dv_set, model, device):
    model.eval()
    total_acc = 0
    for a, b in dv_set:
        a, b = a.to(device), b.to(device)
        with torch.no_grad():
            accuracy = 0
            pred = model(a)
            c = pred.argmax(1)
            for i in range(len(c)):
                if c[i]==y[i]:
                    accuracy =accuracy+1
            total_acc = total_acc + accuracy

    print('total  dev acc:{}'.format(total_acc/len(dv_set.dataset)))
    return total_acc

device = get_device()
print(device)
config = {
    'n_epochs': 300,
    "batch_size": 128,
    'optimizer': 'SGD',
    'optim_hparas': {
        'lr': 5e-4,
        'momentum': 0.9,
        'weight_decay': 0.02
    },
    'early_stop': 300,

}

tr_set = prep_dataloader('train', config['batch_size'])
dv_set = prep_dataloader('dev', config['batch_size'])
model = Dnn().to(device)

model_acc, model_loss = train(tr_set, dv_set, model, config, device)

a=[]
test =loadmat('Test Macc Fpr.mat')
test =test['data']
test = torch.FloatTensor(test)
test = test.to(device)
test_pred =model(test)

x=test_pred



test_pred = test_pred.argmax(1)

for i in range(len(x)):
    print(F.softmax(x[i],dim=0))
    if test_pred[i]==0:
        a.append(torch.min(F.softmax(x[i],dim=0)).cpu().detach().numpy())
    else:
         a.append(torch.max(F.softmax(x[i],dim=0)).cpu().detach().numpy())

print(a)
f = open('D:\\work\\pyGAT-master\\data\\note\\DNN test.csv', 'a', encoding='utf-8', newline='')
csv_write = csv.writer(f)
csv_write.writerow(
        [float(a[j]) for j in range(len(a))])
f.close()