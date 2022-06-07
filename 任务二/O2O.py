import random
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.metrics import f1_score
from transformers import AdamW
from transformers import BertTokenizer, BertConfig, BertModel, BertPreTrainedModel
from torch.utils.data import TensorDataset, DataLoader

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class CommentClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(CommentClassification, self).__init__(config)
        self.num_labels = config.num_labels
        config.output_hidden_states = True
        self.bert = BertModel(config)
        self.fc = nn.Linear(config.hidden_size * 3, self.num_labels)
        self.init_weights()
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Tanh()

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        output = self.bert(input_ids=input_ids, position_ids=None,
                            token_type_ids=token_type_ids, attention_mask=attention_mask, head_mask=None)
        hidden_states = output['hidden_states']
        pooled_output = output['pooler_output']
        output = torch.cat((pooled_output, hidden_states[-1][:, 0], hidden_states[-2][:, 0]), 1)
        output = self.activation(output)
        output = self.dropout(output)
        logits = self.fc(output)

        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits.view(-1, self.num_labels), labels.view(-1))
            output = (loss, logits)
        else:
            output = (logits)

        return output


if __name__ == '__main__':
    max_len = 200
    lr = 2e-5
    epochs = 10
    dropout = 0.2
    batch_size = 16
    early_stop = 5
    nfold = 5
    model_name = 'F:/chr/model/chinese-bert-wwm-ext'
    seed = 2022

    # 设定随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    #加载预训练模型和数据
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
    train = pd.read_csv('./datasets/process_train.csv')
    test = pd.read_csv('./datasets/process_test.csv')
    test['label'] = [0] * len(test)
    train['label'] = train['label'].map(lambda x : int(x))

    #训练集数据处理
    Train_encode = tokenizer.batch_encode_plus(train['comment'].tolist(), max_length=max_len, padding='max_length', truncation=True)
    Train_input_ids = np.array(Train_encode['input_ids'])
    Train_attention_mask = np.array(Train_encode['attention_mask'])
    Train_token_type_ids = np.array(Train_encode['token_type_ids'])
    Train_labels = np.array(train['label'].tolist())

    #测试集数据处理
    Test_encode = tokenizer.batch_encode_plus(test['comment'].tolist(), max_length=max_len, padding='max_length', truncation=True)
    Test_input_ids = torch.tensor(Test_encode['input_ids'], dtype=torch.long)
    Test_attention_mask = torch.tensor(Test_encode['attention_mask'], dtype=torch.long)
    Test_token_type_ids = torch.tensor(Test_encode['token_type_ids'], dtype=torch.long)
    Test_dataset = TensorDataset(Test_input_ids, Test_attention_mask, Test_token_type_ids)

    #五折交叉训练
    SKF = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed).split(Train_labels, Train_labels)
    pred_train = np.zeros((len(train), 2), dtype=np.float32)
    pred_test = np.zeros((len(test), 2), dtype=np.float32)

    for i, (train_idx, vaild_idx) in enumerate(SKF):
        print('第{}折训练'.format(i + 1))

        #训练集
        train_dataset = TensorDataset(torch.tensor(Train_input_ids[train_idx], dtype=torch.long),
                                        torch.tensor(Train_attention_mask[train_idx], dtype=torch.long),
                                        torch.tensor(Train_token_type_ids[train_idx], dtype=torch.long),
                                        torch.tensor(Train_labels[train_idx], dtype=torch.long))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        #验证集
        vaild_dataset = TensorDataset(torch.tensor(Train_input_ids[vaild_idx], dtype=torch.long),
                                        torch.tensor(Train_attention_mask[vaild_idx], dtype=torch.long),
                                        torch.tensor(Train_token_type_ids[vaild_idx], dtype=torch.long),
                                        torch.tensor(Train_labels[vaild_idx], dtype=torch.long))
        vaild_loader = DataLoader(vaild_dataset, batch_size=batch_size, shuffle=False)

        #测试集
        test_loader = DataLoader(Test_dataset, batch_size=batch_size, shuffle=False)

        #模型
        model = CommentClassification.from_pretrained(model_name, config=BertConfig.from_pretrained(model_name, num_labels=2))
        model.cuda()
        fgm = FGM(model)

        #优化器定义
        param_optimizer = list(model.named_parameters())
        s = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = []
        t = [p for n, p in param_optimizer if not any(nd in n for nd in s)]
        optimizer_grouped_parameters.append({'params':t, 'weight_decay':0.01})
        t = [p for n, p in param_optimizer if any(nd in n for nd in s)]
        optimizer_grouped_parameters.append({'params':t, 'weight_decay':0.00})
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-6)

        best = 0.
        vaild_best = np.zeros((len(vaild_idx), 2))
        patience = 0
        for j in range(epochs):
            model.train()
            for batch in tqdm(train_loader):
                batch = tuple(t.cuda() for t in batch)
                output = model(batch[0], batch[1], batch[2], batch[3])
                loss = output[0]
                loss.backward()
                fgm.attack()
                output1 = model(batch[0], batch[1], batch[2], batch[3])
                loss_adv = output1[0]
                loss_adv.backward()
                fgm.restore()
                optimizer.step()
                optimizer.zero_grad()
                model.zero_grad()
            model.eval()
            vaild_preds_fold = np.zeros((len(vaild_idx), 2))
            with torch.no_grad():
                for i, batch in tqdm(enumerate(vaild_loader)):
                    batch = tuple(t.cuda() for t in batch)
                    outputs = model(batch[0], batch[1], batch[2], batch[3])
                    vaild_preds_fold[i * batch_size: (i + 1) * batch_size] = torch.softmax(outputs[1].detach(),dim=1).cpu().numpy()

            f1 = f1_score(Train_labels[vaild_idx], np.argmax(vaild_preds_fold, axis=1))
            if f1 > best:
                best = f1
                patience = 0
                vaild_best = vaild_preds_fold
                torch.save(model.state_dict(), './model_save/model.bin')
            else:
                patience += 1
            print('epoch:{}   F1:{}   Best_F1:{}'.format(j + 1, f1, best))
            torch.cuda.empty_cache()
            if patience >= early_stop:
                break

        test_preds_fold = np.zeros((len(test), 2))
        chkpoint = torch.load('./model_save/model.bin')
        model.load_state_dict(chkpoint)
        model.eval()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(test_loader)):
                batch = tuple(x.cuda() for x in batch)
                outputs = model(batch[0], batch[1], batch[2])
                test_preds_fold[i * batch_size: (i + 1) * batch_size] = torch.softmax(outputs.detach(), dim=1).cpu().numpy()

        pred_test += test_preds_fold / nfold

    sub = pd.read_csv('./datasets/sample.csv')
    path = model_name.split('/')
    path = path[-1].replace('-', '_')
    sub['label'] = np.argmax(pred_test, axis=1)
    sub[['id', 'label']].to_csv('./result/{}.csv'.format(path), index=None)