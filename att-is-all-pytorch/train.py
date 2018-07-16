import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import datetime
import constant
from transformer.model import Transformer
import torch.optim as optim
import argparse
from DataLoader import DataLoader

def print_data_length(train,valid,test):
    train_src = train['src']
    train_tgt = train['tgt']
    valid_src = valid['src']
    valid_tgt = valid['tgt']
    test_src = test['src']
    test_tgt = test['tgt']

    len_train_src_lst = [len(si) for si in train_src]
    len_train_tgt_lst = [len(si) for si in train_tgt]
    len_valid_src_lst = [len(si) for si in valid_src]
    len_valid_tgt_lst = [len(si) for si in valid_tgt]
    len_test_src_lst = [len(si) for si in test_src]
    len_test_tgt_lst = [len(si) for si in test_tgt]
    print('train:\n[src] max:{}, min:{}, avg:{}\n[tgt] max:{}, min:{}, avg:{}\n'.format(
        max(len_train_src_lst), min(len_train_src_lst), sum(len_train_src_lst) / len(len_train_src_lst),
        max(len_train_tgt_lst), min(len_train_tgt_lst), sum(len_train_tgt_lst) / len(len_train_tgt_lst)))
    print('valid:\n[src] max:{}, min:{}, avg:{}\n[tgt] max:{}, min:{}, avg:{}\n'.format(
        max(len_valid_src_lst), min(len_valid_src_lst), sum(len_valid_src_lst) / len(len_valid_src_lst),
        max(len_valid_tgt_lst), min(len_valid_tgt_lst), sum(len_valid_tgt_lst) / len(len_valid_tgt_lst)))
    print('test:\n[src] max:{}, min:{}, avg:{}\n[tgt] max:{}, min:{}, avg:{}\n'.format(
        max(len_test_src_lst), min(len_test_src_lst), sum(len_test_src_lst) / len(len_test_src_lst),
        max(len_test_tgt_lst), min(len_test_tgt_lst), sum(len_test_tgt_lst) / len(len_test_tgt_lst)))

def pad_loss(n_tgt_classes):
    tgt_mask = torch.ones(n_tgt_classes)
    tgt_mask[constant._PAD_] = 0
    return nn.CrossEntropyLoss(weight=tgt_mask,size_average=False)

def pad_acc(batch_predict,batch_groudtruth):
    batch_predict = batch_predict.view(-1)
    batch_groudtruth = batch_groudtruth.view(-1)
    mask = batch_groudtruth.ne(constant._PAD_).type(torch.IntTensor)
    right = sum((batch_predict == batch_groudtruth).type(torch.IntTensor) * mask)
    total = sum(mask)
    return right.type(torch.FloatTensor),total.type(torch.FloatTensor)

def train_step(model,dataloader,pad_loss,optimizer):
    # model.train()
    loss = 0
    n_correct = 0
    n_total = 0
    while not dataloader.stop:
        optimizer.zero_grad()
        batch_src, batch_tgt = dataloader.next()
        batch_src_idx = batch_src[0].view(-1,opt.max_seq_len).type(torch.FloatTensor)
        batch_src_positi = batch_src[1].view(-1,opt.max_seq_len).type(torch.LongTensor)
        batch_src = (batch_src_idx,batch_src_positi)

        batch_tgt_idx = batch_tgt[0].view(-1,opt.max_seq_len).type(torch.LongTensor)
        batch_tgt_positi = batch_tgt[1].view(-1, opt.max_seq_len).type(torch.LongTensor)
        batch_tgt = (batch_tgt_idx,batch_tgt_positi)

        batch_predict_logits = model(batch_src,batch_tgt) #[batch * max_length * n_tgt_tokens]
        _,batch_predict_labels = torch.max(batch_predict_logits,2)
        batch_predict_logits = torch.transpose(batch_predict_logits,1,2)
        # batch_predict_labels = batch_predict_labels.type(torch.LongTensor)
        batch_loss = pad_loss(batch_predict_logits,batch_tgt_idx)
        batch_n_correct,batch_n_total = pad_acc(batch_predict_labels,batch_tgt_idx)

        batch_loss.backward()
        optimizer.step()

        loss += batch_loss
        n_correct += batch_n_correct
        n_total += batch_n_total
        if dataloader.i_batch % 100 == 0:
            print(' batch idx: {},loss: {loss:.3f},ratio: {ratio:.3f}'.format(dataloader.i_batch,loss=batch_loss.item(),ratio=batch_n_correct.item()/batch_n_total.item()))

    loss /= dataloader.end_idx
    acc = n_correct / n_total
    if dataloader.stop:
        dataloader.i_batch = 0
        dataloader.stop = False

    return loss,acc

def valid_step(model,dataloader,pad_loss):
    # model.eval()
    loss = 0
    n_correct = 0
    n_total = 0
    while not dataloader.stop:
        batch_src, batch_tgt = dataloader.next()
        batch_src_idx = batch_src[0].view(-1, opt.max_seq_len).type(torch.FloatTensor)
        batch_src_positi = batch_src[1].view(-1, opt.max_seq_len).type(torch.LongTensor)
        batch_src = (batch_src_idx, batch_src_positi)

        batch_tgt_idx = batch_tgt[0].view(-1, opt.max_seq_len).type(torch.LongTensor)
        batch_tgt_positi = batch_tgt[1].view(-1, opt.max_seq_len).type(torch.LongTensor)
        batch_tgt = (batch_tgt_idx, batch_tgt_positi)


        batch_predict_logits = model(batch_src,batch_tgt)
        _, batch_predict_labels = torch.max(batch_predict_logits,2)

        batch_predict_logits = batch_predict_logits.view(-1, dataloader.n_tgt_tokens, opt.max_seq_len)
        batch_loss = pad_loss(batch_predict_logits, batch_tgt_idx)
        batch_n_correct, batch_n_total = pad_acc(batch_predict_labels, batch_tgt_idx)

        loss += batch_loss
        n_correct += batch_n_correct
        n_total += batch_n_total

    loss /= dataloader.end_idx
    acc = n_correct / n_total

    if dataloader.stop:
        dataloader.i_batch = 0
        dataloader.stop = False

    return loss, acc

def plot_curve(acc,loss):
    acc_train = [a[0] for a in acc]
    acc_valid = [a[1] for a in acc]
    acc_test = [a[2] for a in acc]
    plt.subplot(211)
    plt.plot(acc_train, label='train')
    plt.plot(acc_valid,label="valid")
    plt.plot(acc_test, label="test")
    plt.legend(loc='best', shadow=True)
    plt.ylabel('accuracy')
    plt.xlabel('epochs')

    loss_train = [a[0] for a in loss]
    loss_valid = [a[1] for a in loss]
    loss_test = [a[2] for a in loss]
    plt.subplot(212)
    plt.plot(loss_train, label='train')
    plt.plot(loss_valid,label="valid")
    plt.plot(loss_test, label="test")
    plt.legend(loc='best', shadow=True)
    plt.ylabel('loss')
    plt.xlabel('epochs')

    nowTime = datetime.datetime.now().strftime('%Y%m%d-%H_%M_%S')
    plt.savefig(str(nowTime))
    plt.show()

def train(model,train_dataloder,valid_dataloader,test_dataloader,pad_loss,optimizer):
    if opt.use_trained_model:
        if os.path.exists(opt.save_model_dir):
            models_files = os.listdir(opt.save_model_dir)
            if (len(models_files) != 0):
                valid_acc_lst = [float(s[7:].split('.chkpt')[0]) for s in models_files]
                max_valid_acc_idx = valid_acc_lst.index(max(valid_acc_lst))
                max_valid_model_path = opt.save_model_dir + models_files[max_valid_acc_idx]
                model.load_state_dict(torch.load(max_valid_model_path))
                print(max_valid_model_path)

    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(opt.log_file,"a",encoding='utf-8') as f:
        f.write('\n' + '*'*10 + ' '+ nowTime + ' '+ '*'*10 + '\n')
        f.write("loss | acc\n")

    max_valid_acc = -1
    # this plot calculate acc and loss on valid and test dataset every epochs.
    plt_acc = []
    plt_loss = []

    for i in range(opt.max_epochs):
        print('>>>>>>epochs: {}'.format(i))
        with open(opt.log_file, "a", encoding='utf-8') as f:
            f.write('>>>>>>epochs: {}, '.format(i))

        train_loss,train_acc = train_step(model,train_dataloder,pad_loss,optimizer)
        print('loss:{loss:.5f}\nacc:{acc:.3f}\n'.format(loss=train_loss.item(),acc=train_acc.item() * 100))
        with open(opt.log_file, "a", encoding='utf-8') as f:
            f.write('>{loss: .5f} | {acc: .3f}<\n'.format(loss=train_loss.item(),acc=train_acc.item() * 100))

        if i % opt.epochs_per_valid == 0:
            valid_loss, valid_acc = valid_step(model, valid_dataloader,pad_loss)
            print('>【Valid】\nloss: {loss:.5f}\nacc: {acc:.3f}'.format(loss=valid_loss.item(), acc=valid_acc.item() * 100))
            print('-' * 25)

            with open(opt.log_file, "a", encoding='utf-8') as f:
                f.write('>【Valid】\n{loss: .5f} | {acc: .3f}\n'.format(loss=valid_loss.item(), acc=valid_acc.item() * 100))
                # f.write('-' * 35 + '\n')

            # Test
            test_loss, test_acc = valid_step(model, test_dataloader, pad_loss)
            print('>【Test】\nloss: {loss:.5f}\nacc: {acc:.3f}'.format(loss=test_loss.item(), acc=test_acc.item() * 100))
            print('-' * 25)
            with open(opt.log_file, "a", encoding='utf-8') as f:
                f.write('>【Test】\n{loss: .5f} | {acc: .3f}\n'.format(loss=test_loss.item(), acc=test_acc.item() * 100))
                f.write('-' * 35 + '\n')
            # lists needed to plot the picture.
            plt_acc.append((train_acc,valid_acc,test_acc))
            plt_loss.append((train_loss,valid_loss,test_loss))

            # save modle here.
            if opt.save_model == True:
                if max_valid_acc < valid_acc:
                    max_valid_acc = valid_acc
                if not os.path.exists(opt.save_model_dir):
                    os.makedirs(opt.save_model_dir)
                saved_model_path = opt.save_model_dir + 'v_accu_{accu:3.3f}.chkpt'.format(accu = max_valid_acc * 100)
                torch.save(model.state_dict(),saved_model_path)
    # plot.
    plot_curve(plt_acc,plt_loss)

def test(model,test_dataloader,pad_loss):
    '''
    use the model which has the highest acc in valid data,
    :param model: the reloaded model.
    :param test_dataloader:test data.
    :param pad_loss: cross entropy loss function.
    :return: nothing.
    '''

    if os.path.exists(opt.save_model_dir):
        models_files = os.listdir(opt.save_model_dir)
        if (len(models_files) != 0):
            valid_acc_lst = [float(s[7:].split('.chkpt')[0]) for s in models_files]
            max_valid_acc = max(valid_acc_lst)
            max_valid_model_path = opt.save_model_dir + 'v_accu_' + str(max_valid_acc) + '.chkpt'
            model.load_state_dict(torch.load(max_valid_model_path))

        else:
            print('trained model doesn\'t exist!')
            exit()
    else:
        print('trained model doesn\'t exist!')
        exit()

    loss, acc = valid_step(model, test_dataloader, pad_loss)
    print('>【Test】\nloss: {loss:.3f}\nacc: {acc:.3f}'.format(loss=loss.item(), acc=acc.item() * 100))
    print('-' * 25)

parser = argparse.ArgumentParser()
parser.add_argument('-data',default= 'data/multi30k.atok.low.pt')
parser.add_argument('-batch_size',type=int,default=5)
parser.add_argument('-max_seq_len',type=int, default=35)
parser.add_argument('-max_epochs',type = int,default=20)
parser.add_argument('-epochs_per_valid',type= int,default=1)
parser.add_argument('-n_hidden_state',type = int,default=8)
parser.add_argument('-log_file',type=str,default='log.txt')
parser.add_argument('-save_model',type=bool,default=False)
parser.add_argument('-use_trained_model',type=bool,default=False)
parser.add_argument('-save_model_dir',type=str,default='saved_model/')
opt = parser.parse_args()

data = torch.load("data/multi30k.atok.low.pt")
src_w2i_dict = data['dict']['src']
tgt_w2i_dict = data['dict']['tgt']
src_i2w_dict = dict([(i,w) for w,i in src_w2i_dict.items()])
tgt_i2w_dict = dict([(i,w) for w,i in tgt_w2i_dict.items()])
n_src_tokens = len(src_w2i_dict)
n_tgt_tokens = len(tgt_w2i_dict)

train_src = data['train']['src']
train_tgt = data['train']['tgt']
valid_src = data['valid']['src']
valid_tgt = data['valid']['tgt']

test_data = torch.load('data/multi30k.test.atok.low.pt')
test_src = test_data['test']['src']
test_tgt = test_data['test']['tgt']

# print data length.
# print_data_length(data['train'],data['valid'],test_data['test'])

n_train_samples = len(train_src)
n_valid_samples = len(valid_src)
n_test_samples =  len(test_src)

sml_train_src = train_src[:200]
sml_train_tgt = train_tgt[:200]
sml_valid_src = valid_src[:50]
sml_valid_tgt = valid_tgt[:50]
sml_test_src = test_src[:50]
sml_test_tgt = test_tgt[:50]
n_sml_train_samples = len(sml_train_src)
n_sml_valid_samples = len(sml_valid_src)
n_sml_test_samples = len(sml_test_src)

# train_data_loader = DataLoader(train_src,train_tgt,opt.batch_size,opt.max_seq_len,n_train_samples,n_src_tokens,n_tgt_tokens,is_test = False)
# valid_data_loader = DataLoader(valid_src,valid_tgt,opt.batch_size,opt.max_seq_len,n_valid_samples,n_src_tokens,n_tgt_tokens,is_test = False)
# test_data_loader = DataLoader(test_src,test_tgt,opt.batch_size,opt.max_seq_len,n_test_samples,n_src_tokens,n_tgt_tokens,is_test = True)

train_data_loader = DataLoader(sml_train_src,sml_train_tgt,opt.batch_size,opt.max_seq_len,n_sml_train_samples,n_src_tokens,n_tgt_tokens)
valid_data_loader = DataLoader(sml_valid_src,sml_valid_tgt,opt.batch_size,opt.max_seq_len,n_sml_valid_samples,n_src_tokens,n_tgt_tokens)
test_data_loader = DataLoader(sml_test_src,sml_test_tgt,opt.batch_size,opt.max_seq_len,n_sml_test_samples,n_src_tokens,n_tgt_tokens)

model = Transformer(n_layers=3,n_head=8,d_hidden=8,dv=16,dk=16,dq=16,n_src_tokens=n_src_tokens,n_tgt_tokens=n_tgt_tokens)
optimizer = optim.SGD(model.get_trainable_parameters(),lr = 0.005,momentum=0.9)
cross_entro = pad_loss(n_tgt_tokens)

train(model,train_data_loader,valid_data_loader,test_data_loader,cross_entro,optimizer)

# test(model,test_data_loader,cross_entro)

