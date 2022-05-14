import time
import torch
import torch.optim as optim
import torch.nn.functional as func


def train_model(model, device, train_dataset, test_dataset, epoch):
    # define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    f1_seq, training_loss_seq = [], []
    for _ in range(epoch):
        model.train()
        st = time.time()
        tot_loss = torch.tensor(0, device=device, dtype=torch.float32)
        for idx, (x, y) in enumerate(train_dataset):
            y_pred = model(x)
            loss = func.cross_entropy(y_pred.permute(0, 2, 1), y)
            # print('loss =', loss)
            tot_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch{0:} :'.format(_))
        training_loss_seq.append(tot_loss.item() / len(train_dataset))
        print('avg_training_loss = ', training_loss_seq[-1])
        print('used time = {0:}'.format(time.time() - st))
        # evaluate model
        model.eval()
        tot_cnt, acc_cnt, true_cnt, pred_cnt, true_pred_cnt = 0, 0, 0, 0, 0
        for idx, (x, y) in enumerate(test_dataset):
            y_pred = model(x)
            y_pred = torch.max(y_pred, dim=2)[1]
            tot_cnt += len(y_pred[0])
            for i in range(len(y_pred[0])):
                if y_pred[0][i] == y[0][i]:
                    acc_cnt += 1
                if y_pred[0][i] != 0:
                    pred_cnt += 1
                    if y_pred[0][i] == y[0][i]:
                        true_pred_cnt += 1
                if y[0][i] != 0:
                    true_cnt += 1
            # print(y_pred, y)
        print('tot_cnt={0:} acc_cnt={1:} true_cnt={2:} pred_cnt={3:} true_pred_cnt={4:}'
              .format(tot_cnt, acc_cnt, true_cnt, pred_cnt, true_pred_cnt))
        eps = 1e-8
        acc = acc_cnt / tot_cnt
        pre = true_pred_cnt / (pred_cnt + eps)
        rec = true_pred_cnt / (true_cnt + eps)
        f1 = 2 * pre * rec / (pre + rec + eps)
        f1_seq.append(f1)
        print('acc.={0:.6f} pre.={1:.6f} rec.={2:.6f} f1={3:.6f}'.format(acc, pre, rec, f1))
        print('----------------------------------------------------------------------------------------')
    return training_loss_seq, f1_seq
