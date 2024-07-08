import numpy as np
import awkward as ak
import tqdm
import time
import torch

from collections import defaultdict, Counter
from .metrics import evaluate_metrics
from ..data.tools import _concat
from ..logger import _logger

def _flatten_label(label, mask=None):
    if label.ndim > 1:
        label = label.view(-1)
        if mask is not None:
            label = label[mask.view(-1)]
    # print('label', label.shape, label)
    return label


def _flatten_preds(preds, mask=None, label_axis=1):
    if preds.ndim > 2:
        # assuming axis=1 corresponds to the classes
        preds = preds.transpose(label_axis, -1).contiguous()
        preds = preds.view((-1, preds.shape[-1]))
        if mask is not None:
            preds = preds[mask.view(-1)]
    # print('preds', preds.shape, preds)
    return preds


def train_classification_domain(model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None, tb_helper=None):
    model.train()

    data_config = train_loader.dataset.config

    num_batches, total_loss, total_cat_loss, total_domain_loss, count_cat, count_domain = 0, 0, 0, 0, 0, 0
    label_cat_counter = Counter()
    total_cat_correct, total_domain_correct = 0, 0
    inputs, label_cat, label_domain, model_output, model_output_cat, model_output_domain = None, None, None, None, None, None
    loss, loss_cat, loss_domain, pred_cat, pred_domain, correct_cat, correct_domain = None, None, None, None, None, None, None


    ### number of classification labels
    num_labels = len(data_config.label_value)
    ### number of domain regions
    num_domains = len(data_config.label_domain_names)
    ### total number of domain labels
    if type(data_config.label_domain_value) == dict:
        num_labels_domain = sum(len(dct) if type(dct) == list else 1 for dct in data_config.label_domain_value.values())
    else:
        num_labels_domain = len(data_config.label_domain_value)
    ### number of labels per region as a list
    if type(data_config.label_domain_value) == dict:
        ldomain = [len(dct) if type(dct) == list else 1 for dct in data_config.label_domain_value.values()]
    else:
        ldomain = [len(data_config.label_domain_value)]
    ### label domain counter
    label_domain_counter = []
    for idx, names in enumerate(data_config.label_domain_names):
        label_domain_counter.append(Counter())


    start_time = time.time()




    with tqdm.tqdm(train_loader) as tq:
        # for X, y, _ in tq:
        for X, y_cat, y_domain, _, y_cat_check, y_domain_check in tq:
            
            # inputs = [X[k].to(dev) for k in data_config.input_names]
            # label = y[data_config.label_names[0]].long()
            # try:
            #     label_mask = y[data_config.label_names[0] + '_mask'].bool()
            # except KeyError:
            #     label_mask = None
            # label = _flatten_label(label, label_mask)
            # num_examples = label.shape[0]
            # label_counter.update(label.cpu().numpy())
            # label = label.to(dev)

            ### input features for the model
            inputs = [X[k].to(dev,non_blocking=True) for k in data_config.input_names]

            ### build classification true labels (numpy argmax)
            label_cat  = y_cat[data_config.label_names[0]].long()
            cat_check  = y_cat_check[data_config.labelcheck_names[0]].long()
            index_cat  = cat_check.nonzero()
            label_cat  = label_cat[index_cat]

            # print ("label_cat",label_cat)  
            # print ("label_cat",label_cat.nelement())  

            # print ("y_domain",y_domain)  
            # print ("y_domain",y_domain.nelement())  
            # print ("y_domain_check",y_domain_check)  
            # print ("y_domain_check",y_domain_check.nelement())  

            ### build domain true labels (numpy argmax)
            for idx, (k, v) in enumerate(y_domain.items()):
                if idx == 0:
                    label_domain = v.long()
                else:
                    label_domain = torch.column_stack((label_domain,v.long()))
                
            ### store indexes to separate classification events from DA
            for idx, (k, v) in enumerate(y_domain_check.items()):
                if idx == 0:
                    label_domain_check = v.long()
                    index_domain_all = v.long().nonzero()
                else:
                    label_domain_check = torch.column_stack((label_domain_check,v.long()))
                    index_domain_all = torch.cat((index_domain_all,v.long().nonzero()),0)

            ### edit labels
            label_domain = label_domain[index_domain_all]
            label_domain_check = label_domain_check[index_domain_all]
            label_cat = _flatten_label(label_cat,None)
            # print(label_domain)  
            # print ("label_domain",label_domain.nelement())
            # print(label_domain_check)  
            # print ("label_domain_check",label_domain_check.nelement())
            if label_domain.nelement() == 0  or label_domain.nelement()> 1:
                label_domain = label_domain.squeeze()
            else:
                label_domain = label_domain[0]
            label_domain_check = label_domain_check.squeeze()            

            # print(label_domain)  s


            ### Number of samples in the batch
            num_cat_examples = label_cat.shape[0]
            num_domain_examples = label_domain.shape[0]

            label_cat_np = label_cat.cpu().numpy().astype(dtype=np.int32)
            if np.iterable(label_cat_np):
                label_cat_counter.update(label_cat_np)
            else:
                _logger.info('label_cat not iterable --> shape %s'%(str(label_cat_np.shape)))
                
            index_domain = defaultdict(list)
            for idx, (k,v) in enumerate(y_domain_check.items()):
                if num_domains == 1:
                    index_domain[k] = label_domain_check.nonzero();
                    label_domain_np = label_domain[index_domain[k]].squeeze().cpu().numpy().astype(dtype=np.int32)
                    if np.iterable(label_domain_np):
                        label_domain_counter[idx].update(label_domain_np)
                    else:
                        _logger.info('label_domain not iterable --> shape %s'%(str(label_domain_np.shape)))
                else:                    
                    index_domain[k] = label_domain_check[:,idx].nonzero();
                    label_domain_np = label_domain[index_domain[k],idx].squeeze().cpu().numpy().astype(dtype=np.int32);
                    if np.iterable(label_domain_np):
                        label_domain_counter[idx].update(label_domain_np)
                    else:
                        _logger.info('label_domain %d not iterable --> shape %s'%(idx,str(label_domain_np.shape)))

            ### send to GPU
            label_cat = label_cat.to(dev,non_blocking=True)
            label_domain = label_domain.to(dev,non_blocking=True)
            label_domain_check = label_domain_check.to(dev,non_blocking=True)



            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                model_output = model(*inputs)
                model_output_cat = model_output[:,:num_labels]
                model_output_domain = model_output[:,num_labels:num_labels+num_labels_domain]
                model_output_cat = _flatten_preds(model_output_cat,None)
                model_output_cat = model_output_cat[index_cat].squeeze().float()
                model_output_domain = model_output_domain[index_domain_all].squeeze().float()
                label_cat = label_cat.squeeze()
                label_domain = label_domain.squeeze()
                label_domain_check = label_domain_check.squeeze()
                # logits = _flatten_preds(model_output, label_mask)
                # loss = loss_func(logits, label)
                loss, loss_cat, loss_domain = loss_func(model_output_cat, label_cat, model_output_domain, label_domain, label_domain_check)

            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', False):
                scheduler.step()

            # _, preds = logits.max(1)
            # loss = loss.item()
            # num_batches += 1
            # count += num_examples
            # correct = (preds == label).sum().item()
            # total_loss += loss
            # total_correct += correct

            ### evaluate loss function and counters
            num_batches += 1
            loss = loss.detach().item()
            total_loss += loss
            if loss_cat:
                loss_cat = loss_cat.detach().item()
                total_cat_loss += loss_cat
            if loss_domain:
                loss_domain = loss_domain.detach().item()
                total_domain_loss += loss_domain

            ## take the classification prediction and compare with the true labels            
            label_cat = label_cat.detach()
            label_domain = label_domain.detach()
            model_output_cat = model_output_cat.detach()
            model_output_domain = model_output_domain.detach()
            
            if torch.is_tensor(label_cat) and torch.is_tensor(model_output_cat) and np.iterable(label_cat) and np.iterable(model_output_cat):
                _, pred_cat = model_output_cat.max(1)
                if pred_cat.shape == label_cat.shape:
                    correct_cat = (pred_cat == label_cat).sum().item()
                    total_cat_correct += correct_cat
                    count_cat += num_cat_examples

            ## single domain region
            if num_domains == 1:
                if torch.is_tensor(label_domain) and torch.is_tensor(model_output_domain) and np.iterable(label_domain) and np.iterable(model_output_domain):
                    _, pred_domain = model_output_domain.max(1)
                    if pred_domain.shape == label_domain.shape:
                        correct_domain = (pred_domain == label_domain).sum().item()
                        total_domain_correct += correct_domain
                        count_domain += num_domain_examples
            ## multiple domain regions
            else:
                correct_domain = 0
                for idx, (k,v) in enumerate(y_domain_check.items()):                    
                    id_dom = idx*ldomain[idx]
                    label  = label_domain[index_domain[k],idx].squeeze()
                    if not torch.is_tensor(label) or not np.iterable(label): continue
                    pred_domain = model_output_domain[index_domain[k],id_dom:id_dom+ldomain[idx]].squeeze()
                    if not torch.is_tensor(pred_domain) or not np.iterable(pred_domain): continue
                    _, pred_domain = pred_domain.max(1)
                    if pred_domain.shape != label.shape: continue
                    correct_domain += (pred_domain == label).sum().item()
                total_domain_correct += correct_domain
                count_domain += num_domain_examples

            # print (num_domain_examples)
            # print ("correct_domain, num_domain_examples", correct_domain, num_domain_examples)
            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Loss': '%.5f' % loss,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'Acc': '%.5f' % (correct_cat / num_cat_examples if num_cat_examples else 0),
                'AvgAcc': '%.5f' % (total_cat_correct / count_cat if count_cat else 0),
                'AccDomain': '%.5f' % (correct_domain / (num_domain_examples) if (num_domain_examples and correct_domain) else 0),
                'AvgAccDomain': '%.5f' % (total_domain_correct / (count_domain) if (count_domain and total_domain_correct) else 0),
                # 'AvgAccDomain': '%.5f' % (total_domain_correct / (count_domain) if count_domain else 0),
                })

            if tb_helper:
                tb_helper.write_scalars([
                    ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                    ("Acc/train", correct_cat / num_cat_examples if num_cat_examples else 0, tb_helper.batch_train_count + num_batches),
                    ("AccDomain/train", correct_domain / (num_domain_examples) if (num_domain_examples and correct_domain) else 0, tb_helper.batch_train_count + num_batches),
                    ])
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches, mode='train')

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count_cat+count_domain, (count_cat+count_domain) / time_diff))
    _logger.info('Train AvgLoss: %.5f'% (total_loss / num_batches))
    _logger.info('Train AvgLoss Cat: %.5f'% (total_cat_loss / num_batches))
    _logger.info('Train AvgLoss Domain: %.5f'% (total_domain_loss / num_batches))
    _logger.info('Train AvgAccCat: %.5f'%(total_cat_correct / count_cat if count_cat else 0))
    _logger.info('Train AvgAccDomain: %.5f'%(total_domain_correct / (count_domain) if count_domain else 0))
    _logger.info('Train class distribution: \n %s', str(sorted(label_cat_counter.items())))
    _logger.info('Train domain distribution: \n %s', ' '.join([str(sorted(i.items())) for i in label_domain_counter]))


    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("Loss Cat/train (epoch)", total_cat_loss / num_batches, epoch),
            ("Loss Domain/train (epoch)", total_domain_loss / num_batches, epoch),
            ("AccCat/train (epoch)", total_cat_correct / count_cat if count_cat else 0, epoch),
            ("AccDomain/train (epoch)", total_domain_correct / (count_domain) if (count_domain and total_domain_correct) else 0, epoch),
            ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')
        # update the batch state
        tb_helper.batch_train_count += num_batches

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()


def evaluate_classification_domain(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None,
                            eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                            eval_cat_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                            tb_helper=None):
    model.eval()

    data_config = test_loader.dataset.config

    # label_counter = Counter()
    # total_loss = 0
    # num_batches = 0
    # total_correct = 0
    # entry_count = 0
    # count = 0
    # scores = []
    # labels = defaultdict(list)
    # labels_counts = []
    # labels_domain = defaultdict(list)
    # observers = defaultdict(list)


    label_cat_counter = Counter()
    total_loss, total_cat_loss, total_domain_loss, num_batches, total_cat_correct, total_domain_correct = 0, 0, 0, 0, 0, 0
    count_cat, count_domain = 0, 0
    inputs, label_cat, label_domain, model_output, model_output_cat, model_output_domain  = None, None, None, None, None , None
    pred_cat, pred_domain, correct_cat, correct_domain = None, None, None, None
    loss, loss_cat, loss_domain = None, None, None
    scores_cat, indexes_cat = [], []
    scores_domain  = defaultdict(list)
    labels_cat, labels_domain, observers = defaultdict(list), defaultdict(list), defaultdict(list)
    indexes_domain = defaultdict(list)
    index_offset = 0

    ### number of classification labels
    num_labels = len(data_config.label_value)
    ### total number of domain regions
    num_domains = len(data_config.label_domain_names)
    ### total number of domain labels
    if type(data_config.label_domain_value) == dict:
        num_labels_domain = sum(len(dct) if type(dct) == list else 1 for dct in data_config.label_domain_value.values())
    else:
        num_labels_domain = len(data_config.label_domain_value)

    ### number of labels per region as a list
    if type(data_config.label_domain_value) == dict:
        ldomain = [len(dct) if type(dct) == list else 1 for dct in data_config.label_domain_value.values()]
    else:
        ldomain = [len(data_config.label_domain_value)]
    ### label counter
    label_domain_counter = []
    for idx, names in enumerate(data_config.label_domain_names):
        label_domain_counter.append(Counter())

    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            # for X, y, Z in tq:
            for X, y_cat, y_domain, Z, y_cat_check, y_domain_check in tq:

                inputs = [X[k].to(dev) for k in data_config.input_names]

                # label = y[data_config.label_names[0]].long()
                # entry_count += label.shape[0]
                # try:
                #     label_mask = y[data_config.label_names[0] + '_mask'].bool()
                # except KeyError:
                #     label_mask = None
                # if not for_training and label_mask is not None:
                #     labels_counts.append(np.squeeze(label_mask.numpy().sum(axis=-1)))
                # label = _flatten_label(label, label_mask)
                # num_examples = label.shape[0]
                # label_counter.update(label.cpu().numpy())
                # label = label.to(dev)
                # model_output = model(*inputs)
                # logits = _flatten_preds(model_output, label_mask).float()

                # scores.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
                # for k, v in y.items():
                #     labels[k].append(_flatten_label(v, label_mask).cpu().numpy())
                # if not for_training:
                #     for k, v in Z.items():
                #         observers[k].append(v.cpu().numpy())

                # _, preds = logits.max(1)
                # loss = 0 if loss_func is None else loss_func(logits, label).item()

                # num_batches += 1
                # count += num_examples
                # correct = (preds == label).sum().item()
                # total_loss += loss * num_examples
                # total_correct += correct

                ### build classification true labels
                label_cat = y_cat[data_config.label_names[0]].long()
                cat_check = y_cat_check[data_config.labelcheck_names[0]].long()
                index_cat = cat_check.nonzero()
                label_cat = label_cat[index_cat]

                ### build domain true labels (numpy argmax)                                                                                                                                   
                for idx, (k,v) in enumerate(y_domain.items()):
                    if idx == 0:
                        label_domain = v.long()
                    else:
                        label_domain = torch.column_stack((label_domain,v.long()))

                for idx, (k, v) in enumerate(y_domain_check.items()):
                    if idx == 0:
                        label_domain_check = v.long()
                        index_domain_all = v.long().nonzero()
                    else:
                        label_domain_check = torch.column_stack((label_domain_check,v.long()))
                        index_domain_all = torch.cat((index_domain_all,v.long().nonzero()),0)

                label_domain = label_domain[index_domain_all]
                label_domain_check = label_domain_check[index_domain_all]


                # print(label_domain)  
                # print ("label_domain",label_domain.nelement())
                # print(label_domain_check)  
                # print ("label_domain_check",label_domain_check.nelement())

                ### edit labels                                                                                                                                                                      
                label_cat = _flatten_label(label_cat,None)
                # label_domain = label_domain.squeeze()
                if label_domain.nelement() == 0  or label_domain.nelement()> 1:
                    label_domain = label_domain.squeeze()
                    label_domain_check = label_domain_check.squeeze()
                else:
                    label_domain = label_domain[0]
                    label_domain_check = label_domain_check[0]
                # label_domain_check = label_domain_check.squeeze()

                ### counters
                label_cat_np = label_cat.cpu().numpy().astype(dtype=np.int32)
                if np.iterable(label_cat_np):
                    label_cat_counter.update(label_cat_np)
                else:
                    _logger.info('label_cat not iterable --> shape %s'%(str(label_cat_np.shape)))

                # print (label_domain_check)
                # if label_domain_check.nelement()

                index_domain = defaultdict(list)
                for idx, (k,v) in enumerate(y_domain_check.items()):
                    if num_domains == 1:
                        index_domain[k] = label_domain_check.nonzero()
                        label_domain_np = label_domain[index_domain[k]].squeeze().cpu().numpy().astype(dtype=np.int32)
                        if np.iterable(label_domain_np):
                            label_domain_counter[idx].update(label_domain_np)
                        else:
                            _logger.info('label_domain not iterable --> shape %s'%(str(label_domain_np.shape)))
                    else:
                        index_domain[k] = label_domain_check[:,idx].nonzero()
                        label_domain_np = label_domain[index_domain[k],idx].squeeze().cpu().numpy().astype(dtype=np.int32)
                        if np.iterable(label_domain_np):
                            label_domain_counter[idx].update(label_domain_np)
                        else:
                            _logger.info('label_domain %d not iterable --> shape %s'%(idx,str(label_domain_np.shape)))

                ### update counters
                num_cat_examples = label_cat.shape[0]
                num_domain_examples = label_domain.shape[0]

                ### send to gpu
                label_cat = label_cat.to(dev,non_blocking=True)
                label_domain = label_domain.to(dev,non_blocking=True)
                label_domain_check = label_domain_check.to(dev,non_blocking=True)

                ### store truth labels for classification and regression as well as observers
                for k, v in y_cat.items():
                    if not for_training:
                        labels_cat[k].append(_flatten_label(v,None).cpu().numpy().astype(dtype=np.int32))
                    else:
                        labels_cat[k].append(_flatten_label(v[index_cat],None).cpu().numpy().astype(dtype=np.int32))
                    
                if not for_training:
                    indexes_cat.append((index_offset+index_cat).cpu().numpy().astype(dtype=np.int32))
                    for k, v in Z.items():                
                        if v.cpu().numpy().dtype in (np.int16, np.int32, np.int64):
                            observers[k].append(v.cpu().numpy().astype(dtype=np.int32))
                        else:
                            observers[k].append(v.cpu().numpy().astype(dtype=np.float32))

                for idx, (k, v) in enumerate(y_domain.items()):
                    # print("idx, (k, v)", idx, (k, v))
                    if not for_training:
                        # print ("v", v)
                        # print ("v.squeeze()", v.squeeze())
                        labels_domain[k].append(v.squeeze().cpu().numpy().astype(dtype=np.int32))
                        # print ("index_offset+index_domain[list(y_domain_check.keys())[idx]]", index_offset+index_domain[list(y_domain_check.keys())[idx]])
                        indexes_domain[k].append((index_offset+index_domain[list(y_domain_check.keys())[idx]]).cpu().numpy().astype(dtype=np.int32))
                    else:
                        # print ("v[index_domain[list(y_domain_check.keys())[idx]]]", v[index_domain[list(y_domain_check.keys())[idx]]])
                        # print ("v[index_domain[list(y_domain_check.keys())[idx]]].squeeze()", v[index_domain[list(y_domain_check.keys())[idx]]].squeeze())
                        if v[index_domain[list(y_domain_check.keys())[idx]]].nelement() == 0 or v[index_domain[list(y_domain_check.keys())[idx]]].nelement()>1:
                            toUse_ = v[index_domain[list(y_domain_check.keys())[idx]]].squeeze()
                        else:
                            toUse_ = v[index_domain[list(y_domain_check.keys())[idx]]][0]
                        # labels_domain[k].append(v[index_domain[list(y_domain_check.keys())[idx]]].squeeze().cpu().numpy().astype(dtype=np.int32))
                        labels_domain[k].append(toUse_.cpu().numpy().astype(dtype=np.int32))
                            
                ### evaluate model
                model_output  = model(*inputs)
                model_output_cat = model_output[:,:num_labels]
                model_output_domain = model_output[:,num_labels:num_labels+num_labels_domain]
                model_output_cat = _flatten_preds(model_output_cat,None)
                label_cat = label_cat.squeeze()


                # print("label_domain", label_domain)  
                # print ("label_domain.nelement()",label_domain.nelement())
                # print("labels_domain", labels_domain)  
                # print ("labels_domain.nelement()",labels_domain.nelement())
                # print("label_domain_check", label_domain_check)  
                # print ("label_domain_check.nelement()",label_domain_check.nelement())

                if label_domain.nelement() == 0  or label_domain.nelement()> 1:
                    label_domain = label_domain.squeeze()
                    label_domain_check = label_domain_check.squeeze()
                else:
                    label_domain = label_domain
                    # label_domain_check = label_domain_check[0]
                # label_domain = label_domain.squeeze()
                # label_domain_check = label_domain_check.squeeze()

                # print("label_domain", label_domain)  
                # print ("label_domain.nelement()",label_domain.nelement())

                ### in validation only filter interesting events
                if for_training:                        
                    model_output_cat = model_output_cat[index_cat]
                    model_output_domain = model_output_domain[index_domain_all]            
                    ### adjust outputs
                    model_output_cat = model_output_cat.squeeze().float()
                    # print ("model_output_domain",model_output_domain)
                    # print ("model_output_domain.nelement()",model_output_domain.nelement())
                    if model_output_domain.nelement() == 0  or model_output_domain.nelement()> 2:
                        model_output_domain = model_output_domain.squeeze().float()
                    else:
                        model_output_domain = model_output_domain[0]
                    # model_output_domain = model_output_domain.squeeze().float()
                    # print ("model_output_domain",model_output_domain)
                    # print ("model_output_cat",model_output_cat)
                    
                    scores_cat.append(torch.softmax(model_output_cat,dim=1).cpu().numpy().astype(dtype=np.float32))
                    # print ("scores_cat", scores_cat)
                    # print (y_domain)
                    for idx, name in enumerate(y_domain.keys()):
                        # print ("idx,name", idx,name)
                        # print ("ldomain", ldomain)
                        id_dom = idx*ldomain[idx]
                        # print ("id_dom,id_dom+ldomain[idx]", id_dom,id_dom+ldomain[idx])
                        score_domain = model_output_domain[:,id_dom:id_dom+ldomain[idx]]
                        # print ("score_domain", score_domain)
                        # print ("y_domain_check.keys()", y_domain_check.keys())
                        # print ("score_domain[index_domain[list(y_domain_check.keys())[idx]]]", score_domain[index_domain[list(y_domain_check.keys())[idx]]])
                        # print ("score_domain[index_domain[list(y_domain_check.keys())[idx]]].squeeze()", score_domain[index_domain[list(y_domain_check.keys())[idx]]].squeeze())
                        # to_use = score_domain[index_domain[list(y_domain_check.keys())[idx]]].squeeze()
                        if score_domain.nelement()==0 or score_domain.nelement()>2:
                            to_use = score_domain[index_domain[list(y_domain_check.keys())[idx]]].squeeze()
                        else:
                            to_use = score_domain[index_domain[list(y_domain_check.keys())[idx]]][0]
                        # scores_domain[name].append(torch.softmax(score_domain[index_domain[list(y_domain_check.keys())[idx]]].squeeze(),dim=1).cpu().numpy().astype(dtype=np.float32))
                        scores_domain[name].append(torch.softmax(to_use,dim=1).cpu().numpy().astype(dtype=np.float32))
                else:

                    model_output_cat = model_output_cat.float()
                    model_output_domain = model_output_domain.float()

                    scores_cat.append(torch.softmax(model_output_cat,dim=1).cpu().numpy().astype(dtype=np.float32))
                    for idx, name in enumerate(y_domain.keys()):
                        id_dom = idx*ldomain[idx]
                        score_domain = model_output_domain[:,id_dom:id_dom+ldomain[idx]]
                        scores_domain[name].append(torch.softmax(score_domain.squeeze(),dim=1).cpu().numpy().astype(dtype=np.float32))
                        
                    model_output_cat = model_output_cat[index_cat]
                    model_output_domain = model_output_domain[index_domain_all]                 
                    ### adjsut outputs
                    model_output_cat = model_output_cat.squeeze().float()
                    model_output_domain = model_output_domain.squeeze().float()

                ### evaluate loss function
                num_batches += 1
                index_offset += (num_cat_examples+num_domain_examples)

                if loss_func != None:
                    # print ("model_output_domain", model_output_domain)
                    # print ("label_domain", label_domain)
                    # print ("label_domain_check", label_domain_check)
                    # print ("model_output_cat", model_output_cat)
                    # print ("label_cat", label_cat)
                    loss, loss_cat, loss_domain = loss_func(model_output_cat,label_cat,model_output_domain,label_domain,label_domain_check);                
                    loss = loss.item()
                    if loss_cat:
                        loss_cat = loss_cat.item()
                    if loss_domain:
                        loss_domain = loss_domain.item()
                else:
                    loss,loss_cat,loss_domain = 0,0,0
                                    
                total_loss += loss
                total_cat_loss += loss_cat
                total_domain_loss += loss_domain

                ## prediction + metric for classification
                if np.iterable(label_cat) and torch.is_tensor(label_cat) and np.iterable(model_output_cat) and torch.is_tensor(model_output_cat):
                    _, pred_cat = model_output_cat.max(1)
                    if pred_cat.shape == label_cat.shape:
                        correct_cat = (pred_cat == label_cat).sum().item()
                        count_cat += num_cat_examples
                        total_cat_correct += correct_cat

                ## single domain region                                                                                                                                                          
                if num_domains == 1:
                    if torch.is_tensor(label_domain) and torch.is_tensor(model_output_domain) and np.iterable(label_domain) and np.iterable(model_output_domain):
                        _, pred_domain = model_output_domain.max(1)
                        if pred_domain.shape == label_domain.shape:
                            correct_domain = (pred_domain == label_domain).sum().item()
                            total_domain_correct += correct_domain
                            count_domain += num_domain_examples                
                ## multiple domains
                else:
                    correct_domain = 0
                    for idx, (k,v) in enumerate(y_domain_check.items()):
                        id_dom = idx*ldomain[idx]
                        label = label_domain[index_domain[k],idx].squeeze()
                        if not torch.is_tensor(label) or not np.iterable(label): continue;
                        pred_domain = model_output_domain[index_domain[k],id_dom:id_dom+ldomain[id]].squeeze()
                        if not torch.is_tensor(pred_domain) or not np.iterable(pred_domain): continue
                        _, pred_domain = pred_domain.max(1)
                        if pred_domain.shape != label.shape: continue
                        correct_domain += (pred_domain == label).sum().item()
                    total_domain_correct += correct_domain
                    count_domain += num_domain_examples    

                tq.set_postfix({
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / num_batches),
                    'Acc': '%.5f' % (correct_cat / num_cat_examples if num_cat_examples else 0),
                    'AvgAcc': '%.5f' % (total_cat_correct / count_cat if count_cat else 0),
                    'AccDomain': '%.5f' % (correct_domain / num_domain_examples if num_domain_examples else 0),
                    'AvgAccDomain': '%.5f' % (total_domain_correct / count_domain if count_domain else 0),
                    })

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches,
                                                mode='eval' if for_training else 'test')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count_cat+count_domain, (count_cat+count_domain / time_diff)))
    _logger.info('Eval AvgLoss Cat: %.5f'% (total_cat_loss / num_batches))
    _logger.info('Eval AvgLoss Domain: %.5f'% (total_domain_loss / num_batches))
    _logger.info('Eval AvgAccCat: %.5f'%(total_cat_correct / count_cat if count_cat else 0))
    _logger.info('Eval AvgAccDomain: %.5f'%(total_domain_correct / (count_domain) if count_domain else 0))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_cat_counter.items())))
    _logger.info('Train domain distribution: \n %s', ' '.join([str(sorted(i.items())) for i in label_domain_counter]))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)"%(tb_mode), total_loss / num_batches, epoch),
            ("Loss Cat/%s (epoch)"%(tb_mode), total_cat_loss / num_batches, epoch),
            ("Loss Domain/%s (epoch)"%(tb_mode), total_domain_loss / num_batches, epoch),
            ("AccCat/%s (epoch)"%(tb_mode), total_cat_correct / count_cat if count_cat else 0, epoch),
            ("AccDomain/%s (epoch)"%(tb_mode), total_domain_correct / count_domain if count_domain else 0, epoch),
            ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)

    # scores = np.concatenate(scores)
        #####
    scores_cat = np.concatenate(scores_cat).squeeze()
    scores_domain = {k: _concat(v) for k, v in scores_domain.items()}
    if not for_training:
        indexes_cat = np.concatenate(indexes_cat).squeeze()
        indexes_domain = {k: _concat(v) for k, v in indexes_domain.items()}
    labels_cat    = {k: _concat(v) for k, v in labels_cat.items()}
    labels_domain = {k: _concat(v) for k, v in labels_domain.items()}
    observers     = {k: _concat(v) for k, v in observers.items()}

    # print ("scores_cat",scores_cat)
    # print ("labels_cat",labels_cat)
    # print ("scores_domain",scores_domain)
    # print ("labels_domain",labels_domain)

    # labels = {k: _concat(v) for k, v in labels.items()}
    # metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)
    # _logger.info('Evaluation metrics: \n%s', '\n'.join(
    #     ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))

    if not for_training:
        metric_cat_results = evaluate_metrics(labels_cat[data_config.label_names[0]][indexes_cat].squeeze(),scores_cat[indexes_cat].squeeze(),eval_metrics=eval_cat_metrics)            
        _logger.info('Evaluation Classification metrics: \n%s', '\n'.join(
            ['    - %s: \n%s' % (k, str(v)) for k, v in metric_cat_results.items()]))

        for idx, (name,element) in enumerate(labels_domain.items()):
            metric_domain_results = evaluate_metrics(element[indexes_domain[name]].squeeze(),scores_domain[name][indexes_domain[name]].squeeze(),eval_metrics=eval_cat_metrics)
            _logger.info('Evaluation Domain metrics for '+name+' : \n%s', '\n'.join(
                ['    - %s: \n%s' % (k, str(v)) for k, v in metric_domain_results.items()]))
           
    else:
        metric_cat_results = evaluate_metrics(labels_cat[data_config.label_names[0]],scores_cat,eval_metrics=eval_cat_metrics)    
        _logger.info('Evaluation Classification metrics: \n%s', '\n'.join(
            ['    - %s: \n%s' % (k, str(v)) for k, v in metric_cat_results.items()]))

        for idx, (name,element) in enumerate(labels_domain.items()):
            metric_domain_results = evaluate_metrics(element,scores_domain[name],eval_metrics=eval_cat_metrics)
            _logger.info('Evaluation Domain metrics for '+name+' : \n%s', '\n'.join(
                ['    - %s: \n%s' % (k, str(v)) for k, v in metric_domain_results.items()]))


    if for_training:
        # return total_correct / count
        return total_loss / num_batches
    else:
        scores_domain = np.concatenate(list(scores_domain.values()),axis=1)
        scores_domain = scores_domain.reshape(len(scores_domain),num_labels_domain)
        scores = np.concatenate((scores_cat,scores_domain),axis=1)
        # convert 2D labels/scores
        # if len(scores_cat) != entry_count:
        #     if len(labels_counts):
        #         labels_counts = np.concatenate(labels_counts)
        #         scores_cat = ak.unflatten(scores_cat, labels_counts)
        #         for k, v in labels.items():
        #             labels[k] = ak.unflatten(v, labels_counts)
        #     else:
        #         assert(count % entry_count == 0)
        #         # scores = scores.reshape((entry_count, int(count / entry_count), -1)).transpose((1, 2))
        #         scores_domain = np.concatenate(list(scores_domain.values()),axis=1)
        #         scores_domain = scores_domain.reshape(len(scores_domain),num_labels_domain)
        #         scores = np.concatenate((scores_cat,scores_domain),axis=1)

        #         for k, v in labels.items():
        #             labels[k] = v.reshape((entry_count, -1))

        # return total_correct / count, scores, labels, labels_domain, observers
        return total_loss / num_batches, scores, labels_cat, labels_domain, observers


def evaluate_onnx_domain(model_path, test_loader, eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                        eval_cat_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix']
                        ):
    import onnxruntime
    sess = onnxruntime.InferenceSession(model_path)

    data_config = test_loader.dataset.config

    label_counter = Counter()
    total_correct = 0
    count = 0
    scores = []
    labels = defaultdict(list)
    labels_domain = defaultdict(list)
    observers = defaultdict(list)
    start_time = time.time()
    with tqdm.tqdm(test_loader) as tq:
        for X, y, Z in tq:
            inputs = {k: v.cpu().numpy() for k, v in X.items()}
            label = y[data_config.label_names[0]].cpu().numpy()
            num_examples = label.shape[0]
            label_counter.update(label)
            score = sess.run([], inputs)[0]
            preds = score.argmax(1)

            scores.append(score)
            for k, v in y.items():
                labels[k].append(v.cpu().numpy())
            for k, v in Z.items():
                observers[k].append(v.cpu().numpy())

            correct = (preds == label).sum()
            total_correct += correct
            count += num_examples

            tq.set_postfix({
                'Acc': '%.5f' % (correct / num_examples),
                'AvgAcc': '%.5f' % (total_correct / count)})

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))

    scores = np.concatenate(scores)
    labels = {k: _concat(v) for k, v in labels.items()}
    metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)
    _logger.info('Evaluation metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))
    observers = {k: _concat(v) for k, v in observers.items()}
    return total_correct / count, scores, labels, labels_domain, observers

class TensorboardHelper(object):

    def __init__(self, tb_comment, tb_custom_fn):
        self.tb_comment = tb_comment
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(comment=self.tb_comment)
        _logger.info('Create Tensorboard summary writer with comment %s' % self.tb_comment)

        # initiate the batch state
        self.batch_train_count = 0

        # load custom function
        self.custom_fn = tb_custom_fn
        if self.custom_fn is not None:
            from weaver.utils.import_tools import import_module
            from functools import partial
            self.custom_fn = import_module(self.custom_fn, '_custom_fn')
            self.custom_fn = partial(self.custom_fn.get_tensorboard_custom_fn, tb_writer=self.writer)

    def __del__(self):
        self.writer.close()

    def write_scalars(self, write_info):
        for tag, scalar_value, global_step in write_info:
            self.writer.add_scalar(tag, scalar_value, global_step)
