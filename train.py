# train.py

import torch
import torch.optim as optim
import torch.nn as nn
from model import FuSITSNet
from data import CropDataset_train, CropDataset

class MarginBasedContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(MarginBasedContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, target):
        distance = torch.nn.functional.pairwise_distance(anchor, positive)
        loss = torch.mean((1 - target) * torch.pow(distance, 2) +
                          (target) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return loss

def train(model, optimizer,epcs):
	cl_loss1= []
	cl_loss2= [] 
	cls_loss= []     
	mse_loss = []
    
	history_train = []
	history_val1 = []
	history_val2 = []
   
	output_train = []
	target_train = []

	for epoch in range(1, epcs+1):

		model.train()
      
		for batch_idx, (data_m, data_l, st_m,cnt_m,yr_m, target) in enumerate(train_dataloader):

			data_m, data_l, st_m,cnt_m,yr_m, target = (data_m.float()).to(device),(data_l.float()).to(device),(st_m.float()).to(device),(cnt_m.float()).to(device),(yr_m.float()).to(device),target.float().to(device)
			optimizer.zero_grad(set_to_none = True)     
			positive_pairs = []
			negative_pairs = []                        
			for i in range(0,len(target)):
			    positive_idx, negative_idx = make_pairs(st_m[i],cnt_m[i], yr_m[i],st_m,cnt_m,yr_m)            
			    positive_pairs.append(positive_idx)
			    negative_pairs.append(negative_idx)                

			data_m_pos = data_m[positive_pairs]
			data_l_pos = data_l[positive_pairs]
			lnt_pos = data_l_pos
			mds_pos = data_m_pos   

			anc_output, positive_out, negative_out= model(data_m, data_l,mds_pos,lnt_pos,data_m[negative_pairs], data_l[negative_pairs])  
			target = target.repeat(anc_output.shape[1], 1).T
			output = anc_output.squeeze(-1) 
            
			loss_1 = torch.sqrt(criterion_pred(output,target))
			loss_2 = criterion_triplet(anc_output, positive_out,target)
			loss_3 = criterion_triplet(anc_output, negative_out,target)            
			loss = loss_1 +loss_2 +loss_3

			loss.backward()
			optimizer.step()

		if epoch >= 0:
		  torch.save(model, os.path.join(path_to_save, "model_soy_%d.pt"%(epoch)))   
		print('Train Epoch: {} Training Loss: {}'.format(epoch,loss.item()))

		result_val1,train_preds, labels = evaluate(model,val_dataloader,'Validation')

		mse_loss.append(loss_1.item()) 
		history_train.append(loss.item())
		history_val1.append(result_val1)       
		output_train.append(train_preds)
		target_train.append(labels) 

        
	return history_train, history_val1,output_train, target_train, mse_loss # ,cls_loss,    
    

def evaluate(model,data_loader, dataset):
    model.eval()
    
    eval_loss1 = float(0) 
    eval_loss2 = float(0)  
    
    correct = 0
    avg_loss = []
    output_list = []
    target_list = []
    count = 0
    model = model.to(device)
    with torch.no_grad():
        for data_m, data_l, st_m,cnt_m,yr_m, target in data_loader:
            data_m, data_l, st_m,cnt_m,yr_m, target = (data_m.float()).to(device),(data_l.float()).to(device),(st_m.float()).to(device),(cnt_m.float()).to(device),(yr_m.float()).to(device),target.float().to(device)
            output, output_pos, output_neg = model(data_m, data_l,data_m, data_l,data_m, data_l)
            target = target.squeeze(-1) 
            output = output.squeeze(-1) 
            eval_lossb1 = criterion_print(output, target).mean(axis=0)*len(target)
            eval_lossb1 = torch.tensor(eval_lossb1)
            eval_loss1 += eval_lossb1        
            count += len(target)
            output_list += output.tolist()
            target_list += target.tolist()
            
    eval_loss1 /= count
    eval_loss1 = torch.sqrt(torch.tensor(eval_loss1))    
    eval_loss1 = eval_loss1.cpu().numpy()
   
    print('{} set: loss1: {}\n'.format(dataset,  eval_loss2))
    if dataset == 'Validation':
      return eval_loss1, output_list, target_list
    if dataset == 'Test':
      return eval_loss1, output_list, target_list

def make_pairs(st_m,cnt_m,m_yr,indix_stm,indix_cntm,year_m):

    pos_idxs = []
    neg_idxs = []
    total_idx = list(range(0,len(indix_cntm)))
    
    for i in range(0,len(indix_cntm)):
        posidxs = torch.where((indix_stm[:] == st_m) &(indix_cntm[:] == cnt_m) &(year_m[:] == m_yr))[0]
        posidxs = posidxs.cpu().numpy()                
        pos_id  =  np.random.choice(posidxs)
        negIdxs = torch.where((indix_stm[:] != st_m) |(indix_cntm[:] != cnt_m)|(year_m[:] != m_yr))[0]
        negIdxs = negIdxs.cpu().numpy()
        if len(negIdxs)==0:
            negIdxs = posidxs
        neg_id  =  np.random.choice(negIdxs)
        
       
    return [pos_id, neg_id]
