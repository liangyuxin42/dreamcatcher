import torch
import torch.nn as nn
from baukit import Trace, TraceDict
from sklearn.metrics import confusion_matrix
import copy

very_small_value = 0.000001

class LinearProbe(nn.Module):
    def __init__(self, hidden_size: int,):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1, bias=True)
        self.loss = nn.BCELoss()
        
    def forward(self,state, label):
        out = torch.sigmoid(self.linear(state))
        loss = self.loss(out,label)
        return out, loss
    
class ProbeTrainer():
    def __init__(self, model, tokenizer, epoch=10, lr=0.01, bs=32):
        self.model = model
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.lr = lr
        self.bs = bs
        # check key
        if not any(["model.layers.0.mlp" in key for key in model.state_dict().keys()]):
            # QWEN model
            
            self.MLPS = [f"transformer.h.{i}.mlp" for i in range(model.config.num_hidden_layers)]
        else:
            self.MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]

#         self.HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(model.config.num_hidden_layers)]
    
    def get_activation_data(self,text,label,answer=None):
        if answer is not None:
            text+=answer
        input_ids = self.tokenizer(text,return_tensors="pt",padding=True,truncation=True,max_length=2048).input_ids
#         with TraceDict(self.model, self.HEADS+self.MLPS) as ret:
        with TraceDict(self.model, self.MLPS) as ret: 
            with torch.no_grad():
                output = self.model(input_ids.to(self.model.device), output_hidden_states = True)
        hidden_states = [s.cpu()[0] for s in output.hidden_states]
        mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in self.MLPS]
#         head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in self.HEADS]
        d = {
            "question_text":text,
            "knownledge_label":label,
            "token_id":input_ids[0].tolist(),
            "hidden_states":hidden_states,
            "mlp_wise_hidden_states":mlp_wise_hidden_states,
#             "head_wise_hidden_states":head_wise_hidden_states,
        }
        return d 
    
    def get_state(self,data,state_key,token_key,layer,attn_head):
        label = torch.Tensor([d["knownledge_label"]=="all-right" for d in data]).unsqueeze(1)
        
        if isinstance(layer,list):
            state = [d[state_key] for d in data]
            state = [torch.cat([s[l] for l in layer],dim=1) for s in state]
        else:
            state = [d[state_key][layer] for d in data]

        if token_key=="last":
            state = torch.stack([s[-1] for s in state]).to(torch.float)
        elif token_key=="before_bot":
            state = torch.stack([s[-3] for s in state]).to(torch.float)
        else:
            raise NotImplementedError  
            
        if attn_head is not None:
            state_dim = self.model.config.hidden_size//self.model.config.num_attention_heads
            state = state.reshape(-1,self.model.config.num_attention_heads,state_dim)[:,attn_head]
        return state,label
    

    def get_acc(self,score,label):
        predict = torch.Tensor([s>0.5 for s in score])
        label = label.squeeze()
        acc = sum(predict==label)/len(label)
        tn, fp, fn, tp = confusion_matrix(label,predict).ravel()
        precision = tp/(tp+fp+very_small_value)
        recall = tp/(tp+fn+very_small_value)
        specificity = tn/(tn+fp+very_small_value)
        result = {"acc":round(acc.item(),3),"precision":round(precision,3), "recall":round(recall,3),"specificity":round(specificity,3)}

        return result

    def fit_one_probe(self,train,test,state_key,token_key,layer,attn_head=None):
        if state_key=="head_wise_hidden_states" and attn_head is not None:
            state_dim = self.model.config.hidden_size//self.model.config.num_attention_heads
        else:
            state_dim = self.model.config.hidden_size
        if isinstance(layer,list):
            state_dim=state_dim*len(layer)
        
        prober = LinearProbe(state_dim)
        best_probe = None
        best_acc = 0
        optimizer = torch.optim.SGD(prober.parameters(), lr=self.lr, momentum=0.9)
        test_results = []
        test_cout = int(len(train)/10)
        for ep in range(self.epoch):
            for i in range(0,len(train),self.bs):
                optimizer.zero_grad()
                state,label = self.get_state(train[i:i+self.bs],state_key,token_key,layer,attn_head)
                out,loss = prober(state,label)
                loss.backward()
                optimizer.step()
                if i%test_cout==0:
                    prober.eval()
                    state,label = self.get_state(test,state_key,token_key,layer,attn_head)
                    out,loss = prober(state,label)
                    result = self.get_acc(out, label)
                    if result["acc"]>best_acc:
                        best_probe = copy.deepcopy(prober)
                        best_acc=result["acc"]
                    test_results.append(result)
                    prober.train()
        return test_results,best_probe
