"""Intent decomposition and a Transformer double-DQN bundle policy."""
from __future__ import annotations
from collections import deque
import copy
import json
import math
from pathlib import Path
import random
import numpy as np
from rabr.types import Prediction
from rabr.io import digest, file_hash
from .text2bundle_assets import NativeInteractions, RobertaEncoder
from .llm_common import CatalogIndex, input_context, stable_seed


def user_id(row):
    return str(row.get('user_id', row.get('input', {}).get('user_id', '__unknown__')))


def coverage_reward(item_vectors, interest_vectors):
    if not len(item_vectors):return 0.0
    groups=np.argmax(np.asarray(item_vectors) @ np.asarray(interest_vectors).T,axis=1)
    counts=np.bincount(groups,minlength=len(interest_vectors));p=counts[counts>0]/len(groups)
    return float(-(p*np.log(p)).sum())


def build_network(item_vectors, user_vectors, text_vectors, config):
    import torch
    from torch import nn
    hidden=int(config.get('hidden_size',320));heads=int(config.get('heads',2));projection_width=int(config.get('projection_hidden_size',64))
    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.items=nn.Embedding.from_pretrained(torch.tensor(item_vectors,dtype=torch.float32),freeze=False)
            self.users=nn.Embedding.from_pretrained(torch.tensor(user_vectors,dtype=torch.float32),freeze=False)
            self.register_buffer('text',torch.tensor(text_vectors,dtype=torch.float32))
            self.projection=nn.Sequential(nn.Linear(text_vectors.shape[1],projection_width),nn.ReLU(),nn.Linear(projection_width,hidden))
            self.types=nn.Embedding(6,hidden);self.positions=nn.Embedding(256,hidden)
            self.state_encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden,heads,hidden*4,float(config.get('dropout',.1)),batch_first=True),num_layers=int(config.get('layers',1)),enable_nested_tensor=False)
            self.action_encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(2*hidden,heads,hidden*8,float(config.get('dropout',.1)),batch_first=True),num_layers=int(config.get('layers',1)),enable_nested_tensor=False)
            self.end=nn.Parameter(torch.zeros(2*hidden));nn.init.normal_(self.end,std=.02)
        def forward(self, profile, history, selected, candidates, interests, normalize=False):
            device=self.text.device
            def lookup(values,table):return table(torch.tensor(values,dtype=torch.long,device=device)) if values else torch.empty((0,hidden),device=device)
            q=self.projection(torch.as_tensor(interests,device=device,dtype=torch.float32))
            ht=self.projection(self.text[history]) if history else q[:0]
            bt=self.projection(self.text[selected]) if selected else q[:0]
            hi,bi=lookup(history,self.items),lookup(selected,self.items)
            if profile>=0:u=self.users.weight[profile:profile+1]
            else:raise ValueError('Text2Bundle requires a pretrained user ID embedding')
            chunks=[q,ht,bt,u,hi,bi];sequence=torch.cat(chunks)
            kinds=torch.cat([torch.full((len(c),),i,dtype=torch.long,device=device) for i,c in enumerate(chunks)])
            if len(sequence)>256:raise ValueError('state exceeds configured positional capacity')
            sequence=sequence+self.types(kinds)+self.positions(torch.arange(len(sequence),device=device))
            encoded=self.state_encoder(sequence[None])[0]
            cut=len(q)+len(ht)+len(bt)
            state=torch.cat([encoded[:cut].mean(0),encoded[cut:].mean(0)])
            if candidates:
                actions=torch.cat([lookup(candidates,self.items),self.projection(self.text[candidates])],dim=1)
                if not normalize:actions=self.action_encoder(actions[None])[0]
                actions=torch.cat([actions,self.end[None]])
            else:actions=self.end[None]
            if normalize:
                actions=torch.nn.functional.normalize(actions,dim=-1)
                state=torch.nn.functional.normalize(state,dim=-1)
            return actions @ state
    return Network()


class Text2Bundle:
    def __init__(self, config, backend):
        self.config,self.backend=dict(config),backend
        self.intent_cache={};self.intent_trace={};self.intent_calls=0

    def interests(self, context):
        query=input_context(context)['intent']
        if not query.strip():raise ValueError('Text2Bundle requires an explicit user query')
        if query not in self.intent_cache:
            prompt=[{'role':'user','content':'Infer four or five product entities for this shopping request. Separate the entities using |. For example, a request for expanded storage may imply a hard drive, SSD, USB key, or SD card. Query: '+query}]
            raw=self.backend.generate(prompt,seed=stable_seed(int(self.config.get('seed',42)),'intent',query));self.intent_calls+=1
            parts=[part.strip() for part in raw.split('|')]
            valid=4 <= len(parts) <= 5 and all(parts)
            text=list(dict.fromkeys((parts if valid else [])+[query]))
            self.intent_cache[query]=self.text_encoder.encode(text)
            self.intent_trace[query]={'raw_output':raw,'parsed':valid,'texts':text}
        return self.intent_cache[query]

    def state_input(self, row):
        if any(key in row for key in ('gold_items','target_items')):raise ValueError('Targets must not enter Text2Bundle state inputs')
        profile=self.users.get(user_id(row),-1)
        if profile<0:raise ValueError('Missing pretrained user embedding for '+user_id(row))
        history=[self.index.lookup[i['item_id']] for i in input_context(row)['history'] if i['item_id'] in self.index.lookup][-int(self.config.get('history_length',20)):]
        return profile,history,self.interests(row)

    def candidates(self, row, profile, history):
        query=self.index.encode([input_context(row)['intent'] or 'Products'])[0]
        cosine=query@self.index.vectors.T
        recalled=np.flatnonzero(cosine>=float(self.config.get('similarity_threshold',0.3))).tolist()
        if not recalled:return []
        scores=self.native.rank(profile,recalled)
        order=np.argsort(-scores,kind='stable')[:int(self.config.get('candidate_count',50))]
        return [recalled[int(i)] for i in order]

    def _choose(self, row, limit):
        import torch
        profile,history,interests=self.state_input(row);remaining=self.candidates(row,profile,history);selected=[]
        self.model.eval()
        with torch.inference_mode():
            while remaining and len(selected)<limit:
                q=self.model(profile,history,selected,remaining,interests)
                choice=int(q.argmax())
                if choice==len(remaining):break
                selected.append(remaining.pop(choice))
        return selected

    def fit(self, train, validation, catalog):
        import torch
        train,validation=list(train),list(validation)
        if not train or not validation:raise ValueError('Text2Bundle requires train and validation rows')
        if {r['session_id'] for r in train}&{r['session_id'] for r in validation}:raise ValueError('train/validation overlap')
        seed=int(self.config.get('seed',42));torch.manual_seed(seed);rng=random.Random(seed)
        torch.set_num_threads(int(self.config.get('cpu_threads',4)))
        self.native=NativeInteractions(self.config,[row['item_id'] for row in catalog])
        self.users=self.native.users
        self.text_encoder=RobertaEncoder(self.config)
        self.index=CatalogIndex(catalog,self.text_encoder)
        self.text_vectors=self.text_encoder.encode([row['title'] for row in catalog])
        identity=digest({'train':train,'validation':validation,'catalog':catalog,'config':self.config,
                         'native_assets':self.native.hashes,'text_encoder':self.text_encoder.identity,
                         'code':{name:file_hash(Path(__file__).with_name(name)) for name in ('llm_text2bundle.py','text2bundle_assets.py')}})
        directory=Path(self.config.get('work_dir',self.config.get('artifact_dir','runs/text2bundle')));directory.mkdir(parents=True,exist_ok=True)
        checkpoint=directory/'training.pt';saved=None
        if checkpoint.exists():
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if saved['identity']!=identity:raise ValueError('Text2Bundle checkpoint input/config mismatch')
            self.intent_cache=saved['intent_cache'];self.intent_calls=saved['intent_calls'];self.intent_trace=saved['intent_trace']
        item_vectors,user_vectors=self.native.item_vectors,self.native.user_vectors
        self.model=build_network(item_vectors,user_vectors,self.text_vectors,self.config).to(self.config.get('device','cpu'))
        target=copy.deepcopy(self.model).eval();optimizer=torch.optim.Adam(self.model.parameters(),lr=float(self.config.get('learning_rate',.00005)))
        for row in train+validation:self.interests(row)
        states=[self.state_input(r) for r in train]
        pools=[self.candidates(r,*state[:2]) for r,state in zip(train,states)]
        replay=deque(maxlen=int(self.config.get('replay_size',20000)))
        step=0;best=-float('inf');best_state=None;validation_history=[];episode=None
        if saved:
            self.model.load_state_dict(saved['model']);target.load_state_dict(saved['target']);optimizer.load_state_dict(saved['optimizer']);rng.setstate(saved['rng']);torch.set_rng_state(saved['torch_rng'])
            if torch.cuda.is_available() and saved.get('cuda_rng'):torch.cuda.set_rng_state_all(saved['cuda_rng'])
            replay.extend(saved['replay']);step=saved['step'];best=saved['best'];best_state=saved['best_state'];validation_history=saved['validation_history'];episode=saved.get('episode')
        updates=int(self.config.get('updates',20000));interval=int(self.config.get('eval_every',1000));maximum=int(self.config.get('max_bundle_size',5));batch_size=int(self.config.get('batch_size',256));gamma=float(self.config.get('discount',.99))
        if min(updates,interval,maximum,batch_size,int(self.config.get('target_update',100)))<1:raise ValueError('training budgets must be positive')
        if not saved:
            self.model.train()
            for _ in range(int(self.config.get('pretrain_steps',500))):
                row_index=rng.randrange(len(train));p,h,interests=states[row_index]
                gold=list(dict.fromkeys(self.index.lookup[i] for i in train[row_index]['target']['gold_items']))
                negatives=sorted(set(range(len(catalog)))-set(gold))
                if not gold or not negatives:continue
                rng.shuffle(gold);cut=rng.randrange(len(gold));prefix=gold[:cut]
                positive=gold[cut];negative=rng.choice(negatives)
                scores=self.model(p,h,prefix,[positive,negative],interests,normalize=True)
                loss=torch.nn.functional.softplus(-(scores[0]-scores[1]))
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            target.load_state_dict(self.model.state_dict())
        def save():
            state={'identity':identity,'intent_trace':self.intent_trace,
                   'intent_cache':self.intent_cache,'intent_calls':self.intent_calls,'model':self.model.state_dict(),'target':target.state_dict(),'optimizer':optimizer.state_dict(),
                   'rng':rng.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                   'replay':list(replay),'step':step,'best':best,'best_state':best_state,'validation_history':validation_history,'episode':episode}
            temporary=checkpoint.with_suffix('.tmp');torch.save(state,temporary);temporary.replace(checkpoint)
        if not saved:save()
        while step<updates:
            if episode is None:
                row_index=rng.randrange(len(train));remaining=list(pools[row_index]);selected=[]
            else:
                row_index=episode['row_index'];remaining=list(episode['remaining']);selected=list(episode['selected'])
            profile,history,interests=states[row_index];gold={self.index.lookup[i] for i in train[row_index]['target']['gold_items']}
            while step<updates:
                before=tuple(selected);candidates=tuple(remaining);self.model.eval()
                epsilon=max(float(self.config.get('epsilon_min',.05)),1-step/max(1,updates))
                with torch.inference_mode():q=self.model(profile,history,selected,remaining,interests)
                action=rng.randrange(len(remaining)+1) if rng.random()<epsilon else int(q.argmax())
                end=action==len(remaining)
                if not end:selected.append(remaining.pop(action))
                precision=len(set(selected)&gold)/len(selected) if selected else 0.
                personal=0. if end else float(user_vectors[profile]@item_vectors[selected[-1]]) if profile>=0 else 0.
                complement=float(np.mean(self.text_vectors[selected[:-1]]@self.text_vectors[selected[-1]])) if not end and len(selected)>1 else 0.
                reward=float(self.config.get('main_reward_weight',3.))*precision+personal+complement+coverage_reward(self.text_vectors[selected],interests)
                done=end or not remaining or len(selected)>=maximum
                episode=None if done else {'row_index':row_index,'remaining':list(remaining),'selected':list(selected)}
                replay.append((row_index,before,candidates,action,reward,tuple(selected),tuple(remaining),done));step+=1
                self.model.train();losses=[]
                for sample in rng.sample(list(replay),min(batch_size,len(replay))):
                    r,b,c,a,reward,next_b,next_c,terminal=sample;p,h,qtext=states[r]
                    estimate=self.model(p,h,list(b),list(c),qtext)[a]
                    with torch.no_grad():
                        if terminal:expected=estimate.new_tensor(reward)
                        else:
                            self.model.eval();chosen=int(self.model(p,h,list(next_b),list(next_c),qtext).argmax());self.model.train()
                            expected=estimate.new_tensor(reward)+gamma*target(p,h,list(next_b),list(next_c),qtext)[chosen]
                    losses.append(torch.nn.functional.smooth_l1_loss(estimate,expected))
                optimizer.zero_grad();loss=torch.stack(losses).mean();loss.backward();torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                if step%int(self.config.get('target_update',100))==0:target.load_state_dict(self.model.state_dict())
                if step%interval==0 or step==updates:
                    scores=[]
                    for row in validation:
                        predicted={self.index.ids[i] for i in self._choose(row,maximum)};gold_ids=set(row['target']['gold_items'])
                        scores.append(2*len(predicted&gold_ids)/(len(predicted)+len(gold_ids)))
                    score=float(np.mean(scores));improved=score>best
                    if improved:best=score;best_state={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
                    validation_history.append({'step':step,'validation_f1':score,'selected':improved});save()
                if done:break
        if best_state is None:raise ValueError('training selected no checkpoint')
        self.model.load_state_dict(best_state)
        self.fit_metadata={'updates':step,'pretrain_steps':int(self.config.get('pretrain_steps',500)),'selection_split':'validation','validation_history':validation_history,'intent_llm_calls':self.intent_calls,'identity':identity,'native_assets':self.native.hashes,
                           'text_encoder':self.text_encoder.identity,'intent_trace':self.intent_trace,'feature_profile':'external_amazon_lightgcn_fm_roberta'}
        (directory/'selection.json').write_text(json.dumps(self.fit_metadata,indent=2)+'\n')
        return self

    def predict(self, context, k, seed):
        if k<1:raise ValueError('k must be positive')
        if any(key in context for key in ('target','gold_items','target_items')):raise ValueError('Prediction context must not include targets')
        previous=self.intent_calls
        selected=self._choose(context,int(self.config.get('max_bundle_size',5)))
        return Prediction(items=[self.index.ids[i] for i in selected],metadata={'method':'text2bundle',
            'implementation_variant':'external_amazon_lightgcn_fm_roberta_double_dqn','llm_calls':self.intent_calls-previous,'fit':self.fit_metadata})
