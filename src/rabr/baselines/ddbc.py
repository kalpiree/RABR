"""Train and sample a residual-code masked diffusion recommender."""
from __future__ import annotations
from argparse import Namespace
import hashlib
import json
from pathlib import Path
import numpy as np
from rabr.types import Prediction
from .llm_common import CatalogIndex, context_text
from .ddbc_tokenizer import pca_whiten, extend_codes, faiss_quantize, load_clhe_features, retrieve_generated_codes


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


class DDBC:
    def __init__(self, config, backend):
        self.config, self.backend = dict(config), backend
        self.feature_mode = self.config.get('feature_mode', 'clhe')
        self.context_mode = self.config.get('context_mode', 'provided_seed_items')
        self.decoder = self.config.get('decoder', 'levelwise')
        self.output_mode = self.config.get('output_mode', 'completion_only')
        if self.feature_mode not in ('clhe', 'shared_text') or self.context_mode not in ('provided_seed_items', 'query_seed'):
            raise ValueError('Unknown DDBC feature or context mode')
        if self.decoder not in ('levelwise', 'catalog_constrained') or self.output_mode not in ('completion_only', 'full_bundle'):
            raise ValueError('Unknown DDBC decoder or output mode')
        self.required_input_fields = ('seed_items',) if self.context_mode == 'provided_seed_items' else ()

    def validate_evaluation_rows(self, rows):
        if self.context_mode != 'provided_seed_items':
            return
        for row in rows:
            seeds = row['input'].get('seed_items')
            if not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds)):
                raise ValueError('Native DDBC requires distinct supplied input.seed_items')
            if self.output_mode == 'completion_only' and set(seeds) & set(row['target']['gold_items']):
                raise ValueError('DDBC completion targets must exclude all supplied seed items')

    def fit(self, train, validation, catalog):
        from .ddbc_training import TorchEngine
        train,validation,catalog=list(train),list(validation),list(catalog)
        if not train or not validation:raise ValueError('DDBC requires training and validation rows')
        if {r['session_id'] for r in train}&{r['session_id'] for r in validation}:raise ValueError('training and validation overlap')
        ids=[row['item_id'] for row in catalog]
        self.index=CatalogIndex(catalog,self.backend) if self.context_mode=='query_seed' or self.feature_mode=='shared_text' else Namespace(ids=ids,lookup={item:i for i,item in enumerate(ids)})
        if self.feature_mode=='clhe':
            vectors,feature_provenance=load_clhe_features(self.config.get('clhe_manifest'),ids,train)
        else:
            vectors=self.index.vectors
            feature_provenance={'feature_type':'shared_text','encoder':dict(getattr(self.backend,'config',{}))}
        seed=int(self.config.get('seed',42));device=self.config.get('device','cpu')
        args=Namespace(seed=seed,device=device,cpu_threads=int(self.config.get('cpu_threads',8)),
             batch_size=int(self.config.get('batch_size',64)),learning_rate=float(self.config.get('learning_rate',3e-4)),
             warmup_steps=int(self.config.get('warmup_steps',500)),ema_decay=float(self.config.get('ema_decay',.9999)),
             sampling_steps=int(self.config.get('sampling_steps',25)))
        steps=int(self.config.get('steps',20000));interval=int(self.config.get('eval_every',500))
        if steps<1 or interval<1:raise ValueError('training budgets must be positive')
        identity=fingerprint({'train':train,'validation':validation,'catalog':catalog,'config':self.config,'features':feature_provenance})
        directory=Path(self.config.get('work_dir',self.config.get('artifact_dir','runs/ddbc')))
        directory.mkdir(parents=True,exist_ok=True)
        identity_path=directory/'identity.json';latest=directory/'training.pt'
        dimensions=int(self.config.get('pca_dimensions',0 if self.feature_mode=='clhe' else 64))
        if not 0<=dimensions<=64:raise ValueError('PCA dimensions must be between 0 and 64')
        books=int(self.config.get('codebook_size',256))
        if identity_path.exists():
            tokenization=json.loads(identity_path.read_text())
            if tokenization['sha256']!=identity or tokenization['catalog_ids']!=self.index.ids:raise ValueError('DDBC tokenizer input/config mismatch')
            for filename,expected in tokenization['files'].items():
                if hashlib.sha256((directory/filename).read_bytes()).hexdigest()!=expected:raise ValueError('DDBC tokenizer artifact hash mismatch')
            self.codes=np.load(directory/'catalog-codes.npy',allow_pickle=False)
            weights=np.load(directory/'codebook-weights.npy',allow_pickle=False)
            self.features=np.load(directory/'catalog-features.npy',allow_pickle=False)
            collision=tokenization['collision_maximum']
        else:
            if latest.exists():raise ValueError('DDBC training checkpoint has no tokenizer identity')
            if dimensions:
                reduced,pca=pca_whiten(vectors,dimensions)
                if dimensions<64:reduced=np.pad(reduced,((0,0),(0,64-dimensions)))
            else:
                if vectors.shape[1]!=64:raise ValueError('Unprojected CLHE features must have the 64 dimensions required by the released small model')
                reduced,pca=np.asarray(vectors,dtype=np.float32),{}
            self.features=reduced
            raw,weights,index_bytes=faiss_quantize(reduced,books,int(self.config.get('tokenizer_seed',42)),args.cpu_threads)
            self.codes,collision=extend_codes(raw,books)
            np.save(directory/'catalog-codes.npy',self.codes,allow_pickle=False)
            np.save(directory/'codebook-weights.npy',weights,allow_pickle=False)
            np.save(directory/'catalog-features.npy',self.features,allow_pickle=False)
            np.savez(directory/'pca.npz',**pca)
            (directory/'faiss.index').write_bytes(index_bytes)
            filenames=['catalog-codes.npy','codebook-weights.npy','catalog-features.npy','pca.npz','faiss.index']
            tokenization={'sha256':identity,'catalog_ids':self.index.ids,'collision_maximum':collision,'features':feature_provenance,
                          'files':{name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in filenames}}
            temporary=identity_path.with_suffix('.tmp');temporary.write_text(json.dumps(tokenization,indent=2)+'\n');temporary.replace(identity_path)
        prepared={'tokens':{'bos_id':0,'boi_id':4*books+1,'eos_id':4*books+2,'mask_id':4*books+3,'vocab_size':4*books+4}}
        self.engine=TorchEngine(args,prepared,self.codes,weights)
        self.codebook_weights=weights
        def examples(rows):
            output=[]
            for row in rows:
                ids=row['target']['gold_items']
                if len(ids)<2 or len(ids)!=len(set(ids)):raise ValueError('DDBC training bundles need at least two distinct items')
                output.append({'session_id':row['session_id'],'item_indices':[self.index.lookup[i] for i in ids]})
            return output
        training,valid=examples(train),examples(validation)
        state={'step':0,'best_loss':None,'best_step':None,'best_state':None,'validation_history':[]}
        if latest.exists():
            recovered=self.engine.load(latest)
            if recovered['identity']!=identity:raise ValueError('DDBC checkpoint input/config mismatch')
            state=recovered['state'];self.engine.restore(recovered['engine'])
        for step in range(state['step']+1,steps+1):
            loss=self.engine.train_step(training,step);state['step']=step
            if step%interval==0 or step==steps:
                score=self.engine.validate(valid)
                selected=state['best_loss'] is None or score<state['best_loss']
                if selected:state.update(best_loss=score,best_step=step,best_state=self.engine.selected_state())
                state['validation_history'].append({'step':step,'validation_loss':score,'selected':selected})
                if latest:
                    temporary=latest.with_suffix('.tmp');self.engine.save(temporary,{'identity':identity,'state':state,'engine':self.engine.snapshot()});temporary.replace(latest)
        self.engine.use_selected(state['best_state'])
        self.fit_metadata={'training_steps':steps,'selected_step':state['best_step'],'selection_split':'validation',
                           'validation_history':state['validation_history'],'training_cost':self.engine.cost,
                           'codebook_size':books,'pca_dimensions':dimensions,'collision_maximum':collision,'identity':identity,
                           'feature_mode':self.feature_mode,'feature_provenance':feature_provenance,'decoder':self.decoder,
                           'context_mode':self.context_mode,'output_mode':self.output_mode,
                           'training_adaptation':'same_length_batches_over_supplied_training_bundles_with_adjacent_swaps'}
        if directory:(directory/'selection.json').write_text(json.dumps(self.fit_metadata,indent=2)+'\n')
        return self

    def predict(self, context, k, seed):
        if k<1 or k>len(self.index.ids):raise ValueError('invalid bundle size')
        import torch
        from .ddbc_model import sample_bundles, sample_levelwise_codes
        if self.context_mode=='provided_seed_items':
            supplied=context['input'].get('seed_items')
            if not isinstance(supplied,list) or not supplied or len(supplied)!=len(set(supplied)) or not set(supplied)<=self.index.lookup.keys():
                raise ValueError('Native DDBC requires distinct input.seed_items supplied independently of targets')
            seed_indices=[self.index.lookup[item] for item in supplied]
        else:
            seed_indices=[self.index.retrieve(context_text(context),1)[0]]
        completion_items=k if self.output_mode=='completion_only' else k-len(seed_indices)
        if completion_items<0 or completion_items>len(self.index.ids)-len(seed_indices):raise ValueError('Invalid completion cardinality')
        if completion_items==0:
            return Prediction(items=[self.index.ids[i] for i in seed_indices],metadata={'method':'ddbc','denoiser_forward_calls':0})
        generator=torch.Generator(device=self.engine.device).manual_seed(seed)
        with torch.inference_mode(),torch.autocast(device_type=self.engine.device.type,dtype=torch.bfloat16,enabled=self.engine.device.type=='cuda'):
            if self.decoder=='levelwise':
                codes,diagnostics=sample_levelwise_codes(self.engine.model,torch.as_tensor(self.codes,device=self.engine.device),
                    torch.tensor([seed_indices],device=self.engine.device),**{key:self.engine.tokens[key] for key in ('bos_id','boi_id','eos_id','mask_id')},
                    completion_items=completion_items,codebook_size=int(self.config.get('codebook_size',256)),
                    steps=self.engine.args.sampling_steps,generator=generator,return_diagnostics=True)
                selected,retrieval=retrieve_generated_codes(codes[0].cpu().numpy(),self.codes,self.codebook_weights,self.features,
                    int(self.config.get('codebook_size',256)),excluded=seed_indices)
                diagnostics.update(retrieval)
            else:
                if len(seed_indices)!=1:raise ValueError('The catalog-constrained adapter supports exactly one seed')
                output=sample_bundles(self.engine.model,torch.as_tensor(self.codes,device=self.engine.device),
                    torch.tensor(seed_indices,device=self.engine.device),**{key:self.engine.tokens[key] for key in ('bos_id','boi_id','eos_id','mask_id')},
                    total_items=completion_items+1,steps=self.engine.args.sampling_steps,generator=generator)
                selected=output[0,1:].cpu().tolist()
                diagnostics={'denoiser_forward_calls':self.engine.args.sampling_steps,'constraint_policy':'catalog_prefix_and_distinct_tuple_adapter'}
        output_indices=seed_indices+selected if self.output_mode=='full_bundle' else selected
        return Prediction(items=[self.index.ids[i] for i in output_indices],metadata={
            'method':'ddbc','implementation_variant':f'{self.feature_mode}_{self.context_mode}_{self.decoder}_{self.output_mode}',
            'llm_calls':0,'seed_item_ids':[self.index.ids[i] for i in seed_indices],**diagnostics,'fit':self.fit_metadata})
