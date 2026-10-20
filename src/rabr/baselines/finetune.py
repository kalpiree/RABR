from __future__ import annotations
import json
import hashlib
import os
from pathlib import Path
import random
import time
import math


class StudentBackend:
    def __init__(self, model, tokenizer, encoder_backend, config):
        self.model, self.tokenizer, self.encoder_backend, self.config = model, tokenizer, encoder_backend, config

    def encode(self, texts):
        return self.encoder_backend.encode(texts)

    def generate(self, messages, seed):
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        self.model.eval()
        tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to(self.model.device)
        started=time.monotonic()
        temperature = float(self.config.get('temperature', .7))
        options = {'do_sample': temperature > 0}
        if temperature > 0:options.update(temperature=temperature, top_p=1.0)
        with torch.inference_mode():
            output = self.model.generate(tokens, max_new_tokens=int(self.config.get('max_new_tokens', 256)),
                     **options, pad_token_id=self.tokenizer.eos_token_id)
        generated=output[0,tokens.shape[1]:]
        measurement={'input_tokens':tokens.shape[1],'output_tokens':len(generated),'seconds':time.monotonic()-started}
        if hasattr(self.encoder_backend,'stats'):
            stats=self.encoder_backend.stats;stats['generation_calls']=stats.get('generation_calls',0)+1
            for key in ('input_tokens','output_tokens'):stats[key]=stats.get(key,0)+measurement[key]
            stats['generation_seconds']=stats.get('generation_seconds',0.)+measurement['seconds']
            self.encoder_backend.last_generation=measurement
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def supervised_tokens(tokenizer, record, max_length):
    prompt = tokenizer.apply_chat_template(record['messages'], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(record['messages'] + [{'role': 'assistant', 'content': record['response']}], tokenize=True, add_generation_prompt=False)
    if full[:len(prompt)] != prompt:
        raise ValueError('chat template does not preserve the assistant prefix')
    if len(full) > max_length or len(full) <= len(prompt):
        raise ValueError('supervised example exceeds context length or has no response tokens')
    return full, [-100] * len(prompt) + full[len(prompt):]


def train_student(train_records, validation_records, config, encoder_backend):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training
    name = os.path.expandvars(config.get('student_model', ''))
    if not name or '$' in name:
        raise ValueError('teacher_student mode requires student_model')
    if not train_records or not validation_records:
        raise ValueError('student training and validation records are required')
    seed = int(config.get('training_seed', config.get('seed',42)))
    torch.manual_seed(seed)
    options={'local_files_only':bool(config.get('local_files_only',True)),
             'revision':config.get('student_revision'),'trust_remote_code':False}
    tokenizer = AutoTokenizer.from_pretrained(name, **options)
    if not tokenizer.chat_template:
        raise ValueError('student tokenizer requires a chat template')
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    teacher_model=getattr(encoder_backend,'model',None)
    if teacher_model is not None:
        teacher_model.to('cpu')
        if torch.cuda.is_available():torch.cuda.empty_cache()
    quantized = bool(config.get('load_in_4bit', False))
    if quantized and not str(device).startswith('cuda'):
        raise ValueError('The 4-bit KD4BG profile requires CUDA; explicitly disable load_in_4bit for CPU verification')
    load_options = dict(options, torch_dtype=torch.bfloat16 if str(device).startswith('cuda') else torch.float32)
    if quantized:
        load_options.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True), device_map={'': device})
    model = AutoModelForCausalLM.from_pretrained(name, **load_options)
    if quantized:
        model = prepare_model_for_kbit_training(model)
    else:
        model = model.to(device)
    torch.manual_seed(int(config.get('adapter_seed', 3407)))
    model = get_peft_model(model, LoraConfig(r=int(config.get('lora_rank', 16)), lora_alpha=int(config.get('lora_alpha', 16)),
              lora_dropout=0., target_modules=config.get('target_modules', ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']), task_type='CAUSAL_LM'))
    torch.manual_seed(seed)
    maximum = int(config.get('max_sequence_length', 2048))
    train = [supervised_tokens(tokenizer, r, maximum) for r in train_records]
    validation = [supervised_tokens(tokenizer, r, maximum) for r in validation_records]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=float(config.get('learning_rate', 2e-4)), weight_decay=float(config.get('weight_decay', .01)))
    accumulation = int(config.get('gradient_accumulation', 4))
    batch_size = int(config.get('batch_size', 2))
    if accumulation < 1 or batch_size < 1:raise ValueError('batch size and gradient accumulation must be positive')
    epochs = int(config.get('epochs', 3))
    total_updates = epochs * math.ceil(math.ceil(len(train) / batch_size) / accumulation)
    warmup = int(config.get('warmup_steps', 5))
    if config.get('lr_scheduler', 'linear') != 'linear':raise ValueError('KD4BG uses the linear warmup/decay schedule')
    def multiplier(step):
        if step < warmup:return step / max(1, warmup)
        return max(0., (total_updates - step) / max(1, total_updates - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    selection = config.get('checkpoint_selection', 'final_epoch')
    if selection not in {'final_epoch', 'validation_loss'}:raise ValueError('Unknown student checkpoint selection')
    directory = Path(config.get('work_dir',config.get('artifact_dir','runs/kd4bg-student')));directory.mkdir(parents=True,exist_ok=True)
    identity=hashlib.sha256(json.dumps({'train':train_records,'validation':validation_records,'config':config},sort_keys=True).encode()).hexdigest()
    checkpoint=directory/'student-training.pt'
    best, best_state, history, start_epoch = float('inf'), None, [], 0
    rng = random.Random(int(config.get('shuffle_seed', 65)))
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['identity']!=identity:raise ValueError('KD4BG student checkpoint input/config mismatch')
        set_peft_model_state_dict(model,saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        best,best_state,history,start_epoch=saved['best'],saved['best_state'],saved['history'],saved['epoch']
        rng.setstate(saved['rng']);torch.set_rng_state(saved['torch_rng'])
        if torch.cuda.is_available() and saved['cuda_rng']:torch.cuda.set_rng_state_all(saved['cuda_rng'])
    def loss(examples):
        maximum = max(len(example[0]) for example in examples)
        padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if padding is None:raise ValueError('Student tokenizer requires a padding or EOS token')
        ids = [x + [padding] * (maximum - len(x)) for x, _ in examples]
        labels = [y + [-100] * (maximum - len(y)) for _, y in examples]
        mask = [[1] * len(x) + [0] * (maximum - len(x)) for x, _ in examples]
        return model(input_ids=torch.tensor(ids,device=device), attention_mask=torch.tensor(mask,device=device),
                     labels=torch.tensor(labels,device=device)).loss
    for epoch in range(start_epoch, epochs):
        order = list(range(len(train)));rng.shuffle(order);model.train()
        batches = [order[start:start + batch_size] for start in range(0, len(order), batch_size)]
        for start in range(0, len(batches), accumulation):
            group = batches[start:start + accumulation];optimizer.zero_grad(set_to_none=True)
            for batch in group:(loss([train[i] for i in batch]) / len(group)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True);optimizer.step();scheduler.step()
        model.eval()
        with torch.inference_mode():score = sum(float(loss([r])) for r in validation) / len(validation)
        history.append({'epoch': epoch+1, 'validation_loss': score})
        if selection == 'final_epoch' or score < best:
            best = score;best_state = {k: v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
        saved={'identity':identity,'epoch':epoch+1,'model':get_peft_model_state_dict(model),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
               'best':best,'best_state':best_state,'history':history,'rng':rng.getstate(),'torch_rng':torch.get_rng_state(),
               'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        temporary=checkpoint.with_suffix('.tmp');torch.save(saved,temporary);temporary.replace(checkpoint)
    if best_state is None:raise ValueError('student training did not select a checkpoint')
    set_peft_model_state_dict(model, best_state)
    model.save_pretrained(directory);tokenizer.save_pretrained(directory)
    (directory/'training.json').write_text(json.dumps({'validation_history':history,'selected_validation_loss':best,'checkpoint_selection':selection,'training_examples':len(train),'optimizer_updates':scheduler.last_epoch},indent=2)+'\n')
    return StudentBackend(model, tokenizer, encoder_backend, config), {'training_examples':len(train), 'selected_validation_loss':best, 'validation_history':history, 'checkpoint_selection':selection, 'optimizer_updates':scheduler.last_epoch, 'batch_size':batch_size, 'gradient_accumulation':accumulation, 'adapter_path':str(directory)}
