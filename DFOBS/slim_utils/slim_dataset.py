import os
import random
import torch
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore", message=".*promote_options.*")
warnings.filterwarnings("ignore", message=".*precompiled_charsmap.*")

def sample_data(num_samples, seqlen, trainenc):
    trainloader = []
    hist_idx = []
    while True:
        idx = random.randint(0, trainenc.shape[1] - seqlen - 1)
        if idx in hist_idx:
            continue
        hist_idx.append(idx)
        inputs = trainenc[:, idx: idx + seqlen]
        trainloader.append(inputs)
        # print(len(trainloader))
        if len(trainloader) == num_samples:
            break
    return trainloader

def sample_data_v2(num_samples, seqlen, trainenc):
    total_len = num_samples * seqlen
    assert trainenc.shape[1] > total_len
    idx = random.randint(0, trainenc.shape[1] - total_len - 1)
    trainenc = trainenc[:, idx: idx + total_len]
    return trainenc.view(num_samples, seqlen)


def get_wikitext2(num_samples, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset("/home/sumingluo/Dataset/wikitext", 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('/home/sumingluo/Dataset/wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(0)
    trainloader = []
    for _ in range(num_samples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))   
    return trainloader


def get_c4(num_samples, seqlen, tokenizer):
    data_file = '/tao-m6-gpt/chenhai/datasets/c4_train_en/c4-train.00000-of-01024.json'  # replace None for local loading
    if data_file is not None:
        traindata = load_dataset('json', data_files=data_file, split='train')
    else:
        traindata = load_dataset(
            'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    
    cache_data_file = 'data_cache/c4/data.pt'
    traindata = traindata.select(range(200000))
    if not os.path.exists(cache_data_file) or True:
        os.makedirs(os.path.dirname(cache_data_file), exist_ok=True)
        traindata = traindata.map(
            lambda x: {'input_ids': tokenizer.encode(x['text'] + '\n\n', add_special_tokens=False)},
            load_from_cache_file=False,
            num_proc=16,
            desc="Running tokenizer on dataset",
        )
        torch.save(traindata, cache_data_file)
    else:
        traindata = torch.load(cache_data_file)
    trainenc = torch.tensor([x2 for x1 in traindata['input_ids'] for x2 in x1]).unsqueeze(0)
    print('200000 samples are prepared!')
    trainloader = sample_data(num_samples, seqlen, trainenc)
    return trainloader


def get_alpaca(name, num_samples, seqlen, tokenizer):
    if name  == 'alpaca':
        data_file = '/tao-m6-gpt/chenhai/datasets/alpaca-cleaned/alpaca_data_cleaned.json'  # replace None for local loading
        if data_file is not None:
            data = load_dataset('json', data_files=data_file)
        else:
            data = load_dataset('yahma/alpaca-cleaned')
    elif name == 'gpt4_alpaca':
        data_file = '/tao-m6-gpt/chenhai/datasets/alpaca-cleaned/alpaca_gpt4_data.json'  # replace None for local loading
        if data_file is not None:
            data = load_dataset('json', data_files=data_file)
        else:
            data = load_dataset('vicgalle/alpaca-gpt4')
    else:
        raise Exception
    
    train_test_split = data["train"].train_test_split(
        test_size=2000, shuffle=True, seed=42
    )
    traindata = train_test_split["train"]

    def func(x):
        if x['input']:
            return {'input_ids': 
                    tokenizer.encode('\n'.join([x['instruction'], x['input'], x['output']]) + '\n\n', add_special_tokens=False)}
        else:
            return {'input_ids': 
                    tokenizer.encode('\n'.join([x['instruction'], x['output']]) + '\n\n', add_special_tokens=False)}


    cache_data_file = f'data_cache/{name}/data.pt'
    if not os.path.exists(cache_data_file) or True:
        os.makedirs(os.path.dirname(cache_data_file), exist_ok=True)
        traindata = traindata.map(
            func,
            load_from_cache_file=False,
            num_proc=16,
            desc="Running tokenizer on dataset",
        )
        torch.save(traindata, cache_data_file)
    else:
        traindata = torch.load(cache_data_file)
    trainenc = torch.tensor([x2 for x1 in traindata['input_ids'] for x2 in x1]).unsqueeze(0)
    trainloader = sample_data(num_samples, seqlen, trainenc)
    return trainloader


def get_loaders(name, num_samples=128, seqlen=128, tokenizer=None):
    if name == 'wikitext2':
        return get_wikitext2(num_samples, seqlen, tokenizer)
    if name == 'c4':
        return get_c4(num_samples, seqlen, tokenizer)
    if name in ('alpaca', 'gpt4_alpaca'):
        return get_alpaca(name, num_samples, seqlen, tokenizer)
