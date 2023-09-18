from transformers import AdamW, AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import json
import torch
from tqdm import tqdm
import math
import functools
from time import time
import gdown
import os


class ELI5DatasetS2S(Dataset):
    """Dataset class for training model

    Args:
        Dataset (Dataset): Inherit Dataset class
    """
    def __init__(
        self,
        data_array,
    ):
        self.data = data_array

    def __len__(self):
        return len(self.data)

    def append(self, question_doc, answer):
        self.data.append([question_doc, answer])

    def __getitem__(self, idx):
        return (self.data[idx][0], self.data[idx][1])
    

class ArgumentsS2S():
    """Class for records model training arguments.
    """
    def __init__(self,
                batch_size = 8,
                backward_freq = 16,
                max_length = 1024,
                print_freq = 5,
                model_save_name = "models/eli5_bart_model",
                learning_rate = 2e-4,
                num_epochs = 3,
                device = 'cuda:0',
                save_freq = 5000,
                continue_training = {"continue": False,
                                    "epoch": 0,
                                    "step": 15000},
                save_logs_path = 'logs.json'):
        
        self.batch_size = batch_size
        self.backward_freq = backward_freq
        self.max_length = max_length
        self.print_freq = print_freq
        self.model_save_name = model_save_name
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device
        self.save_freq = save_freq
        self.continue_training = continue_training
        self.save_logs_path = save_logs_path


def check_dataset_folder():
    """Check if datasets exists, if not, download it."""
    data_path = "data"
    if not ('ELI5.jsonl' in os.listdir(data_path)):
        print("Download ELI5.jsonl from GDrive...")
        url = 'https://drive.google.com/uc?id=19Mb4ZoUzt_6Aa6is4D4_8Y_25eF8Xsba'
        output = data_path + "\ELI5.jsonl"
        gdown.download(url, output, quiet=False)
    if not ('ELI5_val.jsonl' in os.listdir(data_path)):
        print("Download ELI5_val.jsonl from GDrive...")
        url = 'https://drive.google.com/uc?id=13XsUN3gp5N2FbQ4rCmBFSdoieNyebyYm'
        output = data_path + "\ELI5_val.jsonl"
        gdown.download(url, output, quiet=False)
        


def save_logs(path, logs):
    """Saving training logs into json file

    Args:
        path (string): path to save logs
        logs (dict): dictionary of train and eval loss
    """
    try:
        f = open(path, 'x')
    except:
        pass

    with open(path, 'a') as f:
        json.dump(logs, f)


def make_qa_s2s_batch(qa_list, tokenizer, max_len=64, max_a_len=360, device="cuda:0"):
    """Making input batch for s2s model

    Args:
        qa_list (list): list of concatenated question-documents and answer
        tokenizer (AutoTokenizer): corresponding tokenizer is used for model
        max_len (int, optional): max length. Defaults to 64.
        max_a_len (int, optional): Max answer length. Defaults to 360.
        device (str, optional): select device cpu/gpu. Defaults to "cuda:0".

    Returns:
        dict: dictionary for inputs of the model
    """
    q_ls = [q for q, a in qa_list]
    a_ls = [a for q, a in qa_list]
    q_toks = tokenizer.batch_encode_plus(q_ls, max_length=max_len, pad_to_max_length=True)
    q_ids, q_mask = (
        torch.LongTensor(q_toks["input_ids"]).to(device),
        torch.LongTensor(q_toks["attention_mask"]).to(device),
    )
    a_toks = tokenizer.batch_encode_plus(a_ls, max_length=min(max_len, max_a_len), pad_to_max_length=True)
    a_ids, a_mask = (
        torch.LongTensor(a_toks["input_ids"]).to(device),
        torch.LongTensor(a_toks["attention_mask"]).to(device),
    )
    lm_labels = a_ids[:, 1:].contiguous().clone()
    lm_labels[a_mask[:, 1:].contiguous() == 0] = -100
    model_inputs = {
        "input_ids": q_ids,
        "attention_mask": q_mask,
        "decoder_input_ids": a_ids[:, :-1].contiguous(),
        "lm_labels": lm_labels,
    }

    return model_inputs


def qa_s2s_generate(
    question_doc,
    qa_s2s_model,
    qa_s2s_tokenizer,
    num_answers=1,
    num_beams=None,
    min_len=64,
    max_len=256,
    do_sample=False,
    temp=1.0,
    top_p=None,
    top_k=None,
    max_input_length=512,
    device="cuda:0",
):
    """Generate answer from s2s model with inputs of concatenated question-documents,
    qa_s2s_model and qa_s2s_tokenizer.

    Args:
        question_doc (str): _description_
        qa_s2s_model (AutoModelForSeq2SeqLM): Sentence to sentence model
        qa_s2s_tokenizer (AutoTokenizer): Corresponding tokenizer
        num_answers (int, optional): Number of answer. Defaults to 1.
        num_beams (_type_, optional): Number of beams. Defaults to None.
        min_len (int, optional): The minimum length of the answer. Defaults to 64.
        max_len (int, optional): The maximum length of the answer. Defaults to 256.
        do_sample (bool, optional): Whether to use sampling. Defaults to False.
        temp (float, optional): The temperature of sampling. Defaults to 1.0.
        top_p (_type_, optional): The top p of sampling. Defaults to None.
        top_k (_type_, optional): The top k of sampling. Defaults to None.
        max_input_length (int, optional): The maximum length of the input. Defaults to 512.
        device (str, optional): The device to use. Defaults to "cuda:0".

    Returns:
        list: The generated answers.
    """
    model_inputs = make_qa_s2s_batch([(question_doc, "A")], qa_s2s_tokenizer, max_input_length, device=device,)
    n_beams = num_answers if num_beams is None else max(num_beams, num_answers)
    generated_ids = qa_s2s_model.generate(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        min_length=min_len,
        max_length=max_len,
        do_sample=do_sample,
        early_stopping=True,
        num_beams=1 if do_sample else n_beams,
        temperature=temp,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=qa_s2s_tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        num_return_sequences=num_answers,
        decoder_start_token_id=qa_s2s_tokenizer.bos_token_id,
    )

    return [qa_s2s_tokenizer.decode(ans_ids, skip_special_tokens=True).strip() for ans_ids in generated_ids]


def make_qa_s2s_model(model_name="yjernite/bart_eli5", from_file=None, device="cuda:0"):
    """Initialize s2s model base on model_name.

    Args:
        model_name (str, optional): Name of pretrained model on Huggingface. Defaults to "yjernite/bart_eli5".
        from_file (str, optional): Path to local model file. Defaults to None.
        device (str, optional): Choose which device to use cpu/gpu. Defaults to "cuda:0".

    Returns:
        AutoTokenizer, AutoModelForSeq2SeqLM: corresponding tokenizer and model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    if from_file is not None:
        param_dict = torch.load(from_file)  # has model weights, optimizer, and scheduler states
        
        model.load_state_dict(param_dict["model"])

    return tokenizer, model


def train_qa_s2s_epoch(model, dataset, tokenizer, optimizer, scheduler, args, e=0, curriculum=False):
    """Perform model training in an epoch

    Args:
        model (AutoModelForSeq2SeqLM): Model to be trained
        dataset (ELI5DatasetS2S): Training data
        tokenizer (AutoTokenizer): Tokenizer for the model
        optimizer (optimizer): The optimizer for training model
        scheduler (scheduler): The scheduler for training model
        args (ArgumentsS2S): Training arguments
        e (int, optional): Current epoch. Defaults to 0.
        curriculum (bool, optional): whether sample the data . Defaults to False.
    """
    model.train()
    # make iterator
    if curriculum:
        train_sampler = SequentialSampler(dataset)
    else:
        train_sampler = RandomSampler(dataset)
    model_collate_fn = functools.partial(
        make_qa_s2s_batch, tokenizer=tokenizer, max_len=args.max_length, device=args.device
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=model_collate_fn)
    epoch_iterator = tqdm(data_loader, desc="Iteration", disable=True)
    # accumulate loss since last print
    loc_steps = 0
    loc_loss = 0.0
    st_time = time()
    for step, batch_inputs in enumerate(epoch_iterator):
        if not (args.continute_training['continue'] and args.continute_training['epoch'] <= e and args.continute_training['step'] <= step):
            # print(batch_inputs)
            batch_inputs['labels'] = batch_inputs.pop('lm_labels')
            pre_loss = model(**batch_inputs)[0]
            # print(pre_loss)
            loss = pre_loss.sum()# / pre_loss.shape[0]
            loss.backward()
            # optimizer
            if step % args.backward_freq == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            # some printing within the epoch
            loc_loss += loss.item()
            loc_steps += 1
            if step % args.print_freq == 0 or step == 1:
                print(
                    "{:2d} {:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}".format(
                        e, step, len(dataset) // args.batch_size, loc_loss / loc_steps, time() - st_time,
                    )
                )
                loc_loss = 0
                loc_steps = 0
            
            if step % args.save_freq == 0:
                m_save_dict = {
                # "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                }
                model.save_pretrained(args.model_save_name)
                print("Saving model {}".format(args.model_save_name))
                torch.save(m_save_dict, "{}_{}_{}.pth".format(args.model_save_name, e, step))


def eval_qa_s2s_epoch(model, dataset, tokenizer, args):
    """Perform evaluation of the model after an epoch

    Args:
        model (AutoModelForSeq2SeqLM): Model to be trained
        dataset (ELI5DatasetS2S): Training data
        tokenizer (AutoTokenizer): Tokenizer for the model
        args (ArgumentsS2S): Training arguments
    """
    model.eval()
    # make iterator
    train_sampler = SequentialSampler(dataset)
    model_collate_fn = functools.partial(
        make_qa_s2s_batch, tokenizer=tokenizer, max_len=args.max_length, device=args.device
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=model_collate_fn)
    epoch_iterator = tqdm(data_loader, desc="Iteration", disable=True)
    # accumulate loss since last print
    loc_steps = 0
    loc_loss = 0.0
    st_time = time()
    with torch.no_grad():
        for step, batch_inputs in enumerate(epoch_iterator):
            batch_inputs['labels'] = batch_inputs.pop('lm_labels')
            pre_loss = model(**batch_inputs)[0]
            loss = pre_loss.sum() #/ pre_loss.shape[0]
            loc_loss += loss.item()
            loc_steps += 1
            if step % args.print_freq == 0:
                print(
                    "{:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}".format(
                        step, len(dataset) // args.batch_size, loc_loss / loc_steps, time() - st_time,
                    )
                )
    print("Total \t L: {:.3f} \t -- {:.3f}".format(loc_loss / loc_steps, time() - st_time,))


def train_qa_s2s(qa_s2s_model, qa_s2s_tokenizer, s2s_train_dset, s2s_valid_dset, s2s_args):
    """Perform training model

    Args:
        qa_s2s_model (AutoModelForSeq2SeqLM): model to be trained
        qa_s2s_tokenizer (AutoTokenizer): correspongding tokenizer for model
        s2s_train_dset (ELI5DatasetS2S): training data
        s2s_valid_dset (ELI5DatasetS2S): validating data
        s2s_args (ArgumentsS2S): training arguments
    """
    s2s_optimizer = AdamW(qa_s2s_model.parameters(), lr=s2s_args.learning_rate, eps=1e-8)
    s2s_scheduler = get_linear_schedule_with_warmup(
        s2s_optimizer,
        num_warmup_steps=400,
        num_training_steps=(s2s_args.num_epochs + 1) * math.ceil(len(s2s_train_dset) / s2s_args.batch_size),
    )
    
    if s2s_args.continue_training["continue"]:
        m_dict = torch.load(s2s_args.model_save_name + "_" + 
                            str(s2s_args.continue_training["epoch"]) + "_" +
                            str(s2s_args.continue_training["step"]) + ".pth")
        s2s_optimizer.load_state_dict(m_dict['optimizer'])
        s2s_scheduler.load_state_dict(m_dict['scheduler'])

    
    for e in range(s2s_args.num_epochs):      
        train_qa_s2s_epoch(
            qa_s2s_model,
            s2s_train_dset,
            qa_s2s_tokenizer,
            s2s_optimizer,
            s2s_scheduler,
            s2s_args,
            e,
            curriculum=(e == 0),
        )
        m_save_dict = {
            # "model": qa_s2s_model.state_dict(),
            "optimizer": s2s_optimizer.state_dict(),
            "scheduler": s2s_scheduler.state_dict(),
        }
        print("Saving model {}".format(s2s_args.model_save_name))
        eval_qa_s2s_epoch(qa_s2s_model, s2s_valid_dset, qa_s2s_tokenizer, s2s_args)
        torch.save(m_save_dict, "{}_{}.pth".format(s2s_args.model_save_name, e))
        qa_s2s_model.save_pretrained(s2s_args.model_save_name)