from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from utils import *

check_dataset_folder()

# Load training set
path = "data\ELI5.jsonl"
f = open(path, "r")

train_data = ELI5DatasetS2S([])

for id, line in enumerate(f):
  print(id)
  data = json.loads(line)
  # print(data)

  question = data['question']
  doc = " ".join(data['ctxs'])
  answer = "".join(data['answers'])

  question_doc = "question: {} context: {}".format(question, doc)

  train_data.append(question_doc, answer)
  
f.close()


# Load validation set
path = "data\ELI5_val.jsonl"
f = open(path, "r")

val_data = ELI5DatasetS2S([])

for id, line in enumerate(f):
  print(id)
  data = json.loads(line)
  texts = []
  for text in data['ctxs']:
    texts.append(text[0])

  question = data['question']
  doc = " ".join(texts)
  answer = "".join(data['answers'])

  question_doc = "question: {} context: {}".format(question, doc)

  val_data.append(question_doc, answer)

f.close()

# Training
s2s_args = ArgumentsS2S()

qa_s2s_tokenizer, qa_s2s_model = make_qa_s2s_model(
    model_name="yjernite/bart_eli5",
    from_file=None,
    device=s2s_args.device
)

print(type(qa_s2s_model))
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=32, lora_alpha=32, lora_dropout=0.1
)

qa_s2s_model = get_peft_model(qa_s2s_model, peft_config)
qa_s2s_model.print_trainable_parameters()

train_qa_s2s(qa_s2s_model, qa_s2s_tokenizer, train_data, val_data, s2s_args)