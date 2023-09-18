from utils import *
from peft import PeftConfig, PeftModel
from nltk import PorterStemmer
from rouge import Rouge
import spacy
from spacy.tokenizer import Tokenizer

s2s_args = ArgumentsS2S()
model_id = "models/eli5_bart_model"
checkpoint = "yjernite/bart_eli5"

config = PeftConfig.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint
).to(s2s_args.device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

inference_model = PeftModel.from_pretrained(model=model, model_id=model_id)
inference_model.print_trainable_parameters()

predicted = []
reference = []

try:
  f.close()
except:
  print("No file to close")

path = "data/ELI5_val.jsonl"
f = open(path, "r")

val_data = ELI5DatasetS2S([])

for id, line in enumerate(f):
  if id == 50:
    break
  # print(id)
  data = json.loads(line)
  # print(data)

  question = data['question']
  doc = '. '.join(map(str, data['ctxs']))
  answer_true = '. '.join(map(str, data['answers']))

  question_doc = "question: {} context: {}".format(question, doc)
  answer_pred = qa_s2s_generate(
            question_doc, inference_model, tokenizer,
            num_answers=1,
            num_beams=8,
            min_len=96,
            max_len=256,
            max_input_length=512,
            device="cuda:0"
    )[0]
  predicted += [answer_pred]
  reference += [answer_true]

  stemmer = PorterStemmer()
rouge = Rouge()
nlp = spacy.load("en_core_web_sm")
tokenizer = Tokenizer(nlp.vocab)

def compute_rouge_eli5(compare_list):
    preds = [" ".join([stemmer.stem(str(w))
                       for w in tokenizer(pred)])
             for gold, pred in compare_list]
    golds = [" ".join([stemmer.stem(str(w))
                       for w in tokenizer(gold)])
             for gold, pred in compare_list]
    scores = rouge.get_scores(preds, golds, avg=True)
    return scores


compare_list = [(g, p) for p, g in zip(predicted, reference)]
scores = compute_rouge_eli5(compare_list)
df = pd.DataFrame({
    'rouge1': [scores['rouge-1']['p'], scores['rouge-1']['r'], scores['rouge-1']['f']],
    'rouge2': [scores['rouge-2']['p'], scores['rouge-2']['r'], scores['rouge-2']['f']],
    'rougeL': [scores['rouge-l']['p'], scores['rouge-l']['r'], scores['rouge-l']['f']],
}, index=[ 'P', 'R', 'F'])
print(df.style.format({'rouge1': "{:.4f}", 'rouge2': "{:.4f}", 'rougeL': "{:.4f}"}))