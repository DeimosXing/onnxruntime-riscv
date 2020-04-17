#import torch
#from nlp_architect.models.transformers.quantized_bert import QuantizedBertForSequenceClassification
#from nlp_architect.data.glue_tasks import get_glue_task
#from transformers import BertForSequenceClassification

import os
from transformers import BertModel
import torch

model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)
model.eval()
ids = torch.LongTensor([[1, 2, 3]])
tok = torch.zeros_like(ids)
att = torch.ones_like(ids)
#model = torch.jit.script(model, (ids, tok, att, ids))
onnx_savedir = "onnx_models"
torch.onnx.export(model, (ids, tok, att, ids), os.path.join(onnx_savedir, "bert.onnx"), 
                    opset_version=11,
                    verbose=True)
'''
#model = QuantizedBertForSequenceClassification.from_pretrained('./mrpc-8bit', from_8bit=True)
model = BertForSequenceClassification(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
    )
device = torch.device("cpu")
model.to(device)
model.eval()

#script_model = torch.jit.script(model)
task = get_glue_task("mrpc", data_dir="glue_data/MRPC")
examples = task.get_test_examples()
model.inference(examples, 128, 8, evaluate = False)
'''
