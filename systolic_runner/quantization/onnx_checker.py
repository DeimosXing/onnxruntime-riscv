import onnx
import os
import onnxruntime as ort
import torch
import numpy as np

onnx_modeldir = 'onnx_models'
modeldir = os.path.join(onnx_modeldir, "bert.onnx")
quant_modeldir = os.path.join(onnx_modeldir, "quantized_bert.onnx")
model = onnx.load(modeldir)
quant_model = onnx.load(quant_modeldir)
#print([node.name for node in model.graph.node])
#print("quant model")
#print([node.name for node in quant_model.graph.node])
onnx.checker.check_model(quant_model)

def dummy_check(modeldir):
    ort_session_model = ort.InferenceSession(modeldir)
    #ort_session_quant = ort.InferenceSession(quant_modeldir)
    ipts = ort_session_model.get_inputs()
    print(len(ipts))

    ids = np.array([[1, 2, 3]])
    tok = np.zeros_like(ids)
    att = np.ones_like(ids)
    ort_inputs = {ort_session_model.get_inputs()[0].name: ids,
                    ort_session_model.get_inputs()[1].name: tok,
                    ort_session_model.get_inputs()[2].name: att,
                    ort_session_model.get_inputs()[3].name: ids}
    outputs = ort_session_model.run(None, ort_inputs)
    print(outputs[0])
    print(outputs[1])

dummy_check(quant_modeldir)
