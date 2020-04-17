import onnx
from quantize import quantize, QuantizationMode
import os

onnx_savedir = "onnx_models"
quant_mode = QuantizationMode.QLinearOps
onnx_model_name = 'bert.onnx'
onnx_savename = 'quantized_bert_qlinear.onnx'
def onnx_post_quantization(onnx_model_name, quant_mode, onnx_savedir, onnx_savename):
    onnx_modeldir = os.path.join(onnx_savedir, onnx_model_name)
    model = onnx.load(onnx_modeldir)
    quantized_model = quantize(model, static=False, quantization_mode=quant_mode)
    onnx.save(quantized_model,  os.path.join(onnx_savedir,onnx_savename))
onnx_post_quantization(onnx_model_name, quant_mode, onnx_savedir, onnx_savename)
