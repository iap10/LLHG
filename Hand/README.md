I will make code for hand and object detection!

---
## Model Operation Plan
we will convert pt file to onnx file to TensorRT file

# ONNX Conversion bash code
yolo export model=best.pt format=onnx

# TensorRT Conversion bash code
trtexec --onnx=best.onnx --saveEngine=best.engine --fp16

---
## Fine-tuned object lists
"Bottle", "Toy", "Computer keyboard", "Pen", "Mobile phone", "Computer mouse", "Tablet computer", "Human hand"
