I will make code for hand and object detection!

---
# Model Operation Plan
we will convert pt file to onnx file to TensorRT file

### ONNX Conversion bash code
yolo export model=best.pt format=onnx

### TensorRT Conversion bash code
trtexec --onnx=best.onnx --saveEngine=best.engine --fp16

---
# Fine-tuned object lists
"Bottle", "Toy", "Computer keyboard", "Pen", "Mobile phone", "Computer mouse", "Tablet computer", "Human hand" <br/>

## Class ID 

| Class ID | Class Name | 
|------|----------------------------------|
| 0 | Bottle   |
| 1 | Toy  |
| 2 | Computer keyboard | 
| 3 | Pen  |
| 4 | Mobile phone  |
| 5 | Computer mouse  |
| 6 | Tablet computer  |
| 7 | Human hand  |


Class ID	Class Name                <br/>
0	        Bottle                    <br/>
1	        Toy                       <br/>
2	        Computer keyboard         <br/>
3	        Pen                       <br/>
4	        Mobile phone              <br/>
5	        Computer mouse            <br/>
6	        Tablet computer           <br/>
7	        Human hand                <br/>      
