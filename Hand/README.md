I will make code for hand and object detection!  

## Basic Class Construction  
"YoloTRT.py" code is basic class for object detection(for fine-tuned classes) using TensorRT file.

---
## Model Operation Plan
we will convert pt file to onnx file to TensorRT file

### ONNX Conversion bash code
```
yolo export model=best.pt format=onnx
```

### TensorRT Conversion bash code
```
trtexec --onnx=best.onnx --saveEngine=best.engine --fp16
```
---
## Fine-tuned object lists
"Bottle", "Toy", "Computer keyboard", "Pen", "Mobile phone", "Computer mouse", "Tablet computer", "Human hand" <br/>

### Class ID 

| Class ID | Class Name | 
|------|----------------------------------|
| 0 | Bottle   |
| 1 | Toy  |
| 2 | Keyboard | 
| 3 | Pen  |
| 4 | phone  |
| 5 | mouse  |
| 6 | Tablet |
| 7 | paper  |
| 8 | book  |
| 9 | tumbler |
| 10 | hand  |

---
## Study & Play Judgement Standard
| Hold | State | Result | Output |
|------|----------------------------------|---------------|----------------|
| Bottle | Moderate | Judge based on Face Result | 0 |
| Toy  | Play | Don't care Face Result and Return Play Signal | 1 |
| Keyboard | Moderate | Judge based on Face Result | 0 |
| Pen  | Study | Judge based on Face Result(Priority on Study) | 2 |
| phone  | Play | Don't care Face Result and Return Play Signal | 1 |
| mouse  | Moderate | Judge based on Face Result | 0 |
| Tablet  | Moderate | Judge based on Face Result | 0 |
| paper  | Study | Judge based on Face Result(Priority on Study) | 2 |
| book  | Study | Judge based on Face Result(Priority on Study) | 2 |
| tumbler  | Moderate | Judge based on Face Result | 0 |
| Human hand(not hold)  | Moderate | Judge based on Face Result | 0 |

---
## Real Application Code
Real Application codes are included in func Folder
