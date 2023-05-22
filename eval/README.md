# Evaluation

## Dependencies

 - Python 3.7
 - You can install the requirements via pip with

```shell
pip install -r requirements.txt
```

## Usage

 - You can run the following command to evaluate our results. 
 - ERLE and [AECMOS](https://github.com/microsoft/AEC-Challenge/tree/main/AECMOS) are computed.

```
# for the final results of ours
python eval.py --input ../results/output/ours \
    --dataset ../results/blind_test_set_interspeech2021 \
    --model_path Run_1657188842_Stage_0.onnx \
    --score_file result.csv \
    --worker 60

# for the results of the coarse-stage
python eval.py --input ../results/output/coarse_stage \
    --dataset ../results/blind_test_set_interspeech2021 \
    --model_path Run_1657188842_Stage_0.onnx \
    --score_file result_coarse_stage.csv \
    --worker 60
```

 - Result

```shell
# ours
st: Mean echo MOS is 4.804, other degradation MOS is 5.0, erle: 66.1, pesq: -1.0
nst: Mean echo MOS is 4.999, other degradation MOS is 4.318, erle: -1.0, pesq: -1.0
dt: Mean echo MOS is 4.693, other degradation MOS is 4.286, erle: -1.0, pesq: -1.0
AECMOS(avg): 4.525

# coarse-stage
st: Mean echo MOS is 4.79, other degradation MOS is 5.0, erle: 64.65, pesq: -1.0
nst: Mean echo MOS is 4.999, other degradation MOS is 4.273, erle: -1.0, pesq: -1.0
dt: Mean echo MOS is 4.545, other degradation MOS is 4.014, erle: -1.0, pesq: -1.0
AECMOS(avg): 4.405
```
