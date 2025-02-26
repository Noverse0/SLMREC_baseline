# SLMRec
The code for paper "SLMRec: Empowering Small Language Models for Sequential Recommendation".


# SLMRec: Empowering Small Language Models for Sequential Recommendation
This repo presents the implementation of the **SLMRec** 

We present **Empowering Small Language Models for Sequential Recommendation**, abbreviated as **SLMRec**. This paper presents an initial attempt to reassess the need for LLMs in sequential recommendation.

## Key Features of SLMRec🔑

- **Motivational Experiments**: To explore the reasons for the significant improvement of LLMRec methods, we conduct a series of experiments on large-scale industry datasets to investigate the effects of reducing the number of parameters during the training and inference stages on overall performance. From the empirical results, we found some profound insights that the improvement of the rise of the model parameters is not consistent. Meanwhile, it reveals that some layers of LLMs are redundant in the recommendation task.

<div align=center><img src="pic/screenshot.jpg" width="100%" height="100%" /></div>

- **Simple but Effective Method**: Motivated by these findings, we empower small language models for the sequential recommendation, named SLMRec. We adopt the vanilla knowledge distillation approaches to align the representation knowledge. Moreover, multiple supervision signals are crafted to steer the student model toward acquiring task-aware knowledge within its hidden representations. Extensive experiments have shown that SLMRec, with a model size under 1 billion parameters, not only achieves performance comparable to baselines using LLMs with over 7 billion parameters but also delivers up to 6.6x faster training and 8.0x faster inference compared to LLM-based recommendation models.


<div align=center><img src="pic/framework.jpg" width="100%" height="100%" /></div>

## Getting Started 🚀

1. Clone the repository:
```bash
git clone https://github.com/WujiangXu/AgenticMemory.git
cd AgenticMemory

2. Install dependencies:
Option 1: Using venv (Python virtual environment)
```bash
# Create and activate virtual environment
python3.10 -m venv slmrec_env
source slmrec_env/bin/activate  # Linux/Mac
slmrec_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

Option 2: Using Conda
```bash
# Create and activate conda environment
conda create -n slmrec_env python=3.9
conda activate slmrec_env
# Install dependencies
pip install -r requirements.txt
```


### Dataset🧑‍💻

#### Use Pre-processed Dataset

1. Download [prepared dataset](https://drive.google.com/drive/folders/1cambs_D6OpiWJE8ms5pdxcmoVfiOOlKg?usp=sharing).

#### Process by your own

1. Download [raw data](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/).
2. Run the datasets/process.ipynb

### Training Processes
1. Train SR model to obtain the pretrained embedding layer.
```python   
python train_sr_trad.py
```

2. Save weights of embedding layer into local pkl file.
```python   
python extract_emb.py
```

3.1 Download pre-trained LLaMA model.


3.2 Train a teacher model.
```bash
bash run_finetune.sh
```

4. Train a student model via the knowledge distillation.
```python
python distill.py
```

### Hyperparameters

Please refer to the appendix in our paper.

## Citation

If you found the codes useful, please cite our paper.

      @inproceedings{xu2025slmrec,
      title = {SLMRec: Distilling Large Language Models into Small for Sequential Recommendation},
      author = {Wujiang Xu, Qitian Wu, Zujie Liang, Jiaojiao Han, Xuying Ning, Yunxiao Shi, Wenfang Lin, Yongfeng Zhang},
      booktitle = {International Conference on Learning Representations (ICLR 2025)},
      year = {2025}
      }


## Contact us 
Please feel free to contact us with the email to W. Xu "wujiang dot xu at rutgers dot edu" or "swustimp at gmail dot com".
