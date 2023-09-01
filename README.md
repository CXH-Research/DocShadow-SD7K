# High-Resolution Document Shadow Removal

<b><a href='https://arxiv.org/abs/2308.14221'>High-Resolution Document Shadow Removal via A Large-Scale Real-World Dataset and A Frequency-Aware Shadow Erasing Net.</a> </b>
<div>
<span class="author-block">
  <a href='https://zinuoli.github.io/'>Zinuo Li</a><sup> 👨‍💻‍ </sup>
</span>,
  <span class="author-block">
    <a href='https://cxh.netlify.app/'> Xuhang Chen</a><sup> 👨‍💻‍ </sup>
  </span>,
  <span class="author-block">
    <a href="https://www.cis.um.edu.mo/~cmpun/" target="_blank">Chi-Man Pun</a><sup> 📮</sup>
  </span> and
  <span class="author-block">
  <a href="http://vinthony.github.io/" target="_blank">Xiaodong Cun</a><sup> 📮</sup>
</span>
  ( 👨‍💻‍ Equal contributions, 📮 Corresponding )
</div>

<b>University of Macau</b>

In <b>_International Conference on Computer Vision 2023 (ICCV 2023)_</b>


[Paper](https://arxiv.org/abs/2308.14221) | [Website](https://cxh-research.github.io/DocShadow-SD7K/) | [Dataset SD7K (Baidu)](https://pan.baidu.com/s/1PgJ3cPR3OYO7gwF1o0DgDg?pwd=72aq) | [Dataset SD7K (OneDrive)](https://monashuni-my.sharepoint.com/:f:/g/personal/zlii0362_student_monash_edu/EoiaDzQYCplJv0Tfvzj2nKcBquHcFUQKLXCeX0pI8Arjyw?e=fmJrDK)
---
<img src="./teaser/high.png"/>

# Dataset
<img src="./teaser/Dataset.png"/>
<img src="./teaser/Comparison.png"/ width="100%">
SD7K is a large-scale high-resoulution dataset that covers a large number of document shadow images, and it is the only dataset that satisfies all important data features about document shadow currently.

# Usage
### Installation
```
git clone https://github.com/CXH-Research/DocShadow-SD7K.git
cd DocShadow-SD7K
pip install -r requirements.txt
```

### Training
Please first specify TRAIN_DIR, VAL_DIR and SAVE_DIR in the section TRAINING in `config.yml`.

For single GPU training:
```
python train.py
```
For multiple GPUs training:
```
accelerate config
accelerate launch train.py
```
If you have difficulties with the usage of `accelerate`, please refer to <a href="https://github.com/huggingface/accelerate">Accelerate</a>.

### Inference
Please first specify TRAIN_DIR, VAL_DIR and SAVE_DIR in section TESTING in `config.yml`.
```
python infer.py
```
If you need pre-trained models, please download <a href="https://pan.baidu.com/s/1qOSony6HbpZr_S-cKqdfaA?pwd=sd7k">here</a>.

<p>For the results of all baselines experiments, please refer <a href="https://pan.baidu.com/s/1gGR-gDbU2O1clrEW4xpAzg?pwd=z7nj">here</a>.</p>

#### Acknowledgments
If you find our work helpful for your research, please cite:
```bib
@article{docshadow_sd7k,
  title={High-Resolution Document Shadow Removal via A Large-Scale Real-World Dataset and A Frequency-Aware Shadow Erasing Net},
  author={Li, Zinuo and Chen, Xuhang and Pun, Chi-Man and Cun, Xiaodong},
  journal={arXiv preprint arXiv:2308.14221},
  year={2023}
}
```


