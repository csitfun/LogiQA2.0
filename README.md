# LogiQA2.0
Logiqa2.0 dataset - logical reasoning in MRC and NLI tasks

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

> This repository contains the datasets and baseline codes for our paper [LogiQA2.0 - An Improved Dataset for Logic Reasoning in Question Answering and Textual Inference](https://ieeexplore.ieee.org/abstract/document/10174688)

## How to cite
```
@ARTICLE{10174688,
  author={Liu, Hanmeng and Liu, Jian and Cui, Leyang and Teng, Zhiyang and Duan, Nan and Zhou, Ming and Zhang, Yue},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={LogiQA 2.0—An Improved Dataset for Logical Reasoning in Natural Language Understanding}, 
  year={2023},
  volume={31},
  number={},
  pages={2947-2962},
  doi={10.1109/TASLP.2023.3293046}}

```
## About
This is the version 2 of the LogiQA dataset, first released as a multi-choice reading comprehension dataset by our previous paper [LogiQA: A Challenge  Dataset for Machine Reading Comprehension with Logical Reasoning](https://arxiv.org/abs/2007.08124). 

The dataset is collected from the [Chinese Civil Service Entrance Examination](chinagwy.org). The dataset is both in Chinese and English (by translation). you can download the version 1 of the LogiQA dataset from [here](https://github.com/lgw863/logiqa-dataset).

To construct LogiQA2.0 dataset, we:
* collect more newly released exam questions and practice questions. There are about 20 provinces in China that hold the exam annually. The exam materials are publicly available on the Internet after the exams. Besides, practice questions are provided by various sources.
* hire professional translators to re-translate the dataset from Chinese to English; verify the labels and annotations with human experts. This program is conducted by [Speechocean](en.speechocean.com), a data annotation service provider. The project is accomplished with the help of Microsoft Research Asia.
* introduce a new NLI task to the dataset. The NLI version of the dataset is converted from the MRC version of the dataset, following previous work such as [Transforming Question Answering Datasets into Natural Language Inference Datasets](https://arxiv.org/abs/1809.02922).

## Datasets
### MRC
The MRC part of LogiQA2.0 dataset can be found in the `/logiqa/DATA/LOGIQA` folder.

`train.txt`: train split of the dataset in json lines.

`dev.txt`: dev split of the dataset in json lines.

`test.txt`: test split of the  dataset in json lines.

`train_zh.txt`: train split of the Chinese version of dataset in json lines.

`dev_zh.txt`: dev split of the Chinese version of dataset in json lines.

`test_zh.txt`: test split of the Chinese version of dataset in json lines.

`train_fol.zip` is the training data with AMR and FOL annotations. The file is too big so we compressed it.

`dev_fol.jsonl` is the dev data with AMR and FOL annotations.

`test_fol.jsonl` is the test data with AMR and FOL annotations.


An example:
```
{"id": 10471, "answer": 0, "text": "The medieval Arabs had many manuscripts of the ancient Greek. When needed, they translate them into Arabic. Medieval Arab philosophers were very interested in Aristotle's Theory of Poetry, which was obviously not shared by Arab poets, because a poet interested in it must want to read Homer's poems. Aristotle himself often quotes Homer's poems. However, Homer's poems were not translated into Arabic until modern times.", "question": "Which of the following options, if true, strongly supports the above argument?", "options": ["Some medieval Arab translators have manuscripts of Homer poems in ancient Greek.", "Aristotle's Theory of Poetry is often quoted and commented by modern Arab poets.", "In Aristotle's Theory of Poetry, most of the content is related to drama, and medieval Arabs also wrote plays and performed them.", "A series of medieval Arab stories, such as Arab Night, are very similar to some parts of Homer's epic."], "type": {"Sufficient Conditional Reasoning": true, "Necessry Condtional Reasoning": true, "Conjunctive Reasoning": true}}
```
An example of the Chinese dataset:
```
{"id": 8018, "answer": 0, "text": "常春藤通常指美国东部的八所大学。常春藤一词一直以来是美国名校的代名词，这八所大学不仅历史悠久，治学严谨，而且教学质量极高。这些学校的毕业生大多成为社会精英，他们中的多数人年薪超过20万美元，有很多政界领袖来自常春藤，更有为数众多的科学家毕业于长春藤。", "question": "根据以上条件，下面那个选项一定为真:", "options": ["A.有些社会精英年薪超过20万美金", "B.有些政界领袖年薪不足20万美元", "C.有些科学家年薪超过20万美元", "D.有些政界领袖是社会精英"]}
```

### NLI
The NLI part of LogiQA2.0 dataset can be found in the `/logiqa2nli/DATA/QA2NLI` folder.

`train.txt`: train split of the dataset in json lines

`dev.txt`: dev split of the dataset in json lines

`test.txt`: test split of the dataset in json lines

An example:
```
{"label": "not entailed", "major_premise": ["Among the employees of a software company, there are three Cantonese, one Beijinger and three northerners"], "minor_premise": " Four are only responsible for software development and two are only responsible for product sales", "conclusion": "There may be at least 7 people and at most 12 people."}
```
## Annotations
The translation and annotation work is outsourced to [Speechocean](en.speechocean.com), the project fund is provided by Microsoft Research Asia
### Translation

| Final Report |  |
| --- | --- |
| provider | Speechocean |
| Project Duration | 2021/10/20-2021/12/3 |
| Actual Working Hour | 667 hours |
| Cost | 45000 RMB |
 
Translation style/method:

1. Maintain a unified style, and the translated English questions need to inherit the logic of the original questions;

2. The pronoun in the question need to be unique, the translation needs to be unique and consistent without ambiguity;

3. The translated English conforms to the form of a proper question, that is, it is a clear question from the perspective of the respondent;

### Label consistency
The label credibility is mannually verified after the translation was done to maintain the truthfulness of the original text. 3 workers run a consistency test on each example, if 2 or more workers give different answer compared to the original answer, the translation would be redone to guareentee the label is correct.

### Additional annotations
Reasoning types of each question is assigned by a total of 5 workers, each of them corresponds to one reasoning type. We give the description of reasoning types (which can be found in our paper) to the workers. The reasoning types of each question is a collection of 5 workers' decision.
## Baseline Guidance
### Requirements
* python 3.6+
* pytorch 1.0+
* transformers 2.4.1
* sklearn
* tqdm
* tensorboardX

We recommend to use conda to manage virtual environments:

```
conda env update --name logiqa --file requirements.yml
```
### Logiqa
The folder `logiqa` contains both the code and data to run baseline experiments of LogiQA2.0 MRC.

To fine-tune the dataset, type following command from the terminal in your :computer:
```
bash logiqa.sh
```
### Logiqa2nli
The folder `logiqa2nli` contains both the code and data to run baseline experiments of LogiQA2.0 NLI.

To fine-tune the dataset, type following command from the terminal in your :computer:
```
bash qa2nli.sh
```
Note: `./scripts` contains the scripts for running other NLI benchmarks.

## How to Cite
## Acknowledgment
We appreciate the suggestions and critical questions from the reviewers.
