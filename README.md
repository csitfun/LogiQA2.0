# LogiQA2.0
Logiqa2.0 dataset - logical reasoning in MRC and NLI tasks

> This repository contains the datasets and baseline codes for our paper [LogiQA2.0 - An Improved Dataset for Logic Reasoning in Question Answering and Textual Inference Tasks]()
## About
This is the version 2 of the LogiQA dataset, first released as a multi-choice reading comprehension dataset by our previous paper [LogiQA: A Challenge  Dataset for Machine Reading Comprehension with Logical Reasoning](https://arxiv.org/abs/2007.08124). 

The dataset is collected from the [Chinese Civil Service Entrance Examination](chinagwy.org). The dataset is both in Chinese and English (by translation). you can download the version 1 of the LogiQA dataset from [here](https://github.com/lgw863/logiqa-dataset).

To construct LogiQA2.0 dataset, we:
* collect more newly released exam questions and practice questions. There are about 20 provinces in China that hold the exam annually. The exam materials are publicly available on the Internet after the exams. Besides, practice questions are provided by various sources.
* hire professional translators to re-translate the dataset from Chinese to English; verify the labels and annotations with human experts. This program is conducted by [Speechocean](en.speechocean.com), a data annotation service provider. The project is accomplished with the help of Microsoft Research Asia.
* introduce a new NLI task to the dataset. The NLI version of the dataset is converted from the MRC version of the dataset, following previous work such as [Transforming Question Answering Datasets into Natural Language Inference Datasets](https://arxiv.org/abs/1809.02922).

## Datasets
### MRC
The MRC part of LogiQA2.0 dataset can be found in the '/logiqa/DATA/LOGIQA' folder.

'train.txt': train split of the dataset in json lines

'dev.txt': dev split of the dataset in json lines

'test.txt': test split of the dataset in json lines

An example:
```
{"id": 10471, "answer": 0, "text": "The medieval Arabs had many manuscripts of the ancient Greek. When needed, they translate them into Arabic. Medieval Arab philosophers were very interested in Aristotle's Theory of Poetry, which was obviously not shared by Arab poets, because a poet interested in it must want to read Homer's poems. Aristotle himself often quotes Homer's poems. However, Homer's poems were not translated into Arabic until modern times.", "question": "Which of the following options, if true, strongly supports the above argument?", "options": ["Some medieval Arab translators have manuscripts of Homer poems in ancient Greek.", "Aristotle's Theory of Poetry is often quoted and commented by modern Arab poets.", "In Aristotle's Theory of Poetry, most of the content is related to drama, and medieval Arabs also wrote plays and performed them.", "A series of medieval Arab stories, such as Arab Night, are very similar to some parts of Homer's epic."], "type": {"Sufficient Conditional Reasoning": true, "Necessry Condtional Reasoning": true, "Conjunctive Reasoning": true}}
```

### NLI
The NLI part of LogiQA2.0 dataset can be found in the '/logiqa2nli/DATA/QA2NLI' folder.

'train.txt': train split of the dataset in json lines

'dev.txt': dev split of the dataset in json lines

'test.txt': test split of the dataset in json lines

An example:
```
{"label": "not entailed", "major_premise": ["Among the employees of a software company, there are three Cantonese, one Beijinger and three northerners"], "minor_premise": " Four are only responsible for software development and two are only responsible for product sales", "conclusion": "There may be at least 7 people and at most 12 people."}
```

## Baseline Guidance
### Requirements
* python 3.5+
* pytorch 1.0+
* transformers 2.8.0

We recommend to use conda to manage virtual environments:
'''
conda env update --name logiqa --file requirements.yml
'''
## Results
## How to Cite
## Licence
## Acknowledgment
