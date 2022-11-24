import json
with open('test.txt') as f:
    file = f.readlines()
    n = 0
    l = 0
    for line in file:
        d = json.loads(line)
        label = d['answer']
        text = d['text']
        options = d['options']
        text_vocab = set(text.split(' '))
        ratio = []
        for option in options:
            option_vocab = set(option.split(' '))
            intersection = text_vocab.intersection(option_vocab)
            ratio.append(len(intersection)/len(text_vocab))
        value_prev = 0
        for value in ratio:
            if value >= value_prev:
                value_prev = value
        index = ratio.index(value_prev)
        if index == label:
            l += 1
        n += 1
    result = l/n
    print(result)
