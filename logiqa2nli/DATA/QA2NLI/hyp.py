import json

with open('train.txt', 'r') as f:
    file = f.readlines()
    n = 1
    l = 0
    max = 0
    for line in file:
        s = 0
        line = json.loads(line)
        text = line['conclusion']
        l = l + len(text.split(' '))
        s = s + len(text.split(' '))
        n += 1
        if s >= max:
            max = s

    result = l/n
    print(result)
    print(max)