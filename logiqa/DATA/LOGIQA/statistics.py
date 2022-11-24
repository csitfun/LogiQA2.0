import json

with open('test.txt', 'r') as f:
    file = f.readlines()
    n = 1
    l = 0
    max = 0
    for line in file:

        line = json.loads(line)
        text = line['options']
        for option in text:
            s = 0
            l = l + len(option.split(" "))
            s = s + len(option.split(" "))
            n += 1
        if s >= max:
            max = s
    result = l/n
    print(result)
    print(max)