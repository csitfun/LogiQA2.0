import json

with open('train.txt', 'r') as f:
    file = f.readlines()
    n = 1
    l = 0
    max = 0
    for line in file:
        s = 0
        line = json.loads(line)
        text1 = line['major_premise']
        text2 = line['minor_premise']
        if type(text1) == str:
            l = l + len(text1.split(" "))
            s = s + len(text1.split(" "))
        else:
            for text in text1:
                l = l + len(text.split(" "))
                s = s + len(text.split(" "))
        if type(text2) == str:
            l = l + len(text2.split(" "))
            s = s + len(text2.split(" "))
        else:
            for text in text2:
                l = l + len(text.split(" "))
                s = s + len(text.split(" "))
        if s >= max:
            max = s
        n += 1

    result = l/n
    print(result)
    print(max)