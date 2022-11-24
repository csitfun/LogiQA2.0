import json

with open('test.txt', 'r') as f:
    file = f.readlines()
    n = 1
    l = 0
    for line in file:
        line = json.loads(line)
        text1 = line['major_premise']
        text2 = line['minor_premise']
        if type(text1) == str:
            l = l + len(text1.split(" "))
        else:
            for text in text1:
                l = l + len(text.split(" "))
        if type(text2) == str:
            l = l + len(text2.split(" "))
        else:
            for text in text2:
                l = l + len(text.split(" "))

        n += 1

    result = l/n
    print(result)