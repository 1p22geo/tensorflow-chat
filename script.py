import random
with open("./pol.txt", "r") as f:
    lines = f.readlines()

random.shuffle(lines)
sentences = []

with open("./pol2.txt", "w") as f:
    for line in lines:
        en, pol, _ = line.split("\t")
        if en in sentences:
            continue
        sentences.append(en)
        f.write(en + "\t" + pol + "\n")
