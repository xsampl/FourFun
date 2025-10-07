letters = 'abcdefghijklmnopqrstuvwxyz'
com = []
names = []

for letter in letters:
    if letter == 'b' or letter == 'i' or letter == 'e' or letter == 's' or letter == 'p' or letter == 'e' or letter == 'l' or letter == 'p' or letter == 'e' or letter == 's' or letter == 't' or letter == 'a' or letter == 'n' or letter == 'a' or letter == 's':
        continue
    for letter2 in letters:
        if letter2 == 'b' or letter2 == 'i' or letter2 == 'e' or letter2 == 's' or letter2 == 'p' or letter2 == 'e' or letter2 == 'l' or letter2 == 'p' or letter2 == 'e' or letter2 == 's' or letter2 == 't' or letter2 == 'a' or letter2 == 'n' or letter2 == 'a' or letter2 == 's':
            continue
        com.append(letter + letter2)

for c in com:
    if "a" in c or "e" in c or "i" in c or "o" in c or "u" in c:
        for o in com:
            if "a" in c or "e" in c or "i" in c or "o" in c or "u" in c:
                for m in com:
                    if "a" in c or "e" in c or "i" in c or "o" in c or "u" in c:
                        names.append(c + o + m)
namez = []
for name in names:
    # biespltan
    if  "fou" and "uf" in name:
        # print(name)
        namez.append(name)
    else:
        continue
print(len(namez))