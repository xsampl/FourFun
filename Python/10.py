import random

names = []
vowels = ["a","e","i","o","u"]
consonants = ["b","c","d","f","g","h","j","k","l","m","n","p","q","r","s","t","v","w","x","y","z"]
syllables = []

for c in consonants:
    for v in vowels:
        syllables += [c + v]

names = []

def name_gen():
    name = ""

    for i in range(3):
        name += random.choice(syllables)

    name += " "

    for i in range(3):
        name += random.choice(syllables)
    
    names.append(name)
    
    return name

name_gen()
name_gen()
name_gen()

print(*names, sep = "\n")
