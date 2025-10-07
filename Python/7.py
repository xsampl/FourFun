def concat(*args, sep='/'):
    args = " ".split(args)
    return sep.join(args)

print(concat("earth", "mars", 4))
print(concat("earth", "mars", "venus", sep='*'))