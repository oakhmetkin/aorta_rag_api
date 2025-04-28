import re


with open('er_ru.txt') as f:
    lines = f.readlines()


lines = [
    line for line in lines 
    if re.findall(r'[^А-Яа-я](с|м)м[^А-Яа-я]', line)
]


with open('er_ru1.txt', 'w') as f:
    f.writelines(lines)
