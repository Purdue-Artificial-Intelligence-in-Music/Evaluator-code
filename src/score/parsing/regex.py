import re

def main():
    str1 = "A-4+100"
    str2 = "C4+3"
    note, offset = [x for x in re.split('([A-G][#-]?[0-9]+)([-+][0-9]+)', str1) if x]
    print(note, offset)

if __name__ == "__main__":
    main()