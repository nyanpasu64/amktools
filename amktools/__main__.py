import sys

from amktools import convert_brr


if __name__ == '__main__':
    if 1 < len(sys.argv):
        convert_brr.main(sys.argv[1])
    else:
        convert_brr.main(None)
