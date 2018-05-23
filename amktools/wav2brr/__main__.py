import sys

from amktools import wav2brr

if __name__ == '__main__':
    if 1 < len(sys.argv):
        wav2brr.main(sys.argv[1])
    else:
        wav2brr.main(None)
