import sys

from amktools import wav2brr

if 1 < len(sys.argv):
    # wav2brr.main(sys.argv[1])
    wav2brr.main(None)
else:
    wav2brr.main(None)
