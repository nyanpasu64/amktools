# amktools: Python tools to powerup AddmusicK

## Installation

The easiest method is to download a binary build under https://github.com/nyanpasu64/amktools/releases and add the folder into your PATH.

To install from source:

- Install Python 3.
- (Optional) create virtualenv (honestly don't, it will make your life harder)
- pip install -e .

There are no external dependencies (brr_encoder/brr_decoder is bundled, and sox is no longer needed).

## wav2brr: Automated WAV to BRR converter and tuner

wav2brr is an automated .brr sample creator, which uses .cfg files to convert, resample, loop, and tune .wav samples.

`python -m amktools.wav2brr [WAV file root] [path to AMK] [name of destination sample folder]`

- `-v` or `--verbose` prints out all calls to brr_encoder, as well as output.
- If you add the `--sf2 [sf2_path]` flag, and name a sample `flute.{wav,cfg}`, it will extract tuning and loop points from the sf2 file's `flute` sample.
    - If you name a sample `hammer.{wav,cfg}` and the sf2 has no no `hammer` sample, tuning and loop information will not be extracted.
    - In the future, I could ensure that `.sf2cfg` files always extract tuning/looping (and name must match a sample within sf2), while `.cfg` files never uses .sf2 information.

Currently, all `--options` are undocumented and missing from `--help`, and the Click CLI is not tested or finalized.

### wav2brr Tutorial

See [docs/wav2brr.md](docs/wav2brr.md)

## mmkparser: TXT tuning import and meta-command parser

mmkparser parses .mmk files. They act like .txt files, except they can reference wav2brr tuning data. MMK files also have human-readable high-level commands to replace raw hex commands.

```txt
Examples:
`mmk_parser file.mmk`                   outputs to file.txt
`mmk_parser file.mmk infile2.mmk`       outputs to file.txt
`mmk_parser file.mmk -o outfile.txt`    outputs to outfile.txt
```

### mmkparser Tutorial

See [docs/mmkparser.md](docs/mmkparser.md).

## .gitignore

This program automatically generates many files, which change frequently and should be excluded from Git (I hope you're using version control for your music).

```bash
*.brr
* decoded.wav
tuning.yaml
(optionally) outfile.txt (.txt file generated by mmkparser)
```

## Credits

- wav2brr uses BRRtools by Bregalad and Optiroc, modified by me (https://github.com/nyanpasu64/BRRtools).
