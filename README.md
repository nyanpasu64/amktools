# AMK WAV to BRR converter (wav2brr)

## Installation

The easiest method is to download a binary build under https://github.com/nyanpasu64/amktools/releases and add the folder into your PATH.

To install from source:

- Install Python 3.
- (Optional) create virtualenv (honestly don't, it will make your life harder)
- pip install -e .

There are no external dependencies (brr_encoder/brr_decoder is bundled, and sox is no longer needed).

## wav2brr Usage

`python -m amktools.wav2brr [WAV file root] [path to AMK] [name of destination sample folder]`

- `-v` or `--verbose` prints out all calls to brr_encoder, as well as output.
- If you add the `--sf2 [sf2_path]` flag, and name a sample `flute.{wav,cfg}`, it will extract tuning and loop points from the sf2 file's `flute` sample.
    - If you name a sample `hammer.{wav,cfg}` and the sf2 has no no `hammer` sample, tuning and loop information will not be extracted.
    - In the future, I could ensure that `.sf2cfg` files always extract tuning/looping (and name must match a sample within sf2), while `.cfg` files never uses .sf2 information.

Currently, all `--options` are undocumented and missing from `--help`, and the Click CLI is not tested or finalized.

## wav2brr Tutorial

I assume your project tree is setup this way. (This is not necessary, you can rearrange your tree however you like, as long as you tell wav2brr via command-line arguments.)

- ZMM Final Hours
    - **convert_brr.cmd**
    - wav
        - any_folder_name
            - **`strings.wav`**
            - **`strings.cfg`**
        - ~Folders beginning with tilde are ignored
- addmusick-1.1.0-beta
    - samples

Create a file in your project root (or location of your choice), named **convert_brr.cmd**. This should contain:

```cmd
python -m amktools.wav2brr wav ..\addmusick-1.1.0-beta zmm-final
```

The WAV file root and AddmusicK folder path must exist. The sample folder (zmm-final) will be created if missing. Whenever the script is executed, *all existing files in the folder will be moved to a backup folder*. (I may add support for BRR file passthrough, if there is significant demand.)

### .wav and .cfg files

Place each `.wav` file in a subfolder under the wav directory. Next to each `.wav` file, create a `.cfg` text-file with the same name (`strings.wav` becomes `strings.cfg`).

Each .cfg file is evaluated as a Python expression using `eval()`. You can use arithmetic operations for any value, and use `#` to comment lines.

- `volume`: Multiplies volume by this number (useful to reduce clipping from [anti-Gaussian prefiltering] and [resampling]).
- `ratio`: Resampling ratio (output rate / input rate). `0.5` and `1/2` all produce a half-size sample. If you have OCD or are debugging this program, adding quotes prevents rounding within Python (but brr_encoder rounds anyway).
- `at`: MIDI pitch of .wav file
- `loop` (optional): Loop point (in samples). Hexadecimal values `0x000038AB` work without modification.
    + Omit to disable looping.
- `truncate` (optional): Truncate the end of audio or looped region (in samples).
    + Omit to disable truncation.

If you supply a sf2 for tuning/looping data (not audio data), these options change:

- `at` (optional)
    + If sf2 file supplied, omit to inherit tuning from soundfont.
- `transpose` (optional)
    + Transpose by this many semitones (decimals/fractions allowed).
- `loop` (optional)
    + Supply a value to override sf2 loop point, or `None` to disable looping.
    + Most percussion is unlooped, but SNES/vgmtrans cymbals may be looped. Use a sf2 editor (I use Polyphone) to play the sample, and see if the loop sounds right.
- `truncate` (optional)
    + Supply a value to override sf2 end-loop point.

Looping is **always enabled** if `loop` and `truncate` are both absent (since sf2 stores loop points separately from "is loop enabled"). If only one is present, the other value will be `None` (supplying `loop` disables loop-end truncation, and supplying `truncate` disables looping).

Recommendations:

- Percussion
    - at=(integer), loop=`None`, and optionally truncate=(integer)
- Non-percussion
    + Use `transpose`, not `at`, to change pitch.
    + Omit all options to use sf2 looping.
    + Supply 'loop' and (optionally) 'truncate' to use custom loop points.

### .cfg examples

Pitched instrument, tuned without .sf2, wav file plays at middle C:

```python
dict(
    volume=0.9,     # optional, I usually omit this line
    ratio=0.4,
    loop=0x000038AB,
    truncate=0x0000D883,
    at=60
)
```

Pitched instrument, tuned from sf2:

```python
{
    'ratio': '0.4'
}
```

Unlooped percussion, truncated to reduce .brr size:
```python
dict(
    at=60,
    truncate=16000
)
```

### Result

Double-click `convert_brr.cmd`, and the program will automatically generate .brr samples in `AddmusicK`/samples/`chosen folder`.

- ZMM Final Hours
    - wav
        - any_folder_name
            - **strings.brr**
            - **strings decoded.wav**
    - **tuning.yaml**
- addmusick-1.1.0-beta
    - samples
        - **zmm-final** (will be created if missing)
            - **strings.brr**

Both copies of strings.brr contains properly formatted AddmusicK loop point information (if looping enabled), and are playable in AMK and BRRPlayer.

## mmkparser Usage

Parse one or more MMK files to a single AddmusicK source file.

```txt
Examples:
`mmk_parser file.mmk`                   outputs to file.txt
`mmk_parser file.mmk infile2.mmk`       outputs to file.txt
`mmk_parser file.mmk -o outfile.txt`    outputs to outfile.txt
```

## mmkparser Tutorial

MMK files can import tuning data directly from wav2brr. To do that, place `build.cmd` in the same folder as `convert_brr.cmd` and `tuning.yaml`.

```cmd
@echo off
set folder=..\addmusick-1.1.0-beta
set NAME=zmm-final

    EITHER:
@call convert_brr.cmd
    OR:
python -m amktools.wav2brr wav ..\addmusick-1.1.0-beta %NAME%

python -m amktools.mmkparser %NAME%.mmk || goto :eof
copy %NAME%.txt %folder%\music\%NAME%.txt
```

Rename your .txt song to `%NAME%.mmk` (in this case `zmm-final.mmk`).

### Automatic Tuning

In the .mmk file, under the #instruments section, locate each sample generated by wav2brr and remove the last 2 (tuning) bytes:

```
"strings.brr"   $8F $E0 $00 $02 $1b
```

And add `%tune` to the beginning of the line:

```
%tune "strings.brr"   $8F $E0 $00
```

And tuning data will be automatically extracted from `tuning.yaml`.

- Only one sample per line. Avoid `%tune "strings.brr" $8F $E0 $00 %tune "piano.brr" $8F $E0 $00`
- Do not add comments after ADSR. Avoid `%tune "strings.brr" $8F $E0 $00 ;slow strings`
    - The tuning values will be commented out. This is a mmkparser bug and may be fixed later.

### ADSR

Use the `%adsr` command to automatically compute ADSR. It takes 4 parameters: `attack speed`, `decay speed`, `sustain level`, `release speed`.

- Attack speed
    + 0 is slowest, 15 or -1 is fastest.
- Decay speed
    + 0 is slowest, 7 or -1 is fastest.
- Sustain level
    + 0 to 7 correspond to `decay` ending at (1/8 to 8/8).
    + `full` and -1 map to 7, or "decay envelope is mostly ignored".
- Release speed
    + 0 is slowest (infinite release, 0 speed), 31 or -1 is fastest.

`%adsr` works both within `#instruments` and in music data.

- Within #instruments:
    + `%tune "strings.brr"   %adsr -1,0,full,0`
    + results in a fast attack (2 milliseconds) and no decay over time.
- Outside #instruments:
    + `%adsr -1,0,full,0    c4`
    + replaces the current instrument's ADSR envelope.

**NOTE:** The AddmusicK readme's ADSR calculator is inaccurate and misleading. `Attack` is a linear rise, while `decay` and `release` are exponential decay with adjustable slope/τ, and `sustain level` controls when the `decay` envelope ends and `release` begins.

### GAIN

Use the `%gain` command to automatically compute GAIN. It takes 2 parameters: `curve`, `rate`. `curve` can be one of the following:

- `direct`
    + `rate` determines the loudness of the sample. 0 is silent, {$7f 0x7f 127} is loudest.
- `down`, `exp`, `up`, `bent`
    + `rate` determines the speed of the curve. 0 is no change (constant envelope), {$1f 0x1f 31} is fastest.
    + `down` and `exp` are decreasing envelopes.
    + `up` and `bent` are increasing envelopes.

### Volume and pan scaling

All lowercase `v` and `y` are interpreted as volume and pan commands, and volumes can be rescaled using `%vmod factor`, (deprecated: `%isvol` `%notvol` `%ispan` `%notpan`).

**All "word=something" where the word contains lowercase `v` or `y` will lead to errors when used.**

I may fix this by ignoring all `v` or `y` commands when rescaling is disabled, and/or ignoring all letters not followed by a number.

### Remote Commands

MMK has no special support for remote commands. Use this template instead:

```
"clear=0"
"kon=-1"
"after=1"
"before=2"
"koff=3"
"now=4"
```

Example of use:

```
(!1)[$f4$09]
(!2)[%gain exp $0c ]
"LONG_DECAY=(!1, kon)(!2, koff)"    ; restore on keyon, decay on keyoff
```

I included a space after `%gain exp $0c`. At the moment, this is necessary for my parser to work. (I could add brackets/parentheses as delimiters if necessary.)

Do not define or use (!0), it will not work in AddmusicK (this is an undocumented AMK restriction).

### Result

Run `build.cmd`. This will compile `zmm-final.mmk` to `zmm-final.txt` and copy it into the AddmusicK folder. You can edit `build.cmd` to automatically build the song, but this varies by version.

AMK 1.0.x (unsure):

```
cd %folder%

addmusick -norom -noblock %NAME%.txt
start SPCs\%NAME%.spc
```

AMK 1.1 beta April 2017:

```
cd %folder%

addmusick -m -noblock %NAME%.txt
start SPCs\%NAME%.spc
```

AMK 1.1 beta May 2017:

```
cd %folder%
echo 01 %NAME%.txt> Trackmusic_list.txt

addmusick -m -noblock
start SPCs\%NAME%.spc
```

AMK Beta may sometimes corrupt samples (happens in May 2017, possibly April 2017).

## .gitignore

```bash
* decoded.wav
*.brr
tuning.yaml
```

## Credits

- wav2brr uses BRRtools by Bregalad and Optiroc, modified by me (https://github.com/nyanpasu64/BRRtools).
