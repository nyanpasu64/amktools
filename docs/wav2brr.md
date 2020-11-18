# wav2brr Tutorial

**This file is outdated. wav2brr's command line has changed.** It now processes one sample at a time, with both input and output passed in through the CLI. It expects to be invoked by an external tool like Ninja, which is responsible for only running wav2brr when the inputs have changed, saving time during builds when samples have not been changed.

------

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

## .wav and .cfg files

Place each `.wav` file in a subfolder under the wav directory. Next to each `.wav` file, create a `.cfg` text-file with the same name (`strings.wav` becomes `strings.cfg`).

Each .cfg file is evaluated as a Python expression using `eval()`. You can use arithmetic operations for any value, and use `#` to comment lines.

- `volume`: Multiplies volume by this number (useful to reduce clipping from [anti-Gaussian prefiltering] and [resampling]).
- `ratio`: Resampling ratio (output rate / input rate). `0.5` and `1/2` all produce a half-size sample. If you have OCD or are debugging this program, adding quotes prevents rounding within Python (but brr_encoder rounds anyway).
- `target_rate`: Target sampling rate, that the sample will be resampled to. Cannot be used with `ratio`.
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

## .cfg examples

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

## Result

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
