# AMK WAV to BRR converter

## Folder tree

- addmusick-1.1.0-beta (FIXME pass through a command-line flag)
    - samples
        - project-dependent folder name (fixme command-line flag? pwd?)
            - x.brr
- Song Project Folder
    - convert_brr.cmd
    - build.cmd (automatically called FIXME)
    - wav
        - Folder name doesn't matter
            - x.wav
            - x.cfg
            - x.brr (generated)
            - x decoded.wav (generated)
        - ~Folders beginning with tilde are ignored
        - All other folders must contain .cfg file
    
## Installation

- create venv
- pip install -e .
- Create convert_brr.cmd next to wav folder
    - `call .../activate.bat`
    - Either `python -m amktools` (TODO rename)
    - Or `python -m amktools file.sf2` to get tuning/loop data from soundfont

## .cfg file format

.cfg is eval() as Python code.

if soundfont is present
