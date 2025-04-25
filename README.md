# custom_tokenizations
## Tokenization Explanations
### RelTok
2D tokenization that replaces bar and position tokens with a single token called
`same_or_not` token. By using the cumulativeness of music, we can think notes 
(including monophonic ones) and rests as they are in a stack. All we need to 
pay attention to is whether they share the same time or not, their exact
position can easily be calculated by cumulative duration.  

### ABCD
Symbols means:
- `a` go up 0.5 interval.
- `b` go down 0.5 interval.
- `c` note player.
- `d` note separator.

NOTE: Right now implementation uses `TokSequence.bytes` to store the string abcd
representation. To modify this feature,, one must subclass `TokSequence` and add
another representation similar to `ids`, `bytes`, `toks` etc. 

## Usage
Uses `miditok` as backend, therefore you may use them just like official miditok
tokenizations.

You may refer to `recipes.py` for tokenizing.

## Test
Run the following:
```bash
python -m unittest test/test_tokenizations.py 
```
This will create output files under `output/test`


## Build
Run the following:
```bash
python -m build . 
```

