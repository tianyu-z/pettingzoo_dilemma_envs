# pettingzoo_dilemma_envs

Acknowledge: The code in the rllib branch is modified from Marco Jiralerspong.  

## How to test envs?
- for Prisoners_Dilemma, Samaritans_Dilemma, Stag_Hunt, Chicken
```python
python dilemma_pettingzoo.py
```
- for Coin_Game
```python
python coin_game_pettingzoo.py
```
## How to test GFN for games?
```python
# IPD as an example
python gfn_dev.py
```
## How to sample using GFN?
Replace "checkpoints/63f794eb/checkpoints_1400.pt" with the correct file path, then run
```python
# IPD as an example
python gfn_sample.py
```
