# FizzBuzz Zero: Mastering FizzBuzz Game without Human Knowledge

![zero](https://user-images.githubusercontent.com/32036527/39819369-3e9bf334-53de-11e8-8f6d-245e76badc63.png)

> A long-standing goal of artificial intelligence is an algorithm that learns, tabula rasa, superhuman proficiency in challenging domains. --[*D.Silver*](https://deepmind.com/research/publications/mastering-game-go-without-human-knowledge/)  

This is implementation for FizzBuzz using AlphaZero method.  
日本語の解説は[こちら](https://qiita.com/)  

## Requirement

- Python 2.7 / 3.5 or later  
- [TensorFlow](https://www.tensorflow.org/) 1.4 or later  

## Usage
To learn (about 10min with CPU):
```
$ python fizzbuzzzero.py --learn

0 total games / next epoch: 1 

100/100 games
match: accuracy=25.9[%] average length=0.3
train: accuracy=29.0[%] mse=0.500
100 total games / next epoch: 2 

100/100 games
match: accuracy=23.1[%] average length=0.3
train: accuracy=24.3[%] mse=0.499
200 total games / next epoch: 3 

(...)

100/100 games
match: accuracy=100.0[%] average length=100.0
train: accuracy=96.2[%] mse=0.172
1900 total games / next epoch: 20 

100/100 games
match: accuracy=100.0[%] average length=100.0

accuracy seems to be stable at 100%
```
![life1](https://user-images.githubusercontent.com/32036527/39851660-280d0480-5453-11e8-8f5e-63c496e58373.png)


Just to test trained network:
```
$ python fizzbuzzzero.py

<test game>
player lives = (1, 1)
game record ([]=wrong answer):

1, 2, Fizz, 4, Buzz, Fizz, 7, 8, Fizz, Buzz, 
11, Fizz, 13, 14, FizzBuzz, 16, 17, Fizz, 19, Buzz, 
Fizz, 22, 23, Fizz, Buzz, 26, Fizz, 28, 29, FizzBuzz, 
31, 32, Fizz, 34, Buzz, Fizz, 37, 38, Fizz, Buzz, 
41, Fizz, 43, 44, FizzBuzz, 46, 47, Fizz, 49, Buzz, 
Fizz, 52, 53, Fizz, Buzz, 56, Fizz, 58, 59, FizzBuzz, 
61, 62, Fizz, 64, Buzz, Fizz, 67, 68, Fizz, Buzz, 
71, Fizz, 73, 74, FizzBuzz, 76, 77, Fizz, 79, Buzz, 
Fizz, 82, 83, Fizz, Buzz, 86, Fizz, 88, 89, FizzBuzz, 
91, 92, Fizz, 94, Buzz, Fizz, 97, 98, Fizz, Buzz
```

Other options are as follows.   

|option||
|:-------|:----------|
|-h, --help|show help|
|--learn|start to learn|
|--game_cnt GAME_CNT|games to be played in an epoch|
|--serch_limit SEARCH_LIMIT|limit of search count|
|--initial_life INITIAL_LIFE|initial life of each player|
|-v, --verbose|show thinking log of the first game|
|--gpu|enable to use GPU|
|--gpu_cnt GPU_CNT|number of GPUs used for learning|

## License
[MIT License](https://github.com/ymgaq/FizzBuzzZero/blob/master/LICENSE)  

## Author
[Yu Yamaguchi](https://twitter.com/ymg_aq)  

