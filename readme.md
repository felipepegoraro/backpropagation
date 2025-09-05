# Iris Dataset Backpropagation

[!img](./img/iris.jpeg)

Este projeto implementa uma rede neural **feedforward com backpropagation** para classificar o [Iris Dataset](https://www.kaggle.com/datasets/uciml/iris)

Use as flags `-DSHUFFLE` para embaralhar os dados e `-DFILE_SAVE` para salvar os resultados em arquivos.

```
git clone https://github.com/felipepegoraro/backpropagation
cd backpropagation
gcc -Wall -Wextra -Werror {-DSHUFFLE -DFILE_SAVE} iris.c -lm -o iris
./iris
```
