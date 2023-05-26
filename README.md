# Machine_Learning_RecSys
Projeto de Dissertação de Mestrado de Machine Learning para Sistemas de Recomendação que visa mitigar o viés de popularidade.

Este projeto utiliza como base os dois modelos:
- https://arxiv.org/abs/1205.2618
- https://dl.acm.org/doi/abs/10.1145/3523227.3546757

O segundo é artigo uma extensão do primeiro.

Este projeto visa adicionar uma terceira extensão para mitigar um problema de performance encontrado e mitigar ainda mais o viés de popularidade na função objetivo.

O viés de popularidade ocorre quando dois itens são igualmente apreciados por um usuário, porém, um deles se sobressai apenas pelo fato de ser popular.

Para treinar, validar e testar o modelo rode o arquivo: 

- main_basic.py

Nesse arquivo há diversos hiperparâmetros de treinamento que podem ser ajustados.

Explicações sobre eles estão nos comentários do código.

<br>

Os dados estão divididos por padrão em 5-Folds e há dados para rodar 20 epochs no máximo.

Com 5-Folds se consegue 60% de dados para treino, 20% para validação e 20% para teste.

Para variar o número de folds e o número máximo de epochs, pode-se gerar os dados novamente,
ajustando e rodando o jupyter notebook:

- K-Fold Data Generation.ipynb
