# trabalho-pai

Trabalho feito para a disciplina de Processamento e Análise de Imagens - PUC Minas

## Dependências

Dependências: (todos instalavéis com pip)
pip install mouse, numpy, opencv, pillow, pynput, tk
## Como usar


**Para recortar:**  
Processamento -> Recortar e salvar recorte na raiz:  
  clique e arraste para formar um retângulo  
  se você estiver satisfeito com a seleção, aperte 1 para salvar  
  se você não estiver satisteito com a seleção, aperte 0 para apagar a seleção  
  a imagem cortada é salva no diretório raiz com o nome "crop.png"  
  
**Para procurar o recorte na imagem aberta:**  
Processamento -> Buscar de um arquivo / buscar do último recorte  
(ambas funcionam de modo parecido, a diferença sendo que na primeira opção você escolhe de qual arquivo quer buscar)  
  você vai ver um retângulo em volta do match mais próximo da imagem  
  a cor do retângulo varia dependendo de quão próximo é o match (valor aproximado)  

  verde: identico  
  amarelo: parecido  
  vermelho: diferente 

**Para classificar**
Escolha uma imagem
  
Trabalho também hosteado no github em https://github.com/jpoCruz/trabalho-pai
