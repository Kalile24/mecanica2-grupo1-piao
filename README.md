# Mecânica II: Grupo I (Pião)

Trabalho de VE de Mecânica II (IME, CFG, 2.º ano, Turma A): precessão permanente de um pião.

- `entrega/`: PDFs finais do relatório, do memorial de cálculo, da apresentação e do roteiro de fala.
- `relatorio/`: fontes em LaTeX e Beamer, figuras e script de cálculos.

## Como compilar

Dentro de `relatorio/`, execute três vezes, nesta ordem (o memorial lê os números das equações do relatório):

```
pdflatex relatorio_grupo_I.tex
pdflatex memorial_de_calculo.tex
pdflatex apresentacao_grupo_I.tex
pdflatex apresentacao_grupo_I_notas.tex
```

Para refazer os cálculos e os gráficos, é preciso ter Python com `numpy` e `matplotlib`:

```
python relatorio/scripts/calculos_e_graficos.py
```

A primeira execução leva alguns minutos, porque roda as simulações numéricas. As seguintes reaproveitam os resultados guardados em `relatorio/scripts/cache_simulacoes/`.
