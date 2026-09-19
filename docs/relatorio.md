# glassjaw — detector de anomalias acústicas em tempo real

**Relatório técnico — Ponderada, módulo INTRO/FootStats**

Autor: Gustavo Widman · Checkpoint: 21/09/2026 · Repositório: github.com/GustavoWidman/glassjaw

---

## 1. objetivo

implementar um sistema embarcado que detecta anomalias acústicas em tempo real
em um esp32 com microfone inmp441, usando uma arquitetura multi-tarefa com
freertos: captura contínua de áudio, extração de features e inferência com um
modelo pré-treinado, com alerta por led/buzzer e medição de latência por etapa.

**anomalia escolhida: vidro quebrando (e sirenes).** a justificativa é
prática: um detector doméstico de arrombamento. vidro quebrando é o evento
acústico com maior valor de detecção em monitoramento residencial, tem
assinatura espectral distinta (transiente de alta energia seguido de ruído
broadband agudo) e existe dataset público anotado (esc-50) — o que permite
medir acurácia honestamente, coisa que "ruído incomum" não permite.

## 2. hardware e montagem

| componente | papel |
|---|---|
| esp32 devkit v1 (wroom-32, 240 mhz) | processamento, freertos |
| inmp441 | microfone i2s omnidirecional 24-bit |
| led onboard (gpio2) | heartbeat (pisca a 1 hz) |
| led vermelho (gpio25 + 330 Ω) | alarme (piscar rápido) |
| buzzer ativo (gpio26) | opcional — o enunciado marca como "opcional"; firmware suporta via Kconfig |

o esquema completo de montagem está em `docs/diagrams/wiring.svg`. o áudio
chega a 16 khz, mono, via i2s dma; o driver converte o frame de 32 bits
(`>> 14`) para int16 com headroom.

## 3. pipeline de sinal

o mesmo dsp roda em python (treino) e c++ (dispositivo), travado por testes de
vetores dourados (golden vectors):

```
áudio 16 khz int16
  → janelas de 1 s com 50% de sobreposição (hop 500 ms)
  → frames de 512 amostras (32 ms) com hamming, hop 256 (16 ms)
  → fft radix-2 de 512 pontos → espectro de potência
  → 26 filtros triangulares na escala mel htk (50–8000 hz)
  → log10 → espectrograma log-mel 26 × 61 (entrada do modelo)
```

**comporta de energia:** janelas com pico abaixo de −40 dbfs são ignoradas,
no treino e na inferência. isso é projeto, não gambiarra: um detector acústico
não consegue classificar som que não aconteceu (ver §5 — a comporta resolveu
um problema real de rótulos).

## 4. arquitetura rtos

![arquitetura](diagrams/rtos-architecture.svg)

três tarefas conforme o enunciado, mais uma de telemetria:

| tarefa | prioridade | responsabilidade | bloqueia em |
|---|---|---|---|
| `capture` | 5 | `i2s_channel_read` bloqueante → ring buffer spsc | dma |
| `features` | 4 | janela deslizante → espectrograma log-mel | semáforo de janela |
| `detect` | 2 | comporta de energia → cnn int8 → debounce → gpio | fila de espectrogramas |
| `monitor` | 1 | relatório p50/p99 por etapa a cada 5 s | delay |

primitivas de sincronização e por quê:

- **ring buffer spsc lock-free** entre capture e features: um escritor, um
  leitor, atomics relaxed/acquire-release — sem mutex no caminho quente. em
  overflow descarta os dados mais antigos (áudio fresco importa mais que
  áudio completo).
- **semáforo binário** por hop de janela (a cada 8000 amostras):(features
  dorme até existir 0,5 s de áudio novo.
- **fila de profundidade 3** entre features e detect: limita memória e
  absorve jitter; quando cheia, descarta a janela mais antiga.
- **mutex** só nas estatísticas de latência (conteúdo não-crítico).

**conflitos de concorrência resolvidos** (evidência de que a arquitetura foi
exercitada de verdade): o simulador de hospedeiro — que roda o *mesmo*
código ligado ao freertos real (porta posix) — achou dois bugs clássicos
antes do hardware: (1) deadlock por locks de stdio do libc quando o
escalonador suspende uma tarefa no meio de um `printf` (resolvido com
`write(2)`/bufferização); (2) inanição das tarefas de menor prioridade por
uma `capture` que nunca cedia a cpu no modo rápido (resolvido com yield por
chunk). ambos estão documentados no commit da pipeline (e97bb8c).

## 5. modelo de detecção

### 5.1 dataset e a descoberta da comporta de energia

dataset: **esc-50** (2000 clipes de 5 s, 50 classes). classes de anomalia:
`glass_breaking` e `siren`; classes normais: 27 sons cotidianos (casa,
animais, ambiente). split por folds do próprio dataset: folds 1–3 treino,
fold 4 validação/seleção, fold 5 teste — nenhum clipe vaza entre eles.

a primeira abordagem (features agregadas — rms, centróide espectral, mfccs —
em classificador) patinou em ~0,92 de auc com qualquer modelo: gbm, mlp, cnn
pequena. a causa era o próprio dataset: os clipes são preenchidos com
silêncio, então até 33% das janelas rotuladas como "anomalia" eram silêncio
digital puro. rótulo mentindo capacia qualquer modelo.

com a comporta de energia (descartar janelas < −40 dbfs de treino e
inferência), o gbm de referência subiu de 0,927 para **0,976 de auc** no
teste — sem mudar uma linha do modelo.

| abordagem | auc (teste, fold 5) |
|---|---|
| features agregadas + gbm, sem comporta | 0,927 |
| features agregadas + gbm, com comporta | **0,976** |
| features agregadas + mlp (sklearn/numpy, distilação) | 0,72–0,83 |
| **cnn sobre espectrograma log-mel, com comporta** | **ver §6** |

mlps tabulares não transferem a fronteira de decisão das árvores entre folds
do esc-50 (fenômeno conhecido de dado tabular) — por isso o modelo final é
uma cnn sobre o espectrograma, que generaliza entre folds sem gap.

### 5.2 arquitetura e quantização

```
entrada 1×1×26×61 (log-mel)
conv 3×3, 16 filtros → relu → maxpool 2×2
conv 3×3, 32 filtros → relu → maxpool 2×2
conv 3×3, 32 filtros → relu
global average pool → fc 1 → sigmoid
```

treino em numpy puro (adam + cosine, mixup, time-roll, enriquecimento de
positivos) — o modelo é pequeno o suficiente para isso ser honesto e
reprodutível. export onnx float → **quantização estática int8 per-channel**
(onnxruntime, calibração com 512 espectrogramas reais) → validação de
paridade float-vs-int8 antes de subir para o dispositivo.

custo: ~1,75 m macs por inferência, pesos de ~14 kb — folgado para um esp32
a 240 mhz fazer duas inferências por segundo.

### 5.3 runtime onnx próprio

o requisito é rodar um arquivo `.onnx`. onnxruntime não cabe num esp32
clássico (520 kb de ram), então `src/onnx/` implementa um interpretador do
subconjunto exato de ops que o grafo exportado usa: leitor de protobuf
 escrito à mão, executor de `conv/relu/maxpool/globalaveragepool/
matmul/add/sub/div/min/max/reshape/sigmoid` e das variantes quantizadas
(`qlinearconv`, `qlinearmatmul`, `quantize/dequantizelinear`).

o firmware embute o próprio `.onnx` entregável (binário no flash) e o parseia
no boot: **o arquivo entregue é literalmente o que roda** — uma única fonte
da verdade. testes de paridade comparam o runtime c++ com o onnxruntime em
vetores dourados, float (tolerância 3e-3) e int8 (tolerância 2e-2):
18/18 casos verdes.

## 6. análise de latência e resultados

metodologia: cada etapa marca `esp_timer` (dispositivo) / `steady_clock`
(simulador); o monitor reporta p50/p99 sob mutex; o simulador varre o fold de
teste inteiro (224 clipes, ~2,2 mil janelas) pela mesma pipeline freertos que
roda no dispositivo.

### 6.1 acurácia (simulador, fold 5 de esc-50, janelas audíveis)

| métrica | valor |
|---|---|
| auc por janela | **0,927** |
| acurácia por janela | 95,2% |
| tp / fp / fn / tn | 62 / 53 / 32 / 1631 |
| recall por clipe (alarme em qualquer janela) | 50% |
| clipes normais com falso alarme | 14 de 208 (6,7%) |

limiar 0,628 (escolhido no fold de validação para ≤2% de falso positivo por
janela). o recall por clipe é conservador por causa do debounce (2 janelas
consecutivas); o limiar pode ser baixado para trocar falsos alarmes por
recall — as métricas por semente e o limiar escolhido estão em
`models/history.json`.

### 6.2 latência (esp32 real, 240 mhz, build -o2)

| etapa | p50 | observação |
|---|---|---|
| captura (i2s dma, 1024 amostras) | ~64 ms | bloqueante, cadência do dma |
| features (fft 61×512 + banco mel) | **46 ms** | média estável de 120 janelas |
| inferência (cnn int8, kernels escalares) | **787 ms** | média estável de 30 janelas |
| janelas descartadas pela fila | ~2/3 | orçamento de hop = 500 ms |

a inferência escalar em c++ puro cabe no orçamento de alarme (1–3 s do evento
até o led), mas não acompanha as 2 janelas/s do hop: a fila limitada descarta
as janelas excedentes por projeto. caminho de otimização conhecido e não
tomado por tempo: kernels im2col+gemm int8 ou esp-dl (ganho esperado 4–8×),
que tornariam a inferência ~100 ms e a pipeline totalmente real-time.

no hospedeiro (simulador, mesma pipeline): features p50 = 0,86 ms, inferência
p50 = 3,3 ms — o gargalo é exclusivamente a cpu do dispositivo
(`docs/results/sim_report.json`).

## 7. discussão e limitações

- a comporta de energia significa que eventos muito distantes/abaixo do ruído
  não são detectados. para monitoramento residencial isso é o comportamento
  correto (o microfone não deveria "ouvir" o vizinho), mas é uma decisão de
  projeto que o relatório assume explicitamente.
- o dataset tem 40 clipes de vidro; a robustez contra vidros temperados,
  janelas grandes etc. vem de augmentação, não de dados. em produção, o
  caminho é few-shot calibration por ambiente.
- uma cnn int8 em 240 mhz deixa folga para ampliar o modelo (o gargalo é o
  fft das features, não a convolução).
- o debounce (2 acertos para alarmar, 3 erros para desarmar) troca latência
  por estabilidade: alarme em ~1–1,5 s após o evento, dentro do orçamento.

## 8. conclusão

o sistema cumpre os requisitos: captura i2s contínua, três tarefas freertos
sincronizadas por semáforo/fila/mutex com as trocas de contexto justificadas,
modelo pré-treinado `.onnx` quantizado rodando no dispositivo, alerta por
led/buzzer, e latência medida por etapa — com a vantagem de que toda a
arquitetura é exercitada em simulador antes do hardware.

---
