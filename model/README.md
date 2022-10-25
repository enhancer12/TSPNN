# Model

## implementation details

In this paper, we proposed a two-stage progressive neural network for AEC task, which consists of a time delay compensation (TDC) block, a coarse-stage module and a fine-stage module.

![The flowchart of the proposed method.](../network.png)

### coarse-stage

1. Eight conv2D layers in the encoder:

    | layer | input channel | output channel | kernel size | stride |
    | :-: | :-: | :-: | :-: | :-: |
    | 0 | 2 | 16 | [5,1] | [1,1] |
    | 1 | 16 | 16 | [1,5] | [1,1] |
    | 2 | 16 | 16 | [6,5] | [2,1] |
    | 3 | 16 | 32 | [4,3] | [2,1] |
    | 4 | 32 | 32 | [6,5] | [2,1] |
    | 5 | 32 | 32 | [5,3] | [2,1] |
    | 6 | 32 | 32 | [3,5] | [2,1] |
    | 7 | 32 | 32 | [3,3] | [1,1] |

2. Two FT-GRU layers:

    | layer | input size | hidden size | bidrectional |
    | :-: | :-: | :-: | :-: |
    | F-GRU 0 | 32 | 32 | True |
    | T-GRU 0 | 64 | 64 | False |
    | BN+PReLu | | | |
    | F-GRU 1 | 64 | 32 | True |
    | T-GRU 1 | 64 | 32 | False |
    | BN+PReLu | | | |

3. VAD Path:

    | Layer | Input size | Ouput size |
    | :-: | :-: | :-: |
    | Conv2D | $32 \times F \times T$ | $16 \times F \times T$ |
    | BN+PReLu | | |
    | F-GRU | $16 \times F \times T$ | hidden state: $8 \times 2 \times T$ |
    | Reshape | $8 \times 2 \times T$ | $16 \times T$ |
    | Conv1D | $16 \times T$ | $16 \times T$ |
    | BN+PReLu | | |
    | Conv1D | $16 \times T$ | $2 \times T$ |

### fine-stage

1. Eight conv2D layers in the encoder:
16,16,32,32,64,64,64,64
    | layer | input channel | output channel | kernel size | stride |
    | :-: | :-: | :-: | :-: | :-: |
    | 0 | 3 | 16 | [5,1] | [1,1] |
    | 1 | 16 | 16 | [1,5] | [1,1] |
    | 2 | 16 | 32 | [6,5] | [2,1] |
    | 3 | 32 | 32 | [4,3] | [2,1] |
    | 4 | 32 | 64 | [6,5] | [2,1] |
    | 5 | 64 | 64 | [5,3] | [2,1] |
    | 6 | 64 | 64 | [3,5] | [2,1] |
    | 7 | 64 | 64 | [3,3] | [1,1] |

2. Two FT-GRU layers:

    | layer | input size | hidden size | bidrectional |
    | :-: | :-: | :-: | :-: |
    | F-GRU 0 | 64 | 64 | True |
    | T-GRU 0 | 128 | 128 | False |
    | BN+PReLu | | | |
    | F-GRU 1 | 128 | 64 | True |
    | T-GRU 1 | 128 | 64 | False |
    | BN+PReLu | | | |

3. Deep-filter:

 - $N_f = 3$
 - $N_t = 3$
 - $N_l = 1$