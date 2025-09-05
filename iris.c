#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* ==========================================================================
 * gcc -Wall -Wextra -Werror {-DSHUFFLE -DFILE_SAVE} iris.c -lm -o iris && ./iris
 * ==========================================================================
 * input layer  hidden layer  ouput layer
 *   +---+       
 *   |=I0|        
 *   +---+         +---+          +---+
 *                 |   |          |=O0| h_0
 *   +---+         +---+          +---+
 *   |=I1|                      
 *   +---+         +---+          +---+
 *        w's_{ih} |   | w's_{oh} |=O1| h_1
 *   +---+         +---+          +---+
 *   |=I2|                      
 *   +---+         +---+          +---+
 *                 |   |          |=O2| h_2
 *   +---+         +---+          +---+
 *   |=I3|  B_h             B_o
 *   +---+       
 *
 * ==========================================================================
 * Felipe S. Pegoraro - 837486
 * -------------------------------------------------------------------------- 
 * [iris.xls] (do 2 ate 151): 
 * A      B          C           D          E          F 
 * id sepalwidth sepallength petalwidth petallength species 
 * -------------------------------------------------------------------------- 
 * [iris.xls] (do G ate J), da linha 2 ate 151: =(B2-B$153)/(B$152-B$153)
 * para normalizar os dados.
 * -------------------------------------------------------------------------- 
 * [iris.xls] (152): maximo de cada coluna 
 * [iris.xls] (153): minimo de cada coluna 
 * -------------------------------------------------------------------------- 
 * [iris.xls] (do K ate M): =SE($F22="Name"; 1; 0) 
 * -------------------------------------------------------------------------- 
 * [data_iris.txt]: os dados das colunas de G ate M 
 *///=========================================================================

#define MAX_EPOCHS 10000
#define N_INPUT 4
#define N_HIDDEN ((N_INPUT/2)+1)
#define N_OUTPUT 3
#define N_DATA_SAMPLES 150
#define LEARNING_RATE 0.5f

#ifdef SHUFFLE
#define FILE_DATASET "./data/data_iris_shuffle.txt"
#else
#define FILE_DATASET "./data/data_iris.txt"
#endif

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float sigmoid_derivative(float x) {
    return x * (1.0f - x);
}

void feedforward(
    const float inputs[N_INPUT],
    const float w_ih[N_INPUT + 1][N_HIDDEN],
    const float w_ho[N_HIDDEN + 1][N_OUTPUT],
    float final_output[N_OUTPUT],
    float hidden_output[N_HIDDEN]
){
    
    // calcula a saída da camada oculta
    for (int j = 0; j < N_HIDDEN; j++) {
        float sum = w_ih[N_INPUT][j]; // começa com bias
        for (int i = 0; i < N_INPUT; i++) {
            sum += inputs[i] * w_ih[i][j];
        }
        hidden_output[j] = sigmoid(sum);
    }

    // calcula a saída da camada final
    for (int j = 0; j < N_OUTPUT; j++) {
        float sum = w_ho[N_HIDDEN][j];
        for (int i = 0; i < N_HIDDEN; i++) {
            sum += hidden_output[i] * w_ho[i][j];
        }
        final_output[j] = sigmoid(sum);
    }
}

int main(void) {
    // --- 1. CARREGAR DADOS ---
    // entradas IN[] e targets TN[] para 0 <= N <= 2
    float I0[N_DATA_SAMPLES], I1[N_DATA_SAMPLES], I2[N_DATA_SAMPLES], I3[N_DATA_SAMPLES];
    float T0[N_DATA_SAMPLES], T1[N_DATA_SAMPLES], T2[N_DATA_SAMPLES];

    FILE *in = fopen(FILE_DATASET, "rt");
    if (in == NULL) {
        printf("Erro: Não foi possível abrir o arquivo data_iris.txt\n");
        return EXIT_FAILURE;
    }

    // MODIFICADO: O loop agora preenche os arrays 1D.
    for (int i = 0; i < N_DATA_SAMPLES; i++) {
        fscanf(in, "%f %f %f %f %f %f %f",
            &I0[i], &I1[i], &I2[i], &I3[i],
            &T0[i], &T1[i], &T2[i]);
    }
    fclose(in);



    // --- 2. INICIALIZAÇÃO DOS PESOS E BIAS ---
    float w_ih[N_INPUT + 1][N_HIDDEN];
    float w_ho[N_HIDDEN + 1][N_OUTPUT];

    srand(42);
    for (int i = 0; i <= N_INPUT; i++) {
        for (int j = 0; j < N_HIDDEN; j++) {
            w_ih[i][j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }
    for (int i = 0; i <= N_HIDDEN; i++) {
        for (int j = 0; j < N_OUTPUT; j++) {
            w_ho[i][j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }


    // --- 3. TREINAMENTO ---
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        float total_error = 0.0f;
        for (int s = 0; s < N_DATA_SAMPLES; s++) {

            float current_input[N_INPUT]   = {I0[s], I1[s], I2[s], I3[s]};
            float current_target[N_OUTPUT] = {T0[s], T1[s], T2[s]};

            // --- FORWARD PASS ---
            float hidden_output[N_HIDDEN];
            float final_output[N_OUTPUT];

            feedforward(current_input, w_ih, w_ho, final_output, hidden_output);

            // --- BACKPROPAGATION ---
            float output_delta[N_OUTPUT];
            for (int j = 0; j < N_OUTPUT; j++) {
                float error = current_target[j] - final_output[j];
                total_error += error * error;
                output_delta[j] = error * sigmoid_derivative(final_output[j]);
            }
            
            float hidden_delta[N_HIDDEN];
            for (int j = 0; j < N_HIDDEN; j++) {
                float error = 0.0f;
                for (int k = 0; k < N_OUTPUT; k++) {
                    error += output_delta[k] * w_ho[j][k];
                }
                hidden_delta[j] = error * sigmoid_derivative(hidden_output[j]);
            }

            // --- ATUALIZAÇÃO DOS PESOS ---
            for (int j = 0; j < N_OUTPUT; j++) {
                for (int i = 0; i < N_HIDDEN; i++) {
                    w_ho[i][j] += LEARNING_RATE * output_delta[j] * hidden_output[i];
                }
                w_ho[N_HIDDEN][j] += LEARNING_RATE * output_delta[j];
            }
            
            for (int j = 0; j < N_HIDDEN; j++) {
                for (int i = 0; i < N_INPUT; i++) {
                    w_ih[i][j] += LEARNING_RATE * hidden_delta[j] * current_input[i];
                }
                w_ih[N_INPUT][j] += LEARNING_RATE * hidden_delta[j];
            }
        }
        
        if (epoch % 1000 == 0) {
            printf("Epoch %d, Erro total: %.4f\n", epoch, total_error);
        }
    }





    // --- 4. TESTANDO A REDE E SALVANDO RESULTADOS ---
    printf("\nResultados Finais\n");

    float h0[N_DATA_SAMPLES], h1[N_DATA_SAMPLES], h2[N_DATA_SAMPLES];

    #ifdef FILE_SAVE 
        FILE *f_inputs = fopen("inputs.txt", "w");
        FILE *f_outputs = fopen("outputs.txt", "w");
        if (f_inputs == NULL || f_outputs == NULL) {
            printf("Erro ao criar arquivos de saída.\n");
            return 1;
        }
    #endif

    for (int s = 0; s < 150; s++) {
        float current_input[N_INPUT] = {I0[s], I1[s], I2[s], I3[s]};
        
        // Forward pass
        float hidden_output[N_HIDDEN];
        float final_output[N_OUTPUT];
        feedforward(current_input, w_ih, w_ho, final_output, hidden_output);

        h0[s] = final_output[0];
        h1[s] = final_output[1];
        h2[s] = final_output[2];
        
        printf("Input: [%.1f, %.1f, %.1f, %.1f]", I0[s], I1[s], I2[s], I3[s]);
        printf(" -> Output: [%.2f, %.2f, %.2f]", h0[s], h1[s], h2[s]);
        printf(" | Target: [%.0f, %.0f, %.0f]\n", T0[s], T1[s], T2[s]);

        #ifdef FILE_SAVE
            fprintf(f_inputs, "%.4f %.4f %.4f %.4f\n", I0[s], I1[s], I2[s], I3[s]);
            fprintf(f_outputs, "%.4f %.4f %.4f\n", h0[s], h1[s], h2[s]);
        #endif
    }

    #ifdef FILE_SAVE
        fclose(f_inputs);
        fclose(f_outputs);
        printf("\nArquivos 'inputs.txt' e 'outputs.txt' salvos com sucesso.\n");
    #endif

    // --- 5. EXIBIR PESOS FINAIS ---
    printf("\n\nPesos Finais da Rede Treinada\n");
    printf("\nCamada entrada -> oculta:\n");
    for (int i = 0; i < N_INPUT; i++) {
        for (int j = 0; j < N_HIDDEN; j++) {
            printf("W_ih[%d][%d]: %+.4f\n", i + 1, j + 1, w_ih[i][j]);
        }
    }
    printf("\nBias para camada oculta:\n");
    for (int j = 0; j < N_HIDDEN; j++) {
        printf("B_h[%d]: %+.4f\n", j + 1, w_ih[N_INPUT][j]);
    }
    printf("\nCamada oculta -> saida:\n");
    for (int i = 0; i < N_HIDDEN; i++) {
        for (int j = 0; j < N_OUTPUT; j++) {
            printf("W_ho[%d][%d]: %+.4f\n", i + 1, j + 1, w_ho[i][j]);
        }
    }
    printf("\nBias para camada de saida:\n");
    for (int j = 0; j < N_OUTPUT; j++) {
        printf("B_o[%d]: %+.4f\n", j + 1, w_ho[N_HIDDEN][j]);
    }
    
    return EXIT_SUCCESS;
}
