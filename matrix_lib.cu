/*INF1029 - INT ARQUITETURA COMPUTADORES - 2022.2 - 3WA
Trabalho 3 - Módulo avançado (AVX/FMA) para operações com matrizes
Nome: Eric Leão     Matrícula: 2110694
Nome: Pedro Machado Peçanha    Matrícula: 2110535*/

#include "matrix_lib.h"
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

int n_threads = 256;
int n_blocks = 4096;
#define PARTIAL 0
#define COMPLETE 1
int set_grid_size(int threads_per_block, int max_blocks_per_grid) {
  if (threads_per_block > 1024) {
    fprintf(stderr, "Número máximo de threads por block excedido.");
    return 0;
  }
  if (max_blocks_per_grid > 65535) {
    fprintf(stderr, "Número máximo de blocos por grid excedido.");
    return 0;
  }

  n_threads = threads_per_block;
  n_blocks = max_blocks_per_grid;
  return 1;
}

__global__ void scalar(int n, float *d_rows, float scalar) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < n; i += stride) {
    d_rows[i] *= scalar;
  }
}

int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
  cudaError_t cudaError;
  float *current = matrix->h_rows;
  int n_blocks_new = (((int)(matrix->width*sizeof(float)) + n_threads - 1)/n_threads);
  if (matrix == NULL)
    return 0;
  switch (matrix->alloc_mode) {
  case PARTIAL:
    for (int c = 0; c < matrix->height; c++, current += matrix->width) {
      cudaError =
          cudaMemcpy(matrix->d_rows, current, matrix->width * sizeof(float),
                     cudaMemcpyHostToDevice);
      if (cudaError != cudaSuccess) {
        fprintf(stderr, "Cudaerror");
        return 0;
      }
      scalar<<<n_blocks_new, n_threads>>>(matrix->width, matrix->d_rows,
                                      scalar_value);
      cudaDeviceSynchronize();
      cudaError =
          cudaMemcpy(current, matrix->d_rows, matrix->width * sizeof(float),
                     cudaMemcpyDeviceToHost);
      if (cudaError != cudaSuccess) {
        fprintf(stderr, "Cudaerror");
        return 0;
      }
    }
    break;
  case COMPLETE:
    cudaError = cudaMemcpy(matrix->d_rows, matrix->h_rows,
                           matrix->height * matrix->width * sizeof(float),
                           cudaMemcpyHostToDevice);
    if (cudaError != cudaSuccess) {
      fprintf(stderr, "Cudaerror");
      return 0;
    }
    scalar<<<n_blocks, n_threads>>>(matrix->height * matrix->width,
                                    matrix->d_rows, scalar_value);
    cudaDeviceSynchronize();
    cudaError = cudaMemcpy(matrix->h_rows, matrix->d_rows,
                           matrix->height * matrix->width * sizeof(float),
                           cudaMemcpyDeviceToHost);
    if (cudaError != cudaSuccess) {
      fprintf(stderr, "Cudaerror");
      return 0;
    }
    break;
  default:
    fprintf(stderr, "Não existe esse tipo");
    return 0;
  }
  return 1;
}
__global__ void matrixM(int n, float *d_rowsA, float *d_rowsB, float *d_rowsC,
                        int widthA, int widthB, int widthC) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int i, j, k;
  for (i = index; i < n; i += stride) {
    if (d_rowsC[i] != 0) {
      d_rowsC[i] = 0;
    }
    for (j = 0; j < widthA; j++) {
      d_rowsC[i] +=
          d_rowsA[j + (i / widthC * widthA)] * d_rowsB[j * widthB + i % widthC];
    }
  }
}
int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB,
                       struct matrix *matrixC) {
  cudaError_t cudaError;

  if (matrixA == NULL)
    return 0; // matrx A diferente de NULL
  if (matrixB == NULL)
    return 0; // matrx B diferente de NULL
  if (matrixC == NULL)
    return 0; // matrx C diferente de NULL
  if (matrixA->width != matrixB->height)
    return 0; // matrx A precisa ter largura igual à altura da matrix B
  if (matrixA->height != matrixC->height || matrixB->width != matrixC->width)
    return 0; // matriz C tem que ter altura e largura compativeis com a amatriz
              // A e B
  float *currentA = matrixA->h_rows;
  float *currentC = matrixC->h_rows;
  int matrixASize = matrixA->width * matrixA->height;
  switch (matrixA->alloc_mode) {
  case PARTIAL:
    cudaError = cudaMemcpy(matrixB->d_rows, matrixB->h_rows,
                           matrixB->width * matrixB->height * sizeof(float),
                           cudaMemcpyHostToDevice);
    for (int c = 0; c < matrixA->height; c++) {
      cudaError =
          cudaMemcpy(matrixA->d_rows, currentA, matrixA->width * sizeof(float),
                     cudaMemcpyHostToDevice);
      if (cudaError != cudaSuccess) {
        fprintf(stderr, "Cudaerror");
        return 0;
      }
      cudaError =
          cudaMemcpy(matrixC->d_rows, currentC, matrixC->width * sizeof(float),
                     cudaMemcpyHostToDevice);
      if (cudaError != cudaSuccess) {
        fprintf(stderr, "Cudaerror");
        return 0;
      }
      matrixM<<<n_blocks, n_threads>>>(
          matrixC->width, matrixA->d_rows, matrixB->d_rows, matrixC->d_rows,
          matrixA->width, matrixB->width, matrixC->width);
      cudaDeviceSynchronize();
      cudaError =
          cudaMemcpy(currentC, matrixC->d_rows, matrixC->width * sizeof(float),
                     cudaMemcpyDeviceToHost);
      if (cudaError != cudaSuccess) {
        fprintf(stderr, "Cudaerror");
        return 0;
      }
      currentA += matrixA->width;
      currentC += matrixC->width;
    }
    break;
  case COMPLETE:
    cudaError = cudaMemcpy(matrixA->d_rows, matrixA->h_rows,
                           matrixA->height * matrixA->width * sizeof(float),
                           cudaMemcpyHostToDevice);
    if (cudaError != cudaSuccess) {
      fprintf(stderr, "Cudaerror");
      return 0;
    }
    cudaError = cudaMemcpy(matrixB->d_rows, matrixB->h_rows,
                           matrixB->height * matrixB->width * sizeof(float),
                           cudaMemcpyHostToDevice);
    if (cudaError != cudaSuccess) {
      fprintf(stderr, "Cudaerror");
      return 0;
    }
    cudaError = cudaMemcpy(matrixC->d_rows, matrixC->h_rows,
                           matrixC->height * matrixC->width * sizeof(float),
                           cudaMemcpyHostToDevice);
    if (cudaError != cudaSuccess) {
      fprintf(stderr, "Cudaerror");
      return 0;
    }
    printf("matrixA->width: %d\n", matrixA->width);
    matrixM<<<n_blocks, n_threads>>>(
        matrixC->height * matrixC->width, matrixA->d_rows, matrixB->d_rows,
        matrixC->d_rows, matrixA->width, matrixB->width, matrixC->width);
    cudaDeviceSynchronize();
    cudaError = cudaMemcpy(matrixC->h_rows, matrixC->d_rows,
                           matrixC->height * matrixC->width * sizeof(float),
                           cudaMemcpyDeviceToHost);
    if (cudaError != cudaSuccess) {
      fprintf(stderr, "Cudaerror");
      return 0;
    }
    break;
  default:
    fprintf(stderr, "Não existe esse tipo");
    return 0;
  }
  return 1;
}
