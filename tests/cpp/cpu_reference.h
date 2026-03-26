/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cmath>
#include <vector>

namespace tfa::test {

inline void cpuAttentionFixedLen(const float* Q, const float* K, const float* V, float* O, int batchSize, int seqLenQ,
                         int seqLenKV, int numHeadsQ, int numHeadsKV, int headDim, bool isCausal) {
  int groupSize = numHeadsQ / numHeadsKV;
  float scale = 1.0f / std::sqrt(static_cast<float>(headDim));

  for (int b = 0; b < batchSize; b++) {
    for (int hq = 0; hq < numHeadsQ; hq++) {
      int hkv = hq / groupSize;

      for (int qi = 0; qi < seqLenQ; qi++) {
        // S = Q @ K^T
        std::vector<float> scores(seqLenKV);
        float maxScore = -1e9f;

        for (int kj = 0; kj < seqLenKV; kj++) {
          if (isCausal && kj > qi) {
            scores[kj] = -1e9f;
          } else {
            float dot = 0.0f;
            for (int d = 0; d < headDim; d++) {
              int qIdx = ((b * seqLenQ + qi) * numHeadsQ + hq) * headDim + d;
              int kIdx = ((b * seqLenKV + kj) * numHeadsKV + hkv) * headDim + d;
              dot += Q[qIdx] * K[kIdx];
            }
            scores[kj] = dot * scale;
          }
          maxScore = std::max(maxScore, scores[kj]);
        }

        // softmax
        float sumExp = 0.0f;
        for (int kj = 0; kj < seqLenKV; kj++) {
          scores[kj] = std::exp(scores[kj] - maxScore);
          sumExp += scores[kj];
        }
        for (int kj = 0; kj < seqLenKV; kj++) {
          scores[kj] /= sumExp;
        }

        // O += P @ V
        for (int d = 0; d < headDim; d++) {
          float val = 0.0f;
          for (int kj = 0; kj < seqLenKV; kj++) {
            int vIdx = ((b * seqLenKV + kj) * numHeadsKV + hkv) * headDim + d;
            val += scores[kj] * V[vIdx];
          }
          int oIdx = ((b * seqLenQ + qi) * numHeadsQ + hq) * headDim + d;
          O[oIdx] = val;
        }
      }
    }
  }
}

inline void cpuAttentionVarLen(const float* Q, const float* K, const float* V, float* O, const int* cuSeqLensQ,
                               const int* cuSeqLensKV, int batchSize, int numHeadsQ, int numHeadsKV, int headDim,
                               bool isCausal) {
  int groupSize = numHeadsQ / numHeadsKV;
  float scale = 1.0f / std::sqrt(static_cast<float>(headDim));

  for (int b = 0; b < batchSize; b++) {
    int seqStartQ = cuSeqLensQ[b];
    int seqEndQ = cuSeqLensQ[b + 1];
    int seqLenQ = seqEndQ - seqStartQ;

    int seqStartKV = cuSeqLensKV[b];
    int seqEndKV = cuSeqLensKV[b + 1];
    int seqLenKV = seqEndKV - seqStartKV;

    for (int hq = 0; hq < numHeadsQ; hq++) {
      int hkv = hq / groupSize;

      for (int qi = 0; qi < seqLenQ; qi++) {
        // S = Q @ K^T
        std::vector<float> scores(seqLenKV);
        float maxScore = -1e9f;

        for (int kj = 0; kj < seqLenKV; kj++) {
          if (isCausal && kj > qi) {
            scores[kj] = -1e9f;
          } else {
            float dot = 0.0f;
            for (int d = 0; d < headDim; d++) {
              int qIdx = (seqStartQ + qi) * numHeadsQ * headDim + hq * headDim + d;
              int kIdx = (seqStartKV + kj) * numHeadsKV * headDim + hkv * headDim + d;
              dot += Q[qIdx] * K[kIdx];
            }
            scores[kj] = dot * scale;
          }
          maxScore = std::max(maxScore, scores[kj]);
        }

        // softmax
        float sumExp = 0.0f;
        for (int kj = 0; kj < seqLenKV; kj++) {
          scores[kj] = std::exp(scores[kj] - maxScore);
          sumExp += scores[kj];
        }
        for (int kj = 0; kj < seqLenKV; kj++) {
          scores[kj] /= sumExp;
        }

        // O += P @ V
        for (int d = 0; d < headDim; d++) {
          float val = 0.0f;
          for (int kj = 0; kj < seqLenKV; kj++) {
            int vIdx = (seqStartKV + kj) * numHeadsKV * headDim + hkv * headDim + d;
            val += scores[kj] * V[vIdx];
          }
          int oIdx = (seqStartQ + qi) * numHeadsQ * headDim + hq * headDim + d;
          O[oIdx] = val;
        }
      }
    }
  }
}

}  // namespace tfa::test
