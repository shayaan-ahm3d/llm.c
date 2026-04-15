/*
This file trains the GPT-2 model.
This version is the clean, minimal, reference. As such:
- it runs on CPU.
- it does not make the code too complex; it is readable.
- it does not use any processor-specific instructions, intrinsics and such.
- it _does_ use a few OpenMP pragmas because this is a large speedup at very low cost
There will be other versions of this code that specialize it and make it fast.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>
#ifdef OMP
#include <omp.h>
#endif
#include <blis.h>
// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#include "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
#include "llmc/dataloader.h"
// defines: logger_init, logger_log_eval
#include "llmc/logger.h"

enum Mode {
	TRAIN_VAL = 0,
	INFERENCE = 1
};

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size
/*
void dequantize(QuantizedTensor *qx, float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}

void quantize(QuantizedTensor *qx, float* x, int n) {
    int num_groups = n / GS;
    float Q_MAX = 127.0f;

    for (int group = 0; group < num_groups; group++) {

        // find the max absolute value in the current group
        float wmax = 0.0;
        for (int i = 0; i < GS; i++) {
            float val = fabs(x[group * GS + i]);
            if (val > wmax) {
                wmax = val;
            }
        }

        // calculate and write the scaling factor
        float scale = wmax / Q_MAX;
        qx->s[group] = scale;

        // calculate and write the quantized values
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale; // scale
            int8_t quantized = (int8_t) round(quant_value); // round and clamp
            qx->q[group * GS + i] = quantized;
        }
    }
}
*/
void encoder_forward(
	float* output,
    const int* tokenIdsAtPosition,
    const float* weightTokenEmbeddings,
    const float* weightPositionalEmbeddings,
    const int batchSize,
    const int sequenceLength,
    const int dimensions,
    const enum Mode mode,
    const int currentToken
) {
    // output is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // tokenIdsAtPosition is (B,T) of integers, holding the token ids at each (b,t) position
    // weightTokenEmbeddings is (V,C) of token embeddings
    // weightPositionalEmbeddings is (maxT,C) of position embeddings
    // currentToken should be 0 for training and validation, but the current token for inference
    // sequenceLength should be 1 for KV cached inference, but full sequence length for training/validation
    // can process all tokens in parallel

    switch (mode) {
    	case TRAIN_VAL: {
	   		#pragma omp parallel for collapse(2)
	        for (int sequence = 0; sequence < batchSize; sequence++) {
	            for (int token = 0; token < sequenceLength; token++) {
	                // seek to the output position in out[b,t,:]
	                float* outputPosition = output + sequence*sequenceLength*dimensions + token*dimensions;
	                // get the index of the token at inp[b, t]
	                const int tokenIndex = tokenIdsAtPosition[sequence*sequenceLength + token];
	                // seek to the position in wte corresponding to the token
	                const float* wte_ix = weightTokenEmbeddings + tokenIndex*dimensions;
	                // seek to the position in wpe corresponding to the position
	                const float* wpe_t = weightPositionalEmbeddings + token*dimensions;
	                // add the two vectors and store the result in out[b,t,:]

	                for (int i = 0; i < dimensions; i++) {
	                    outputPosition[i] = wte_ix[i] + wpe_t[i];
	                }
	            }
	        }
     		break;
     	}
      	case INFERENCE: {
       		#pragma omp parallel for
	        for (int sequence = 0; sequence < batchSize; sequence++) {
                // seek to the output position in out[b,t,:]
                float* outputPosition = output + sequence*sequenceLength*dimensions + currentToken*dimensions;
                // get the index of the token at inp[b, t]
                const int tokenIndex = tokenIdsAtPosition[sequence*sequenceLength + currentToken];
                // seek to the position in wte corresponding to the token
                const float* wte_ix = weightTokenEmbeddings + tokenIndex*dimensions;
                // seek to the position in wpe corresponding to the position
                const float* wpe_t = weightPositionalEmbeddings + currentToken*dimensions;
                // add the two vectors and store the result in out[b,t,:]

                for (int i = 0; i < dimensions; i++) {
                    outputPosition[i] = wte_ix[i] + wpe_t[i];
                }
	        }
            break;
       	}
    }
}

void encoder_backward(float* dwte, float* dwpe,
                      float* dout, int* inp,
                      int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            int ix = inp[b * T + t];
            float* dwte_ix = dwte + ix * C;
            float* dwpe_t = dwpe + t * C;
            for (int i = 0; i < C; i++) {
                float d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}

void layernorm_forward(
	float* output,
	float* mean,
	float* rstd,
	const float* input,
	const float* weight,
	const float* bias,
	const int batchSize,
	const int sequenceLength,
	const int dimensions,
	const enum Mode mode,
	const int currentToken
) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    const float eps = 1e-5f;

    switch (mode) {
    	case TRAIN_VAL: {
    		#pragma omp parallel for collapse(2)
         	for (int sequence = 0; sequence < batchSize; sequence++) {
             	for (int token = 0; token < sequenceLength; token++) {
	              	// seek to the input position inp[b,t,:]
	                const float* x = input + sequence*sequenceLength*dimensions + token*dimensions;
	                // calculate the mean
	                float m = 0.0f;

	                for (int i = 0; i < dimensions; i++) {
	                    m += x[i];
	                }
	                m /= dimensions;
	                // calculate the variance (without any bias correction)
	                float v = 0.0f;

	                for (int i = 0; i < dimensions; i++) {
	                    const float xshift = x[i] - m;
	                    v += xshift * xshift;
	                }
	                v /= dimensions;
	                // calculate the rstd (reciprocal standard deviation)
	                const float s = 1.0f / sqrtf(v + eps);
	                // seek to the output position in out[b,t,:]
	                float* out_bt = output + sequence*sequenceLength*dimensions + token*dimensions;

	                for (int i = 0; i < dimensions; i++) {
	                    const float n = (s * (x[i] - m)); // normalize
	                    const float o = n * weight[i] + bias[i]; // scale and shift
	                    out_bt[i] = o; // write
	                }
	                // cache the mean and rstd for the backward pass later
	                mean[sequence*sequenceLength + token] = m;
	                rstd[sequence*sequenceLength + token] = s;
              	}
          	}
          	break;
     	}
      	case INFERENCE: {
       		#pragma omp parallel for
           	for (int sequence = 0; sequence < batchSize; sequence++) {
		       	// seek to the input position inp[b,t,:]
		        const float* x = input + sequence*sequenceLength*dimensions + currentToken*dimensions;
		        // calculate the mean
		        float m = 0.0f;

		        for (int i = 0; i < dimensions; i++) {
		            m += x[i];
		        }
		        m /= dimensions;
		        // calculate the variance (without any bias correction)
		        float v = 0.0f;

		        for (int i = 0; i < dimensions; i++) {
		            const float xshift = x[i] - m;
		            v += xshift * xshift;
		        }
		        v /= dimensions;
		        // calculate the rstd (reciprocal standard deviation)
		        const float s = 1.0f / sqrtf(v + eps);
		        // seek to the output position in out[b,t,:]
		        float* out_bt = output + sequence*sequenceLength*dimensions + currentToken*dimensions;

		        for (int i = 0; i < dimensions; i++) {
		            const float n = (s * (x[i] - m)); // normalize
		            const float o = n * weight[i] + bias[i]; // scale and shift
		            out_bt[i] = o; // write
		        }
		        // cache the mean and rstd for the backward pass later
		        mean[sequence*sequenceLength + currentToken] = m;
		        rstd[sequence*sequenceLength + currentToken] = s;
           	}
           	break;
       	}
    }
}

void layernorm_backward(float* dinp, float* dweight, float* dbias,
                        float* dout, float* inp, float* weight, float* mean, float* rstd,
                        int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            float* inp_bt = inp + b * T * C + t * C;
            float* dinp_bt = dinp + b * T * C + t * C;
            float mean_bt = mean[b * T + t];
            float rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

void matmul_forward_naive(
	float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const int batchSize,
    const int sequenceLength,
    const int dimensions,
    const int outputDimensions,
    const enum Mode mode,
    const int currentToken
) {
    // input is (B,T,C), weight is (OC, C), bias is (OC)
    // output will be (B,T,OC)
    // combining weights and biases would mean I might have to change the initial allocations as all the tensors are contiguous
    // but weight and bias are adjacent there so would I?

    switch (mode) {
    	case TRAIN_VAL: {
	   		for (int sequence = 0; sequence < batchSize; ++sequence) {
	          	if (bias != NULL) {
	           		// fill matrix with bias terms already
	          		for (int token = 0; token < sequenceLength; ++token) {
	          			const int index = sequence*sequenceLength*outputDimensions + token*outputDimensions;
	          			memcpy(output+index, bias, outputDimensions*sizeof(float));
	             	}
	           	}
	         	//    C   =    A   *    B      + C
	         	// output = input * weight.T + output
	         	cblas_sgemm(CblasRowMajor,
	          	CblasNoTrans,
	           	CblasTrans,
	          	sequenceLength,
	           	outputDimensions,
	            dimensions,
	          	1.f,
	           	input + sequence*sequenceLength*dimensions,
	            dimensions,
	            weight,
	            dimensions,
	            1.f,
	            output + sequence*sequenceLength*outputDimensions,
	            outputDimensions);
	        }
     		break;
     	}
     	case INFERENCE: {
      		for (int sequence = 0; sequence < batchSize; ++sequence) {
	          	if (bias != NULL) {
	           		// fill matrix with bias terms already
	          		const int index = sequence*sequenceLength*outputDimensions + currentToken*outputDimensions;
	          		memcpy(output + index, bias, outputDimensions*sizeof(float));
	           	}
	         	//    C   =    A   *    B      + C
	         	// output = input * weight.T + output
	         	cblas_sgemm(CblasRowMajor,
	          	CblasNoTrans,
	           	CblasTrans,
	          	1, // single token only
	           	outputDimensions,
	            dimensions,
	          	1.f,
	           	input + sequence*sequenceLength*dimensions + currentToken*dimensions,
	            dimensions,
	            weight,
	            dimensions,
	            1.f,
	            output + sequence*sequenceLength*outputDimensions + currentToken*outputDimensions,
	            outputDimensions);
	        }
      		break;
      	}
    }

    /*for (int t = 0; t < sequenceLength; t++) {
            int bt = batch * sequenceLength + t;
            for (int o = 0; o < outputChannels; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                for (int i = 0; i < dimensions; i++) {
                    val += inp[bt*dimensions + i] * weight[o*dimensions + i];
                }
                out[bt * outputChannels + o] = val;
            }
        }*/

}

void matmul_forward(
	float* out,
    const float* inp,
    const float* weight,
    const float* bias,
    const int batchSize,
    const int sequenceLength,
    const int dimensions,
    const int outputDimensions,
    const enum Mode mode,
    const int currentToken
) {
    // most of the running time is spent here and in matmul_backward
    // therefore, the implementation below is very mildly optimized
    // this function is otherwise identical to that of matmul_forward_naive()
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    // make sure the tiled loop will be correct or fallback to naive version

    matmul_forward_naive(out, inp, weight, bias, batchSize, sequenceLength, dimensions, outputDimensions, mode, currentToken);
    return;
    const int LOOP_UNROLL = 8;
    if (batchSize*sequenceLength % LOOP_UNROLL != 0) {
        matmul_forward_naive(out, inp, weight, bias, batchSize, sequenceLength, dimensions, outputDimensions, mode, currentToken);
        return;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    #pragma omp parallel for
    for (int obt = 0; obt < batchSize * sequenceLength; obt += LOOP_UNROLL) {
        for (int o = 0; o < outputDimensions; o++) {
            // we'll keep LOOP_UNROLL many results in registers
            float result[LOOP_UNROLL];
            // initialize the bias, if it exists
            for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                result[ibt] = (bias != NULL) ? bias[o] : 0.0f;
            }
            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
            for (int i = 0; i < dimensions; i++) {
                float w = weight[i + o * dimensions];
                for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                    int bt = obt + ibt;
                    result[ibt] += inp[bt * dimensions + i] * w;
                }
            }
            // write back results to main memory
            for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                int bt = obt + ibt;
                out[bt * outputDimensions + o] = result[ibt];
            }
        }
    }
}

void matmul_backward(
	float* dinp,
	float* dweight,
	float* dbias,
    const float* dout,
    const float* inp,
    float* weight,
    int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* dout_bt = dout + b * T * OC + t * OC;
            float* dinp_bt = dinp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                for (int i = 0; i < C; i++) {
                	float* wrow = weight + o*C;
                    const float d = dout_bt[o];
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }
    // backward into weight/bias, parallelize over output channels OC
    #pragma omp parallel for
    for (int o = 0; o < OC; o++) {
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                const float* dout_bt = dout + b * T * OC + t * OC;
                const float* inp_bt = inp + b * T * C + t * C;
                float* dwrow = dweight + o*C;
                float d = dout_bt[o];
                if (dbias != NULL) { dbias[o] += d; }
                for (int i = 0; i < C; i++) {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
    }
}

// use with key-value cache to do partial update of key and value, reducing duplicate calculations
// pass in cache pointer to correct location within cache for that layer and token
void cached_matmul_forward(
	float* const restrict output,
	const float* input,
	const float* const restrict weight,
	const float* const restrict bias,
    const int batchSize,
    const int sequenceLength,
    const int dimensions,
    const int outputDimensions,
    const enum Mode mode,
    const int currentToken
) {
	assert(mode == INFERENCE);
    assert(dimensions == outputDimensions);
    for (int sequence = 0; sequence < batchSize; ++sequence) {
    	// use 3*dimensions so it goes over the QKV to get to the next token, otherwise would index into wrong thing
    	// fill that cache line with the bias values already
        memcpy(output + sequence*sequenceLength*3*dimensions + currentToken*3*dimensions, bias, dimensions*sizeof(float));

        // only adding one token therefore do matrix-vector multiplication and append the result to K & V
        //   y   =        A        *      x
        // cache = weight(K or V) * input(token)
        cblas_sgemv(CblasRowMajor, // row-major in C
        CblasNoTrans,
        outputDimensions, // rows of weight matrix W
        dimensions, // cols of weight matrix W
        1.f,
        weight, // weight matrix W(k) or W(v)
        dimensions,
        input + sequence*sequenceLength*dimensions + currentToken*dimensions, // current token
        1, // increment for token vector
        1.f,
        output + sequence*sequenceLength*3*dimensions + currentToken*3*dimensions, // output to the cache
        1); // increment for cache vector
    }
}

void attention_forward(
	float* out,
	float* preatt,
	float* att,
    const float* qkv,
    const int batchSize,
    const int sequenceLength,
    const int dimensions,
    const int numHeads,
    const enum Mode mode,
    const int currentToken
) {
    // qkv is (batchSize, sequenceLength, 3*dimensions) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    const int C3 = dimensions*3;
    const int headSize = dimensions / numHeads; // split across number of heads
    const float scale = 1.0 / sqrtf(headSize);

    switch (mode) {
    	case TRAIN_VAL: {
	  		// already does parallel attention prefill
	        #pragma omp parallel for collapse(3)
	        for (int sequence = 0; sequence < batchSize; sequence++) {
	            for (int token = 0; token <  sequenceLength; token++) {
	                for (int head = 0; head < numHeads; head++) {
	                    const float* query_t = qkv + sequence*sequenceLength*C3 + token*C3 + head*headSize;
	                    float* preatt_bth = preatt + sequence*numHeads*sequenceLength*sequenceLength + head*sequenceLength*sequenceLength + token*sequenceLength;
	                    float* att_bth = att + sequence*numHeads*sequenceLength*sequenceLength + head*sequenceLength*sequenceLength + token*sequenceLength;

	                    // pass 1: calculate query dot key and maxval
	                    float maxval = -10000.0f; // TODO something better
	                    // this loop iterates over all previous elements in K

	                    for (int t2 = 0; t2 <= token; t2++) {
	                    	// query is t but key is t2 as key requires full history of previous tokens, whereas query just requires current
	                        const float* key_t2 = qkv + sequence*sequenceLength*C3 + t2*C3 + head*headSize + dimensions; // +dimensions because it's key

	                        // (query_t) dot (key_t2)
	                        float val = 0.0f;
	                        for (int i = 0; i < headSize; i++) {
	                            val += query_t[i] * key_t2[i];
	                        }
	                        val *= scale;
	                        if (val > maxval) {
	                            maxval = val;
	                        }

	                        preatt_bth[t2] = val;
	                    }

	                    // pass 2: calculate the exp and keep track of sum
	                    // maxval is being calculated and subtracted only for numerical stability
	                    float expsum = 0.0f;

	                    for (int t2 = 0; t2 <= token; t2++) {
	                        float expv = expf(preatt_bth[t2] - maxval);
	                        expsum += expv;
	                        att_bth[t2] = expv;
	                    }
	                    float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

	                    // pass 3: normalize to get the softmax

	                    for (int t2 = 0; t2 <= sequenceLength - 1; t2++) {
	                        if (t2 <= token) {
	                            att_bth[t2] *= expsum_inv;
	                        } else {
	                            // causal attention mask. not strictly necessary to set to zero here
	                            // only doing this explicitly for debugging and checking to PyTorch
	                            att_bth[t2] = 0.0f;
	                        }
	                    }

	                    // pass 4: accumulate weighted values into the output of attention
	                    float* out_bth = out + sequence*sequenceLength*dimensions + token*dimensions + head*headSize;

                        for (int i = 0; i < headSize; i++) {
	                    	out_bth[i] = 0.0f;
	                    }
	                    for (int t2 = 0; t2 <= token; t2++) {
	                        for (int i = 0; i < headSize; i++) {
	                            const float att_btht2 = att_bth[t2];
	                            const float* value_t2 = qkv + sequence*sequenceLength*C3 + t2*C3 + head*headSize + dimensions*2; // +dimensions*2 because it's value
	                            out_bth[i] += att_btht2 * value_t2[i];
	                        }
	                    }
	                }
	            }
	        }
     		break;
     	}
      	case INFERENCE: {
     		#pragma omp parallel for collapse(2)
	        for (int sequence = 0; sequence < batchSize; sequence++) {
                for (int head = 0; head < numHeads; head++) {
                    const float* query_t = qkv + sequence*sequenceLength*C3 + currentToken*C3 + head*headSize;
                    float* preatt_bth = preatt + sequence*numHeads*sequenceLength*sequenceLength + head*sequenceLength*sequenceLength + currentToken*sequenceLength;
                    float* att_bth = att + sequence*numHeads*sequenceLength*sequenceLength + head*sequenceLength*sequenceLength + currentToken*sequenceLength;

                    // pass 1: calculate query dot key and maxval
                    float maxval = -10000.0f; // TODO something better
                    // this loop iterates over all previous elements in K

                    for (int t2 = 0; t2 <= currentToken; t2++) {
                    	// query is t but key is t2 as key requires full history of previous tokens, whereas query just requires current
                        const float* key_t2 = qkv + sequence*sequenceLength*C3 + t2*C3 + head*headSize + dimensions; // +dimensions because it's key

                        // (query_t) dot (key_t2)
                        float val = 0.0f;
                        for (int i = 0; i < headSize; i++) {
                            val += query_t[i] * key_t2[i];
                        }
                        val *= scale;
                        if (val > maxval) {
                            maxval = val;
                        }

                        preatt_bth[t2] = val;
                    }

                    // pass 2: calculate the exp and keep track of sum
                    // maxval is being calculated and subtracted only for numerical stability
                    float expsum = 0.0f;

                    for (int t2 = 0; t2 <= currentToken; t2++) {
                        float expv = expf(preatt_bth[t2] - maxval);
                        expsum += expv;
                        att_bth[t2] = expv;
                    }
                    float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                    // pass 3: normalize to get the softmax

                    for (int t2 = 0; t2 <= currentToken; t2++) {
                        if (t2 <= currentToken) {
                            att_bth[t2] *= expsum_inv;
                        } else { // not used in inference
                            // causal attention mask. not strictly necessary to set to zero here
                            // only doing this explicitly for debugging and checking to PyTorch
                            att_bth[t2] = 0.0f;
                        }
                    }

                    // pass 4: accumulate weighted values into the output of attention
                    float* out_bth = out + sequence*sequenceLength*dimensions + currentToken*dimensions + head*headSize;

                    for (int i = 0; i < headSize; i++) {
                    	out_bth[i] = 0.0f;
                    }
                    for (int t2 = 0; t2 <= currentToken; t2++) {
                        for (int i = 0; i < headSize; i++) {
                            const float* value_t2 = qkv + sequence*sequenceLength*C3 + t2*C3 + head*headSize + dimensions*2; // +dimensions*2 because it's value
                            const float att_btht2 = att_bth[t2];
                            out_bth[i] += att_btht2 * value_t2[i];
                        }
                    }
                }
	        }
       		break;
       	}
    }



}

void attention_backward(float* dinp, float* dpreatt, float* datt,
                        float* dout, float* inp, float* att,
                        int B, int T, int C, int NH) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.f / sqrtf(hs);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;
                float* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                float* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                float* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                float* dout_bth = dout + b * T * C + t * C + h * hs;
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for (int t2 = 0; t2 <= t; t2++) {
                    for (int t3 = 0; t3 <= t; t3++) {
                        float indicator = t2 == t3 ? 1.0f : 0.0f;
                        float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    float* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        // so now we have:
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale;
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
                    }
                }
            }
        }
    }
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward(
	float* out,
	const float* const inp,
	const int batchSize,
	const int sequenceLength,
	const int dimensions,
	const enum Mode mode,
	const int currentToken
) {
	// (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
	switch (mode) {
		case TRAIN_VAL: {

    		for (int i = 0; i < batchSize*sequenceLength*dimensions; i++) {
        		const float x = inp[i];
          		const float cube = 0.044715f * x * x * x;
            	out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
      		}
			break;
		}
		case INFERENCE: {
    		for (int sequence = 0; sequence < batchSize; sequence++) {
      			for (int dim = 0; dim < dimensions; dim++) {
         			const int index = sequence*sequenceLength*dimensions + currentToken*dimensions + dim;
            		const float x = inp[index];
                    const float cube = 0.044715f * x * x * x;
                    out[index] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
         		}
      		}
			break;
		}
	}

}

// we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it (#168)
#pragma float_control(precise, on, push)
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-finite-math-only")))
#endif
void gelu_backward(float* dinp, float* inp, float* dout, int N) {
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] += local_grad * dout[i];
    }
}
#pragma float_control(pop)

void residual_forward(
	float* out,
	const float* inp1,
	const float* inp2,
	const int batchSize,
	const int sequenceLength,
	const int dimensions,
	const enum Mode mode,
	const int currentToken
) {
	switch (mode) {
		case TRAIN_VAL: {

    		for (int i = 0; i < batchSize*sequenceLength*dimensions; i++) {
        		out[i] = inp1[i] + inp2[i];
      		}
			break;
		}
		case INFERENCE: {
			for (int sequence = 0; sequence < batchSize; ++sequence) {
				for (int dim = 0; dim < dimensions; ++dim) {
					const int index = sequence*sequenceLength*dimensions + currentToken*dimensions + dim;
        			out[index] = inp1[index] + inp2[index];
      			}
			}
			break;
		}
	}
}

void residual_backward(float* dinp1, float* dinp2, float* dout, int N) {
    for (int i = 0; i < N; i++) {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
}

void softmax_forward(
	float* probs,
	const float* logits,
	const int batchSize,
	const int sequenceLength,
	const int vocabSize,
	const int paddedVocabSize,
	const enum Mode mode,
	const int currentToken
) {
    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257

    switch (mode) {
    	case TRAIN_VAL: {
     		#pragma omp parallel for collapse(2)
	        for (int sequence = 0; sequence < batchSize; sequence++) {
	            for (int token = 0; token < sequenceLength; token++) {
	                // probs <- softmax(logits)
	                const float* logits_bt = logits + sequence*sequenceLength*paddedVocabSize + token*paddedVocabSize;
	                float* probs_bt = probs + sequence*sequenceLength*paddedVocabSize + token*paddedVocabSize;

	                // maxval is only calculated and subtracted for numerical stability
	                float maxval = -10000.0f; // TODO something better

	                for (int i = 0; i < vocabSize; i++) {
	                    if (logits_bt[i] > maxval) {
	                        maxval = logits_bt[i];
	                    }
	                }
	                float sum = 0.0f;

	                for (int i = 0; i < vocabSize; i++) {
	                    probs_bt[i] = expf(logits_bt[i] - maxval);
	                    sum += probs_bt[i];
	                }
	                // note we only loop to V, leaving the padded dimensions

	                for (int i = 0; i < vocabSize; i++) {
	                    probs_bt[i] /= sum;
	                }
	                // for extra super safety we may wish to include this too,
	                // forcing the probabilities here to be zero, but it shouldn't matter

	                for (int i = vocabSize; i < paddedVocabSize; i++) {
	                    probs_bt[i] = 0.0f;
	                }
	            }
	        }
     		break;
     	}
      	case INFERENCE: {
     		#pragma omp parallel for
	        for (int sequence = 0; sequence < batchSize; sequence++) {
                // probs <- softmax(logits)
                const float* logits_bt = logits + sequence*sequenceLength*paddedVocabSize + currentToken*paddedVocabSize;
                float* probs_bt = probs + sequence*sequenceLength*paddedVocabSize + currentToken*paddedVocabSize;

                // maxval is only calculated and subtracted for numerical stability
                float maxval = -10000.0f; // TODO something better

                for (int i = 0; i < vocabSize; i++) {
                    if (logits_bt[i] > maxval) {
                        maxval = logits_bt[i];
                    }
                }
                float sum = 0.0f;

                for (int i = 0; i < vocabSize; i++) {
                    probs_bt[i] = expf(logits_bt[i] - maxval);
                    sum += probs_bt[i];
                }
                // note we only loop to V, leaving the padded dimensions

                for (int i = 0; i < vocabSize; i++) {
                    probs_bt[i] /= sum;
                }
                // for extra super safety we may wish to include this too,
                // forcing the probabilities here to be zero, but it shouldn't matter

                for (int i = vocabSize; i < paddedVocabSize; i++) {
                    probs_bt[i] = 0.0f;
                }
	        }
       		break;
       	}
    }
}

void crossentropy_forward(
	float* losses,
    const float* probs,
    const int* targets,
    const int batchSize,
    const int sequenceLength,
    const int paddedVocabSize,
    const enum Mode mode,
    const int currentToken
) {
    // output: losses is (batchSize,sequenceLength) of the individual losses at each position
    // input: probs are (batchSize,sequenceLength,paddedVocabSize) of the probabilities
    // input: targets is (batchSize,sequenceLength) of integers giving the correct index in logits - correct index is 1, rest are 0
    #pragma omp parallel for
    for (int sequence = 0; sequence < batchSize; sequence++) {
    	switch (mode) {
     		case TRAIN_VAL: {

       			for (int token = 0; token < sequenceLength; token++) {
	            	// loss = -log(probs[target])
		            const float* probs_bt = probs + sequence*sequenceLength*paddedVocabSize + token*paddedVocabSize;
		            const int targetToken = targets[sequence*sequenceLength + token];
		            losses[sequence*sequenceLength + token] = -logf(probs_bt[targetToken]);
          		}
          		break;
       		}
       		case INFERENCE: { // only process currentToken
	            // loss = -log(probs[target])
		        const float* probs_bt = probs + sequence*sequenceLength*paddedVocabSize + currentToken*paddedVocabSize;
		        const int targetToken = targets[sequence*sequenceLength + currentToken];
		        losses[sequence*sequenceLength + currentToken] = -logf(probs_bt[targetToken]);
				break;
         	}
     	}
    }
}

void crossentropy_softmax_backward(float* dlogits,
                           float* dlosses, float* probs, int* targets,
                           int B, int T, int V, int Vp) {
    // backwards through both softmax and crossentropy
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dlogits_bt = dlogits + b * T * Vp + t * Vp;
            float* probs_bt = probs + b * T * Vp + t * Vp;
            float dloss = dlosses[b * T + t];
            int ix = targets[b * T + t];
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so gradient there stays at zero
            for (int i = 0; i < V; i++) {
                float p = probs_bt[i];
                float indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

typedef struct {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int padded_vocab_size; // padded to e.g. %128==0, 50304
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768 (embedding vector)
} GPT2Config;

// the parameters of the model
#define NUM_PARAMETER_TENSORS 16
typedef struct {
    float* wte; // (V, C)
    float* wpe; // (maxT, C)
    float* ln1w; // (L, C)
    float* ln1b; // (L, C)
    float* qkvw; // (L, 3*C, C)
    float* qkvb; // (L, 3*C)
    float* attprojw; // (L, C, C)
    float* attprojb; // (L, C)
    float* ln2w; // (L, C)
    float* ln2b; // (L, C)
    float* fcw; // (L, 4*C, C)
    float* fcb; // (L, 4*C)
    float* fcprojw; // (L, C, 4*C)
    float* fcprojb; // (L, C)
    float* lnfw; // (C)
    float* lnfb; // (C)
} ParameterTensors;

void fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;
    param_sizes[0] = Vp * C; // wte
    param_sizes[1] = maxT * C; // wpe
    param_sizes[2] = L * C; // ln1w
    param_sizes[3] = L * C; // ln1b
    param_sizes[4] = L * (3 * C) * C; // qkvw: input is divided across channels so you don't need to multiply by num of heads
    param_sizes[5] = L * (3 * C); // qkvb
    param_sizes[6] = L * C * C; // attprojw
    param_sizes[7] = L * C; // attprojb
    param_sizes[8] = L * C; // ln2w
    param_sizes[9] = L * C; // ln2b
    param_sizes[10] = L * (4 * C) * C; // fcw
    param_sizes[11] = L * (4 * C); // fcb
    param_sizes[12] = L * C * (4 * C); // fcprojw
    param_sizes[13] = L * C; // fcprojb
    param_sizes[14] = C; // lnfw
    param_sizes[15] = C; // lnfb
}

// allocate memory for the parameters and point the individual tensors to the right places
float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes) {
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    // malloc all parameters all at once
    float* params_memory = (float*)mallocCheck(num_parameters * sizeof(float));
    // assign all the tensors
    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    float* params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

#define NUM_ACTIVATION_TENSORS 23
typedef struct {
    float* encoded; // (B, T, C)
    float* ln1; // (L, B, T, C)
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    float* qkv; // (L, B, T, 3*C)
    float* atty; // (L, B, T, C)
    float* preatt; // (L, B, NH, T, T)
    float* att; // (L, B, NH, T, T)
    float* attproj; // (L, B, T, C)
    float* residual2; // (L, B, T, C)
    float* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    float* fch; // (L, B, T, 4*C)
    float* fch_gelu; // (L, B, T, 4*C)
    float* fcproj; // (L, B, T, C)
    float* residual3; // (L, B, T, C)
    float* lnf; // (B, T, C)
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)
    float* logits; // (B, T, V(p))
    float* probs; // (B, T, V(p))
    float* losses; // (B, T)
} ActivationTensors;

void fill_in_activation_sizes(size_t* act_sizes, GPT2Config config, int B, int T) {
    size_t C = config.channels;
    size_t NH = config.num_heads;
    size_t L = config.num_layers;
    size_t Vp = config.padded_vocab_size;
    act_sizes[0] = B * T * C; // encoded
    act_sizes[1] = L * B * T * C; // ln1
    act_sizes[2] = L * B * T; // ln1_mean
    act_sizes[3] = L * B * T; // ln1_rstd
    act_sizes[4] = L * B * T * 3 * C; // qkv
    act_sizes[5] = L * B * T * C; // atty
    act_sizes[6] = L * B * NH * T * T; // preatt
    act_sizes[7] = L * B * NH * T * T; // att
    act_sizes[8] = L * B * T * C; // attproj
    act_sizes[9] = L * B * T * C; // residual2
    act_sizes[10] = L * B * T * C; // ln2
    act_sizes[11] = L * B * T; // ln2_mean
    act_sizes[12] = L * B * T; // ln2_rstd
    act_sizes[13] = L * B * T * 4 * C; // fch
    act_sizes[14] = L * B * T * 4 * C; // fch_gelu
    act_sizes[15] = L * B * T * C; // fcproj
    act_sizes[16] = L * B * T * C; // residual3
    act_sizes[17] = B * T * C; // lnf
    act_sizes[18] = B * T; // lnf_mean
    act_sizes[19] = B * T; // lnf_rstd
    act_sizes[20] = B * T * Vp; // logits
    act_sizes[21] = B * T * Vp; // probs
    act_sizes[22] = B * T; // losses
}

typedef struct {
    GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float* params_memory;
    size_t num_parameters;
    // gradients of the weights
    ParameterTensors grads;
    float* grads_memory;
    // buffers for the AdamW optimizer
    float* m_memory;
    float* v_memory;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float* acts_memory;
    size_t num_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    float* grads_acts_memory;
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    int* inputs; // the input tokens for the current forward pass
    int* targets; // the target tokens for the current forward pass
    float mean_loss; // after a forward pass with targets, will be populated with the mean loss
} GPT2;

float* malloc_and_point_activations(GPT2* model, ActivationTensors* acts, size_t* act_sizes) {
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        num_activations += act_sizes[i];
    }
    model->num_activations = num_activations;
    float* acts_memory = (float*)mallocCheck(num_activations * sizeof(float));
    float** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses
    };
    float* acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path) {
    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { printf("Bad magic model file\n"); exit(1); }
    if (model_header[1] != 3) {
        printf("Bad version in model file\n");
        printf("---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(1);
    }

    // read in hyperparameters
    size_t maxT, V, Vp, L, NH, C; // size_t to prevent int overflow
    model->config.max_seq_len = maxT = model_header[2];
    model->config.vocab_size = V = model_header[3];
    model->config.num_layers = L = model_header[4];
    model->config.num_heads = NH = model_header[5];
    model->config.channels = C = model_header[6];
    model->config.padded_vocab_size = Vp = model_header[7];
    printf("[GPT-2]\n");
    printf("max_seq_len: %zu\n", maxT);
    printf("vocab_size: %zu\n", V);
    printf("padded_vocab_size: %zu\n", Vp);
    printf("num_layers: %zu\n", L);
    printf("num_heads: %zu\n", NH);
    printf("channels: %zu\n", C);

    // allocate space for all the parameters and read them in
    fill_in_parameter_sizes(model->param_sizes,  model->config);

    // count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    printf("num_parameters: %zu\n", num_parameters);
    model->num_parameters = num_parameters;

    // read in all the parameters from file
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    freadCheck(model->params_memory, sizeof(float), num_parameters, model_file);
    fcloseCheck(model_file);

    // other inits
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
}

typedef struct {
    float left;
    float right;
} Pair;

Pair compare_implementations(
    FILE* cpu,
    FILE* gpu,
    const int* targets, // from HellaSwag
    const int batchSize,
    const int sequenceLength,
    const int vocabSize,
    const int paddedVocabSize
) {
    (void)paddedVocabSize;
    const size_t N = batchSize*sequenceLength*vocabSize;

    float* cpuLogits = malloc(sizeof(float)*N);
    float* gpuLogits = malloc(sizeof(float)*N);

    float* cpuProbs = malloc(sizeof(float)*N);
    float* gpuProbs = malloc(sizeof(float)*N);
    float* cpuLosses = malloc(sizeof(float)*batchSize*sequenceLength);
    float* gpuLosses = malloc(sizeof(float)*batchSize*sequenceLength);

    // this only reads from the first sequence
    const size_t cpuRead = fread(cpuLogits, sizeof(float), N, cpu);
    const size_t gpuRead = fread(gpuLogits, sizeof(float), N, gpu);
    if (cpuRead != N || gpuRead != N) {
        fprintf(stderr, "Error: logits read mismatch (cpu=%zu, gpu=%zu, expected=%zu)\n", cpuRead, gpuRead, N);
        exit(1);
    }

    softmax_forward(cpuProbs, cpuLogits, batchSize, sequenceLength, vocabSize, vocabSize, TRAIN_VAL, 0);
    softmax_forward(gpuProbs, gpuLogits, batchSize, sequenceLength, vocabSize, vocabSize, TRAIN_VAL, 0);

    crossentropy_forward(cpuLosses, cpuProbs, targets, batchSize, sequenceLength, vocabSize, TRAIN_VAL, 0);
    crossentropy_forward(gpuLosses, gpuProbs, targets, batchSize, sequenceLength, vocabSize, TRAIN_VAL, 0);

    float cpuLoss = 0.f;
    float gpuLoss = 0.f;
    for (size_t i = 0; i < batchSize*sequenceLength; ++i) {
        cpuLoss += cpuLosses[i];
        gpuLoss += gpuLosses[i];
    }

    free(cpuLogits);
    free(gpuLogits);
    free(cpuProbs);
    free(gpuProbs);
    free(cpuLosses);
    free(gpuLosses);

    return (Pair){.left = cpuLoss, .right = gpuLoss};
}

void gpt2_forward(GPT2* model,
	const int* inputs,
	int* targets, // targets are optional and could be NULL
	const size_t batchSize,
	const size_t sequenceLength,
	const enum Mode mode,
	const size_t currentToken
) {
    // ensure the model was initialized or error out
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(1);
    }

    // convenience parameters (size_t to help prevent int overflow)
    size_t vocabSize = model->config.vocab_size;
    size_t paddedVocabSize = model->config.padded_vocab_size;
    size_t numLayers = model->config.num_layers;
    size_t numHeads = model->config.num_heads;
    size_t dimensions = model->config.channels;

    // validate inputs, all indices must be in the range [0, vocabSize)
    #pragma omp parallel for
    for (int i = 0; i < batchSize * sequenceLength; i++) {
        assert(0 <= inputs[i] && inputs[i] < vocabSize);
        if (targets != NULL) {
            assert(0 <= targets[i] && targets[i] < vocabSize);
        }
    }

    // allocate space for all the activations if needed (done here, lazily)
    if (model->acts_memory == NULL) {
        // record the current B,T as well
        model->batch_size = batchSize;
        model->seq_len = sequenceLength;
        // and now allocate the space
        fill_in_activation_sizes(model->act_sizes, model->config, batchSize, sequenceLength);
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->act_sizes[i];
        }
        printf("num_activations: %zu\n", num_activations);
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(model, &model->acts, model->act_sizes);
        // also create memory for caching inputs and targets
        model->inputs = (int*)mallocCheck(batchSize * sequenceLength * sizeof(int));
        model->targets = (int*)mallocCheck(batchSize * sequenceLength * sizeof(int)); // might be unused if we never have targets but it's small
    } else {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        if (batchSize != model->batch_size || sequenceLength != model->seq_len) {
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, (int)batchSize, (int)sequenceLength);
            exit(EXIT_FAILURE);
        }
    }

    // cache the inputs/targets
    memcpy(model->inputs, inputs, batchSize * sequenceLength * sizeof(int));
    if (targets != NULL) {
        memcpy(model->targets, targets, batchSize * sequenceLength * sizeof(int));
    }

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    float* residual;

    /*size_t tokenCount;
    size_t tokenOffset;

    switch (mode) {
    	case TRAIN_VAL: {
     		tokenCount = sequenceLength;
     		tokenOffset = 0;
     		break;
     	}
      	case INFERENCE: {
       		tokenCount = 1; // only processing single token for decode stage
         	tokenOffset = currentToken; // index into that token's memory location within each tensor
       		break;
       	}
    }

    size_t offset = tokenOffset * dimensions; // that token's memory location within a layer
    size_t offsetAttention = tokenOffset * 3 * dimensions; // x3 for QKV
    size_t offsetFeedForwardLayers = tokenOffset * 4 * dimensions; // x4 for fully connected and GELU
    */


    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, batchSize, sequenceLength, dimensions, mode, currentToken); // encoding goes into residual[0]

    for (int l = 0; l < numLayers; l++) {
        residual = (l == 0 ? acts.encoded : acts.residual3 + (l-1) * batchSize * sequenceLength * dimensions);

        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * dimensions;
        float* l_ln1b = params.ln1b + l * dimensions;
        float* l_qkvw = params.qkvw + l * 3*dimensions * dimensions; // for each layer, as in architecture diagram
        float* l_qkvb = params.qkvb + l * 3*dimensions;
        float* l_attprojw = params.attprojw + l * dimensions * dimensions;
        float* l_attprojb = params.attprojb + l * dimensions;
        float* l_ln2w = params.ln2w + l * dimensions;
        float* l_ln2b = params.ln2b + l * dimensions;
        float* l_fcw = params.fcw + l * 4*dimensions * dimensions;
        float* l_fcb = params.fcb + l * 4*dimensions;
        float* l_fcprojw = params.fcprojw + l * dimensions * 4*dimensions;
        float* l_fcprojb = params.fcprojb + l * dimensions;

        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * batchSize * sequenceLength * dimensions;
        float* l_ln1_mean = acts.ln1_mean + l * batchSize * sequenceLength;
        float* l_ln1_rstd = acts.ln1_rstd + l * batchSize * sequenceLength;
        float* l_qkv = acts.qkv + l * batchSize * sequenceLength * 3*dimensions;
        float* l_atty = acts.atty + l * batchSize * sequenceLength * dimensions;
        float* l_preatt = acts.preatt + l * batchSize * numHeads * sequenceLength * sequenceLength;
        float* l_att = acts.att + l * batchSize * numHeads * sequenceLength * sequenceLength;
        float* l_attproj = acts.attproj + l * batchSize * sequenceLength * dimensions;
        float* l_residual2 = acts.residual2 + l * batchSize * sequenceLength * dimensions;
        float* l_ln2 = acts.ln2 + l * batchSize * sequenceLength * dimensions;
        float* l_ln2_mean = acts.ln2_mean + l * batchSize * sequenceLength;
        float* l_ln2_rstd = acts.ln2_rstd + l * batchSize * sequenceLength;
        float* l_fch = acts.fch + l * batchSize * sequenceLength * 4*dimensions;
        float* l_fch_gelu = acts.fch_gelu + l * batchSize * sequenceLength * 4*dimensions;
        float* l_fcproj = acts.fcproj + l * batchSize * sequenceLength * dimensions;
        float* l_residual3 = acts.residual3 + l * batchSize * sequenceLength * dimensions;

        // now do the forward pass
        // matmul_forward is just a linear layer in the architecture diagram
        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd , residual, l_ln1w, l_ln1b, batchSize, sequenceLength, dimensions, mode, currentToken);
        // might need to do caching here, as this is where QKV calc? need cache for each layer
        if (mode == INFERENCE) {
       		// query calculation: ordered (*Q*, K, V) in l_qkv, l_qkvw & l_qkvb
        	cached_matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, batchSize, sequenceLength, dimensions, dimensions, mode, currentToken);
         	// need correct offset for key and value within l_qkv, l_qkvw & l_qkvb as QKV are all together
          	// key calculation: +dimensions as ordered (Q, *K*, V) in l_qkv, l_qkvw & l_qkvb
          	float* key = l_qkv + dimensions;
           	cached_matmul_forward(key, l_ln1, l_qkvw + dimensions*dimensions, l_qkvb + dimensions, batchSize, sequenceLength, dimensions, dimensions, mode, currentToken);
          	// value calculation: +dimensions*2 as ordered (Q, K, *V*) in l_qkv, l_qkvw & l_qkvb
          	float* value = l_qkv + dimensions*2;
         	cached_matmul_forward(value, l_ln1, l_qkvw + dimensions*dimensions*2, l_qkvb + dimensions*2, batchSize, sequenceLength, dimensions, dimensions, mode, currentToken);
        } else {
       		matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, batchSize, sequenceLength, dimensions, 3*dimensions, mode, currentToken);
        }
        attention_forward(l_atty, l_preatt, l_att, l_qkv, batchSize, sequenceLength, dimensions, numHeads, mode, currentToken);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, batchSize, sequenceLength, dimensions, dimensions, mode, currentToken);
        residual_forward(l_residual2, residual, l_attproj, batchSize, sequenceLength, dimensions, mode, currentToken);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, batchSize, sequenceLength, dimensions, mode, currentToken);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, batchSize, sequenceLength, dimensions, 4*dimensions, mode, currentToken);
        gelu_forward(l_fch_gelu, l_fch, batchSize, sequenceLength, 4*dimensions, mode, currentToken);
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, batchSize, sequenceLength, 4*dimensions, dimensions, mode, currentToken);
        residual_forward(l_residual3, l_residual2, l_fcproj, batchSize, sequenceLength, dimensions, mode, currentToken);
    }
    residual = (acts.residual3 + (numLayers - 1)*batchSize*sequenceLength*dimensions); // last residual is in residual3
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, batchSize, sequenceLength, dimensions, mode, currentToken);
    matmul_forward(acts.logits, acts.lnf, params.wte, NULL, batchSize, sequenceLength, dimensions, paddedVocabSize, mode, currentToken);
    softmax_forward(acts.probs, acts.logits, batchSize, sequenceLength, vocabSize, paddedVocabSize, mode, currentToken);

    // also forward the cross-entropy loss function if we have the targets
    if (targets != NULL) {
        crossentropy_forward(model->acts.losses, model->acts.probs, targets, batchSize, sequenceLength, paddedVocabSize, mode, currentToken);
        // for convenience also evaluate the mean loss
        float mean_loss = 0.0f;
        switch (mode) {
        	case TRAIN_VAL: {
         		for (int i = 0; i < batchSize*sequenceLength; i++) {
                	mean_loss += model->acts.losses[i];
                }
           		mean_loss /= batchSize*sequenceLength;
         		break;
         	}
          	case INFERENCE: {
           		for (int sequence = 0; sequence < batchSize; sequence++) {
               		const int index = sequence*sequenceLength + currentToken;
                  	mean_loss += model->acts.losses[index];
                }
             	mean_loss /= batchSize;
           		break;
           	}
        }

        model->mean_loss = mean_loss;
    } else {
        // if we don't have targets, we don't have a loss
        model->mean_loss = -1.0f;
    }
}

void gpt2_zero_grad(GPT2 *model) {
    if(model->grads_memory != NULL) { memset(model->grads_memory, 0, model->num_parameters * sizeof(float)); }
    if(model->grads_acts_memory != NULL) { memset(model->grads_acts_memory, 0, model->num_activations * sizeof(float)); }
}

void gpt2_backward(GPT2 *model) {

    // double check we forwarded previously, with targets
    if (model->mean_loss == -1.0f) {
        printf("Error: must forward with targets before backward\n");
        exit(1);
    }

    // lazily allocate the memory for gradients of the weights and activations, if needed
    if (model->grads_memory == NULL) {
        model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes);
        model->grads_acts_memory = malloc_and_point_activations(model, &model->grads_acts, model->act_sizes);
        gpt2_zero_grad(model);
    }

    // convenience shortcuts (and size_t to help prevent int overflow)
    size_t B = model->batch_size;
    size_t T = model->seq_len;
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;

    // backward pass: go in the reverse order of the forward pass, and call backward() functions
    ParameterTensors params = model->params; // for brevity
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    ActivationTensors grads_acts = model->grads_acts;

    // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    // technically this is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in the batch
    float dloss_mean = 1.0f / (B*T);
    for (int i = 0; i < B*T; i++) { grads_acts.losses[i] = dloss_mean; }

    crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, model->targets, B, T, V, Vp);
    matmul_backward(grads_acts.lnf, grads.wte, NULL, grads_acts.logits, acts.lnf, params.wte, B, T, C, Vp);
    float* residual = acts.residual3 + (L-1) * B * T * C; // last layer's residual
    float* dresidual = grads_acts.residual3 + (L-1) * B * T * C; // write to last layer's residual
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);

    for (int l = L-1; l >= 0; l--) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;
        dresidual = l == 0 ? grads_acts.encoded : grads_acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        // get the pointers of the gradients of the weights for this layer
        float* dl_ln1w = grads.ln1w + l * C;
        float* dl_ln1b = grads.ln1b + l * C;
        float* dl_qkvw = grads.qkvw + l * 3*C * C;
        float* dl_qkvb = grads.qkvb + l * 3*C;
        float* dl_attprojw = grads.attprojw + l * C * C;
        float* dl_attprojb = grads.attprojb + l * C;
        float* dl_ln2w = grads.ln2w + l * C;
        float* dl_ln2b = grads.ln2b + l * C;
        float* dl_fcw = grads.fcw + l * 4*C * C;
        float* dl_fcb = grads.fcb + l * 4*C;
        float* dl_fcprojw = grads.fcprojw + l * C * 4*C;
        float* dl_fcprojb = grads.fcprojb + l * C;
        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkv = acts.qkv + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        // get the pointers of the gradients of the activations for this layer
        float* dl_ln1 = grads_acts.ln1 + l * B * T * C;
        float* dl_qkv = grads_acts.qkv + l * B * T * 3*C;
        float* dl_atty = grads_acts.atty + l * B * T * C;
        float* dl_preatt = grads_acts.preatt + l * B * NH * T * T;
        float* dl_att = grads_acts.att + l * B * NH * T * T;
        float* dl_attproj = grads_acts.attproj + l * B * T * C;
        float* dl_residual2 = grads_acts.residual2 + l * B * T * C;
        float* dl_ln2 = grads_acts.ln2 + l * B * T * C;
        float* dl_fch = grads_acts.fch + l * B * T * 4*C;
        float* dl_fch_gelu = grads_acts.fch_gelu + l * B * T * 4*C;
        float* dl_fcproj = grads_acts.fcproj + l * B * T * C;
        float* dl_residual3 = grads_acts.residual3 + l * B * T * C;

        // backprop this layer
        residual_backward(dl_residual2, dl_fcproj, dl_residual3, B*T*C);
        matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4*C, C);
        gelu_backward(dl_fch, l_fch, dl_fch_gelu, B*T*4*C);
        matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4*C);
        layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
        residual_backward(dresidual, dl_attproj, dl_residual2, B*T*C);
        matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C);
        attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
        matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3*C);
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
    }
    encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, model->inputs, B, T, C);
}

void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t) {
    // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    // lazily allocate the memory for m_memory and v_memory
    if (model->m_memory == NULL) {
        model->m_memory = (float*)calloc(model->num_parameters, sizeof(float));
        model->v_memory = (float*)calloc(model->num_parameters, sizeof(float));
    }

    for (size_t i = 0; i < model->num_parameters; i++) {
        float param = model->params_memory[i];
        float grad = model->grads_memory[i];

        // update the first moment (momentum)
        float m = beta1 * model->m_memory[i] + (1.0f - beta1) * grad;
        // update the second moment (RMSprop)
        float v = beta2 * model->v_memory[i] + (1.0f - beta2) * grad * grad;
        // bias-correct both moments
        float m_hat = m / (1.0f - powf(beta1, t));
        float v_hat = v / (1.0f - powf(beta2, t));

        // update
        model->m_memory[i] = m;
        model->v_memory[i] = v;
        model->params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }
}

void gpt2_free(GPT2 *model) {
    free(model->params_memory);
    free(model->grads_memory);
    free(model->m_memory);
    free(model->v_memory);
    free(model->acts_memory);
    free(model->grads_acts_memory);
    free(model->inputs);
    free(model->targets);
}

#ifndef TESTING
// if we are TESTING (see test_gpt2.c), we'll skip the int main below
// ----------------------------------------------------------------------------
// sampler

unsigned int random_u32(uint64_t *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(uint64_t *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

// Open file before this
void write_times(FILE* file, int numTokensGenerated, double totalTimeSeconds, double timeToFirstTokenSeconds) {
	fprintf(file, "%d,%lf,%lf\n", numTokensGenerated, totalTimeSeconds, timeToFirstTokenSeconds);
}

void inference(GPT2* model,
	Tokenizer* tokenizer,
	int* prompt,
	const int batchSize,
	const int sequenceLength,
	uint64_t* rng_state,
	FILE* timesFile,
    FILE* logitsFile
) {
	struct timespec start, end, ttft;
	// now sample from the model autoregressively
	printf("Generating:\n---\n");
	clock_gettime(CLOCK_MONOTONIC, &start);
	for (int currentToken = 0; currentToken < sequenceLength - 1; currentToken++) {
		// this is where the key-value caching will come in
		gpt2_forward(model, prompt, NULL, batchSize, sequenceLength, INFERENCE, currentToken);
		// furthermore, below we're only using b=0 (i.e. the first row) of all B rows
		// we're in principle running B "inference streams" in parallel here
		// but only using position 0
		// get the Vp-dimensional vector probs[0, t, :]
		float* probs = model->acts.probs + currentToken*model->config.padded_vocab_size;
		float coin = random_f32(rng_state);
		// note we're only sampling from the first V elements, ignoring padding
		// (the probabilities in the padded region should be zero anyway)
		int next_token = sample_mult(probs, model->config.vocab_size, coin);
        if (currentToken == 0) {
			clock_gettime(CLOCK_MONOTONIC, &ttft);
		}
		prompt[currentToken+1] = next_token;
#define PRINT
#ifdef PRINT
		// print the generated token, either using the Tokenizer or a fallback
		if (tokenizer->init_ok) {
		    const char* token_str = tokenizer_decode(tokenizer, next_token);
		    safe_printf(token_str);
		} else {
		    // fall back to printing the token id
		    printf("%d ", next_token);
		}
		fflush(stdout);
#endif
	}
	clock_gettime(CLOCK_MONOTONIC, &end);
	double timeTakenSeconds = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
	double timeToFirstTokenMillis = 1e3 * ((ttft.tv_sec - start.tv_sec) + (ttft.tv_nsec - start.tv_nsec) / 1e9);
	if (timesFile != NULL) write_times(timesFile, sequenceLength, timeTakenSeconds, timeToFirstTokenMillis);
	if (logitsFile != NULL) {
        const size_t V = model->config.vocab_size;
        const size_t Vp = model->config.padded_vocab_size;
        for (int sequence = 0; sequence < batchSize; ++sequence) {
            for (int token = 0; token < sequenceLength - 1; ++token) {
                const float* logits_bt = model->acts.logits + sequence*(size_t)sequenceLength*Vp + token*Vp;
                const size_t wrote = fwrite(logits_bt, sizeof(float), V, logitsFile);
                if (wrote != V) {
                    fprintf(stderr, "Error: failed to write logits row\n");
                    exit(1);
                }
            }
        }
    }
#define DEBUG
#ifdef DEBUG
	printf("Time to first token: %lf ms\n", timeToFirstTokenMillis);
	printf("\nTime to generate %i tokens: %lf s -> %lf tokens/s\n", sequenceLength, timeTakenSeconds, (double)sequenceLength/timeTakenSeconds);
#endif
	printf("\n---\n");
}

void error_usage() {
    fprintf(stderr, "Usage:   ./train_gpt2 [options]\n");
    fprintf(stderr, "Options:\n");
    // file system input / output
    fprintf(stderr, "  -i <string> train data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_train.bin)\n");
    fprintf(stderr, "  -j <string> val data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_val.bin)\n");
    fprintf(stderr, "  -e <string> input .bin filename or descriptor, see code comments as docs. (default = gpt2_124M_bf16.bin)\n");
    fprintf(stderr, "  -o <string> output log dir (default = NULL, no logging)\n");
    fprintf(stderr, "  -lg <int>   log gpu info every x steps (default = -1; disabled)\n");
    fprintf(stderr, "  -n <int>    write optimization checkpoints every how many steps? (default 0, don't)\n");
    fprintf(stderr, "  -nk <int>   max number of checkpoints to keep in the directory, removing old ones (0 = disable, default)\n");
    fprintf(stderr, "  -nm <int>   every how many step checkpoints are considered major? major checkpoints never get deleted.\n");
    fprintf(stderr, "  -y <int>    resume optimization found inside output log dir? (0=restart/overwrite, 1=resume/append)\n");
    // token layout for each step of the optimization
    fprintf(stderr, "  -b <int>    (per-GPU, micro) batch size B (default = 4)\n");
    fprintf(stderr, "  -t <int>    sequence length T (default = 1024)\n");
    fprintf(stderr, "  -d <int>    total desired batch size (default = B * T * num_processes, i.e. no grad accumulation\n");
    // workload (number of steps)
    fprintf(stderr, "  -x <int>    max_steps of optimization to run (-1 (default) = disable, run 1 epoch)\n");
    // optimization
    fprintf(stderr, "  -k <string> learning rate scheduler (default = cosine)\n");
    fprintf(stderr, "  -l <float>  learning rate (default = 3e-4f)\n");
    fprintf(stderr, "  -u <int>    learning rate warmup iterations (default = 0, no warmup)\n");
    fprintf(stderr, "  -q <float>  learning rate decay: final fraction, at end of training (default = 1.0 (no decay))\n");
    fprintf(stderr, "  -c <float>  weight decay (default = 0.0f)\n");
    fprintf(stderr, "  -sl <float> outlier stability: skip update if loss goes above this in zscore (0.0f=off)\n");
    fprintf(stderr, "  -sg <float> outlier stability: skip update if grad_norm goes above this in zscore (0.0f=off)\n");
    // evaluation
    fprintf(stderr, "  -v <int>    val_loss_every, how often we evaluate val loss (default = 20)\n");
    fprintf(stderr, "  -m <int>    val_max_steps, up to how many val batches to estimate val loss? (default = 20)\n");
    fprintf(stderr, "  -s <int>    sample_every, how often we inference the model (default = 20)\n");
    fprintf(stderr, "  -g <int>    genT, how many steps of inference we do (default = 64)\n");
    fprintf(stderr, "  -h <int>    hellaswag eval run? (default = 0)\n");
    // debugging
    fprintf(stderr, "  -a <int>    overfit a single batch? 0/1. useful for debugging\n");
    // numerics
    fprintf(stderr, "  -f <int>    enable_tf32 override (default: 1, set to 0 to disable tf32)\n");
    fprintf(stderr, "  -w <int>    keep f32 copy of weights for the optimizer? (default: 1)\n");
    fprintf(stderr, "  -ge <int>   gelu fusion: 0=none, 1=forward, 2=forward+backward (default: 2 for >=SM90, 0 for older GPUs)\n");
    // memory management
    fprintf(stderr, "  -z <int>    zero_stage, Zero Optimization Stage, 0,1,2,3 (default = 0)\n");
    fprintf(stderr, "  -r <int>    recompute: less memory but less speed. (default = 1), 0|1|2 = none,gelu,gelu+ln\n");
    // multi-node settings
    fprintf(stderr, "  -pn <int>    num_processes (default = 1)\n");
    fprintf(stderr, "  -pr <int>    process_rank (default = 0)\n");
    fprintf(stderr, "  -pg <int>    gpus_per_node (default = 8)\n");
    fprintf(stderr, "  -pm <string> nccl_init_method: tcp,fs,mpi (default = mpi)\n");
    fprintf(stderr, "  -ps <string> server_ip - used only when nccl_init_method is tcp (default = -1)\n");
    fprintf(stderr, "  -pp <string> fs_path - used only when nccl_init_method is fs (default = /tmp)\n");
    exit(EXIT_FAILURE);
}

// Forwards both the model and the loss and is used for validation splits and evals.
// In particular it populates cpu_losses with loss at each token.
// Some of the evals (e.g. HellaSwag) require the per-token losses, which are produced here.
float gpt2_validate(GPT2* model, const int* inputs, int* targets, const size_t batchSize, const size_t sequenceLength) {
    assert(targets != NULL);
    // forward the model itself
    gpt2_forward(model, inputs, targets, batchSize, sequenceLength, TRAIN_VAL, 0);
    // convenience shortcuts, size_t instead of int so that pointer arithmetics don't overflow
    const size_t vocabSize = model->config.vocab_size;
    // note: we don't need to generate dlogits here
    tokenCheck(targets, batchSize*sequenceLength, vocabSize); // validate the targets
    return model->mean_loss;
}
// ----------------------------------------------------------------------------
// main training loop
int main(int argc, char* argv[]) {
	// read in the (optional) command line arguments
    const char* lr_scheduler_type = "cosine";
    const char* output_log_dir = NULL;
    int checkpoint_every = 0; // write checkpoints every how many steps?
    int checkpoints_keep = 0; // how long checkpoint history do we keep? (in units of checkpoints)
    int major_checkpoint_every = 0; // major checkpoints never get deleted when maintaining history
    int resume = 0; // resume the optimization, if one is found inside output_log_dir?
    int batchSize = 1; // batch size 4 (i.e. 4 independent token sequences will be trained on)
    int maxSequenceLength = 1024; // sequence length max
    int total_batch_size = -1; // will be calculated down below later, if not provided
    float learning_rate = 0.f;
    int log_gpu_every = -1;
    int warmup_iterations = 0;
    float final_learning_rate_frac = 1.0f; // final fraction of learning rate, at end of training
    float weight_decay = 0.0f;
    float skip_update_lossz = 0.0f; // skip update if loss goes above this in zscore
    float skip_update_gradz = 0.0f; // skip update if grad_norm goes above this in zscore
    int val_loss_every = 20; // every how many steps do we eval validation loss?
    int val_max_steps = 20; // how many batches max do we eval for validation loss?
    int sample_every = 20; // every how many steps to do inference?
    int numInferenceSteps = 1024; // number of steps of inference we will do / tokens will be generated
    int overfit_single_batch = 0; // useful for debugging, 1 = only load a single data batch once
    int max_steps = 1;
    int override_enable_tf32 = 0;
    int use_master_weights = 1;
    int gelu_fusion = 2; // 0 = none, 1 = forward, 2 = forward+backward (-1 => per-GPU default)
    int recompute = 1; // recompute during backward setting, 0 = none, 1 = recompute gelu
    int zero_stage = 0; // Zero Optimization Stage for Multi-GPU training
    bool hellaswag_eval = 0;
    // multi-node settings
    int num_processes = 1;  // this should be set by the slurm environment
    int process_rank = 0;  // this should be set by the slurm environment
    int gpus_per_node = 8;  // this should be set by the slurm environment
    char nccl_init_method[256] = "mpi";  // "tcp" or "fs" or "mpi"
    char server_ip[256] = "";  // used if init_method set to "tcp" -> set to your server ip address
    char fs_path[256] = "";  // used if init_method set to "fs" -> set to a shared filesystem path
    const char* tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin";
    const char* tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
    const char* tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    const char* tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    const char* train_tokens = access(tiny_shakespeare_train, F_OK) != -1 ? tiny_shakespeare_train : tiny_stories_train;
    const char* val_tokens = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;
    const char* saved_model_file = "gpt2_124M.bin";
    const char* CPU_TIMES = "cpu_times.csv";
    const char* CPU_LOGITS = "cpu_logits.log";
    const char* GPU_LOGITS = "gpu_logits.log";
    for (int i = 1; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (!(strlen(argv[i]) == 2 || strlen(argv[i]) == 3)) { error_usage(); } // must be -x[y] (one dash, one or two letters)
        // read in the args
        if (argv[i][1] == 'i') { tiny_shakespeare_train = argv[i+1]; }
        else if (argv[i][1] == 'j') { tiny_shakespeare_val = argv[i+1]; }
        else if (argv[i][1] == 'e') { saved_model_file = argv[i+1]; }
        else if (argv[i][1] == 'o') { output_log_dir = argv[i+1]; }
        else if (argv[i][1] == 'n' && argv[i][2] == '\0') { checkpoint_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'y') { resume = atoi(argv[i+1]); }
        else if (argv[i][1] == 'b') { batchSize = atoi(argv[i+1]); } // Per-GPU (micro) batch size
        else if (argv[i][1] == 't') { maxSequenceLength = atoi(argv[i+1]); }
        else if (argv[i][1] == 'd') { total_batch_size = atoi(argv[i+1]); }
        else if (argv[i][1] == 'l' && argv[i][2] == '\0') { learning_rate = atof(argv[i+1]); }
        else if (argv[i][1] == 'l' && argv[i][2] == 'g') { log_gpu_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'u') { warmup_iterations = atoi(argv[i+1]); }
        else if (argv[i][1] == 'q') { final_learning_rate_frac = atof(argv[i+1]); }
        else if (argv[i][1] == 'c') { weight_decay = atof(argv[i+1]); }
        else if (argv[i][1] == 'x') { max_steps = atoi(argv[i+1]); }
        else if (argv[i][1] == 'v') { val_loss_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'm') { val_max_steps = atoi(argv[i+1]); }
        else if (argv[i][1] == 's' && argv[i][2] == '\0') { sample_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'g' && argv[i][2] == 'e') { gelu_fusion = atoi(argv[i+1]); }
        else if (argv[i][1] == 'g') { numInferenceSteps = atoi(argv[i+1]); }
        else if (argv[i][1] == 'a') { overfit_single_batch = atoi(argv[i+1]); }
        else if (argv[i][1] == 'f') { override_enable_tf32 = atoi(argv[i+1]); }
        else if (argv[i][1] == 'w') { use_master_weights = atoi(argv[i+1]); }
        else if (argv[i][1] == 'z') { zero_stage = atoi(argv[i+1]); }
        else if (argv[i][1] == 'r') { recompute = atoi(argv[i+1]); }
        else if (argv[i][1] == 'h') { hellaswag_eval = atoi(argv[i+1]); }
        else if (argv[i][1] == 'k') { lr_scheduler_type = argv[i+1]; }
        else if (argv[i][1] == 'p' && argv[i][2] == 'i') { strcpy(nccl_init_method, argv[i+1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 'f') { strcpy(fs_path, argv[i+1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 's') { strcpy(server_ip, argv[i+1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 'n') { num_processes = atoi(argv[i+1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 'r') { process_rank = atoi(argv[i+1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 'g') { gpus_per_node = atoi(argv[i+1]); }
        else if (argv[i][1] == 's' && argv[i][2] == 'l') { skip_update_lossz = atof(argv[i+1]); }
        else if (argv[i][1] == 's' && argv[i][2] == 'g') { skip_update_gradz = atof(argv[i+1]); }
        else if (argv[i][1] == 'n' && argv[i][2] == 'k') { checkpoints_keep = atoi(argv[i+1]); }
        else if (argv[i][1] == 'n' && argv[i][2] == 'm') { major_checkpoint_every = atoi(argv[i+1]); }
        else { error_usage(); }
    }

    FILE* timeFile = fopenCheck(CPU_TIMES, "a");
    FILE* logitsFile = fopenCheck(CPU_LOGITS, "wb");

    Logger logger;
    logger_init(&logger, output_log_dir, 0, resume);

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, saved_model_file);

    // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    int sequenceLength = numInferenceSteps; // e.g. sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, train_tokens, batchSize, sequenceLength, 0, 1, 1);
    dataloader_init(&val_loader, val_tokens, batchSize, sequenceLength, 0, 1, 0);
    printf("train dataset num_batches: %zu\n", train_loader.num_tokens / (batchSize*sequenceLength));
    printf("val dataset num_batches: %zu\n", val_loader.num_tokens / (batchSize*sequenceLength));
    int val_num_batches = 5;

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");
    // some memory for generating samples from the model
    uint64_t rng_state = 1337;
    int* prompt = (int*)mallocCheck(batchSize * sequenceLength * sizeof(int));
    // train
    struct timespec start, end;
    for (int step = 0; step < max_steps; step++) {
        // once in a while estimate the validation loss
        if (false) {
            float val_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int i = 0; i < val_num_batches; i++) {
                dataloader_next_batch(&val_loader);
                gpt2_forward(&model, val_loader.inputs, val_loader.targets, batchSize, sequenceLength, TRAIN_VAL, 0);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches;
            printf("val loss %f\n", val_loss);
        }

        // once in a while do model inference to print generated text
        // fill up prompt with the GPT2_EOT, which kicks off the generation
		for (int i = 0; i < batchSize * sequenceLength; ++i) {
			prompt[i] = tokenizer.eot_token;
		}
		memset(model.acts_memory, 0, model.num_activations * sizeof(float));
        inference(&model, &tokenizer, prompt, batchSize, sequenceLength, &rng_state, timeFile, logitsFile);
        // do a training step
        /*
        clock_gettime(CLOCK_MONOTONIC, &start);
        dataloader_next_batch(&train_loader);
        gpt2_forward(&model, train_loader.inputs, train_loader.targets, batchSize, sequenceLength, TRAIN_VAL, 0);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("step %d: train loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
        */
    }
    fcloseCheck(logitsFile);
    // build an EvalLoader for HellaSwag
    EvalLoader eval_loader;
    const char* hellaswag_path = "dev/data/hellaswag/hellaswag_val.bin";
    const bool hellaswag_available = access(hellaswag_path, F_OK) == 0;
    const bool run_hellaswag = hellaswag_eval && hellaswag_available;
    if (run_hellaswag) {
        FILE* cpuLogits = fopenCheck(CPU_LOGITS, "rb");
        FILE* gpuLogits = fopenCheck(GPU_LOGITS, "rb");
    	// no multi-GPU so can set index as 0 and total as 1
        evalloader_init(&eval_loader, hellaswag_path, 4, maxSequenceLength, 0, 1);
        printf("| run hellaswag         | %-50s |\n", run_hellaswag ? "yes" : "no");
        puts("+-----------------------+----------------------------------------------------+\n");
	    float eval_acc_norm = 0.0f;
	    evalloader_reset(&eval_loader);
		evalloader_next_batch(&eval_loader);
		int* correctTokens = (int*)malloc(sizeof(int)*eval_loader.T*eval_loader.B*4);
        const int numberOfCompletions = evalloader_get_answers(correctTokens, &eval_loader, 1);
        const int compareSequenceLength = sequenceLength - 1;
        for (int sequence = 0; sequence < numberOfCompletions; ++sequence) {
            int* currentTarget = correctTokens + sequence*eval_loader.T;
            Pair results = compare_implementations(cpuLogits, gpuLogits, currentTarget, 1, compareSequenceLength, model.config.vocab_size, model.config.padded_vocab_size);
            printf("CPU: %lf\n", results.left);
            printf("GPU: %lf\n", results.right);
        }
        /*
        printf("Num Batches: %i\n", eval_loader.num_batches);
	    for (int i = 0; i < eval_loader.num_batches; i++) { // each batch has the 4 possible completion
#ifdef DEBUG
	        printf("evaluating HellaSwag: %d/%d\r", i, eval_loader.num_batches);
			fflush(stdout);
#endif
	        evalloader_next_batch(&eval_loader);
			gpt2_validate(&model, eval_loader.inputs, eval_loader.targets, batchSize, maxSequenceLength);
	        int correct = evalloader_stat_losses(&eval_loader, model.acts.losses);
	        eval_acc_norm += (float)correct;
	    }
		*/
		fcloseCheck(cpuLogits);
        fcloseCheck(gpuLogits);
        free(correctTokens);
	    //printf("HellaSwag: %d/%d = %f\n", (int)eval_acc_norm, eval_loader.num_examples, eval_acc_norm / eval_loader.num_examples);
    }
    // free
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    if (run_hellaswag) {
    	evalloader_free(&eval_loader);
    }
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    free(prompt);
    fcloseCheck(timeFile);
    return EXIT_SUCCESS;
}
#endif
