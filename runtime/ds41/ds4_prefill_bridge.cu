#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>
#include "/tmp/ds4-ref/cuda/mmq/ds4_mmq.h"

__device__ __forceinline__ float ds41_bf16_to_float(uint16_t v) {
    union { uint32_t u; float f; } x;
    x.u = ((uint32_t)v) << 16;
    return x.f;
}

__global__ void ds41_prepare_input_ids(
    const uint16_t* x_bf16, float* x_f32, int64_t nx,
    const int32_t* global_ids, const int32_t* expert_map, int32_t* local_ids,
    int64_t nids, int global_experts) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (i<nx) x_f32[i]=ds41_bf16_to_float(x_bf16[i]);
    if (i<nids) {
        int32_t g=global_ids[i];
        local_ids[i]=(g>=0 && g<global_experts) ? expert_map[g] : -1;
    }
}

__global__ void ds41_swiglu_weighted(
    const float* gate,const float* up,const float* weights,const int32_t* ids,
    float* mid,int64_t total,int mid_dim,float clamp) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=total) return;
    int64_t a=i/mid_dim;
    if(ids[a]<0){ mid[i]=0.0f; return; }
    float g=gate[i], u=up[i];
    if(!isfinite(g)) g=0.0f; if(!isfinite(u)) u=0.0f;
    if(clamp>1.0e-6f){ g=fminf(g,clamp); u=fminf(fmaxf(u,-clamp),clamp); }
    mid[i]=(g/(1.0f+expf(-g)))*u*weights[a];
}

__global__ void ds41_sum6(const float* routes,float* out,int tokens,int hidden){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
    int64_t n=(int64_t)tokens*hidden; if(i>=n) return;
    int t=(int)(i/hidden), h=(int)(i-(int64_t)t*hidden);
    float s=0.0f;
    #pragma unroll
    for(int k=0;k<6;k++) s += routes[((int64_t)t*6+k)*hidden+h];
    out[i]=s;
}

extern "C" int ds41_ds4_moe_full(
    const void* x_bf16,
    const int32_t* global_ids,
    const float* router_weights,
    const int32_t* expert_map,
    const void* w_gate,
    const void* w_up,
    const void* w_down,
    int tokens,
    int hidden,
    int mid_dim,
    int global_experts,
    int local_experts,
    float clamp,
    int32_t* local_ids,
    float* x_f32,
    float* gate,
    float* up,
    float* mid,
    float* down_routes,
    float* out_f32,
    hipStream_t stream) {
    if(!x_bf16||!global_ids||!router_weights||!expert_map||!w_gate||!w_up||!w_down||
       !local_ids||!x_f32||!gate||!up||!mid||!down_routes||!out_f32||tokens<=0) return -100;
    int64_t nx=(int64_t)tokens*hidden, nids=(int64_t)tokens*6;
    int64_t prep_n=nx>nids?nx:nids;
    ds41_prepare_input_ids<<<(prep_n+255)/256,256,0,stream>>>((const uint16_t*)x_bf16,x_f32,nx,global_ids,expert_map,local_ids,nids,global_experts);
    if(hipGetLastError()!=hipSuccess) return -101;
    size_t route_mid=(size_t)tokens*6*(size_t)mid_dim;
    size_t route_out=(size_t)tokens*6*(size_t)hidden;
    hipMemsetAsync(gate,0,route_mid*sizeof(float),stream);
    hipMemsetAsync(up,0,route_mid*sizeof(float),stream);
    hipMemsetAsync(down_routes,0,route_out*sizeof(float),stream);
    int rc=ds4_mmq_iq2_xxs_moe_pair(w_gate,w_up,x_f32,local_ids,gate,up,mid_dim,hidden,tokens,local_experts,6,(cudaStream_t)stream);
    if(rc) return 1000+rc;
    ds41_swiglu_weighted<<<(route_mid+255)/256,256,0,stream>>>(gate,up,router_weights,local_ids,mid,(int64_t)route_mid,mid_dim,clamp);
    if(hipGetLastError()!=hipSuccess) return -102;
    // Each routed row has its own mid activation. Reinterpret as tokens*6 rows with top_k=1.
    rc=ds4_mmq_q2_K_moe(w_down,mid,local_ids,down_routes,hidden,mid_dim,tokens*6,local_experts,1,(cudaStream_t)stream);
    if(rc) return 2000+rc;
    int64_t no=(int64_t)tokens*hidden;
    ds41_sum6<<<(no+255)/256,256,0,stream>>>(down_routes,out_f32,tokens,hidden);
    if(hipGetLastError()!=hipSuccess) return -103;
    return 0;
}
