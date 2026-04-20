// OpenMP target offloading port of flame-kokkos (fractal flame IFS renderer)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <sys/time.h>
#include <omp.h>

#define WIDTH   800
#define HEIGHT  600
#define NUM_FUNCTIONS       20
#define VARIATIONS_PER_FUNCTION 5
#define NUM_THREADS         (1 << 14)
#define NUM_POINTS_PER_THREAD 64
#define NUM_ITERATIONS      15
#define NUM_RANDOMS         (1 << 22)
#define NUM_PERMUTATIONS    2048

#ifndef M_PI
#define M_PI 3.1415926535f
#endif

struct VariationParameter { int idx; float factor; };

struct ConstMemParams {
  VariationParameter variation_parameters[NUM_FUNCTIONS][VARIATIONS_PER_FUNCTION];
  float function_colors[NUM_FUNCTIONS];
  float pre_transform_params[NUM_FUNCTIONS][6];
  float post_transform_params[NUM_FUNCTIONS][6];
  int   frame_counter;
  int   enable_sierpinski;
  int   thread_function_mapping[NUM_FUNCTIONS];
};

struct PermSortElement {
  int value; unsigned short idx;
  bool operator<(const PermSortElement& o) const { return value < o.value; }
};

// Host utilities
unsigned mersenne_twister(unsigned* s)
{
  const int N=624, M=397;
  static int idx=N+1;
  const unsigned A[2]={0,0x9908b0df}, HI=0x80000000, LO=0x7fffffff;
  if(idx>=N){
    for(int k=0;k<N-M;++k){ unsigned h=(s[k]&HI)|(s[k+1]&LO); s[k]=s[k+M]^(h>>1)^A[h&1]; }
    for(int k=N-M;k<N-1;++k){ unsigned h=(s[k]&HI)|(s[k+1]&LO); s[k]=s[k+(M-N)]^(h>>1)^A[h&1]; }
    unsigned h=(s[N-1]&HI)|(s[0]&LO); s[N-1]=s[M-1]^(h>>1)^A[h&1]; idx=0;
  }
  unsigned e=s[idx++]; e^=e>>11; e^=(e<<7)&0x9d2c5680; e^=(e<<15)&0xefc60000; e^=e>>18;
  return e;
}

float radical_inverse(unsigned n, unsigned base)
{
  float res=0, div=1.f/base;
  while(n){ float d=n%base; res+=d*div; n=(n-d)/base; div/=base; }
  return res;
}

#pragma omp declare target

static inline void affine_transform(float& px, float& py, const float* p)
{
  float tx = p[0]*px + p[1]*py + p[2];
  float ty = p[3]*px + p[4]*py + p[5];
  px=tx; py=ty;
}

static inline void sierpinski(float& px, float& py, int i)
{
  px*=.5f; py*=.5f;
  switch(i%3){case 1:px+=.5f;break;case 2:py+=.5f;break;}
}

static inline int get_function_idx(int thread_idx, const ConstMemParams* params)
{
  int start=0, end=NUM_FUNCTIONS, func=0;
  thread_idx &= ~31;
  for(int i=0;i<6;i++){
    func=(start+end)/2;
    if(params->thread_function_mapping[func]<=thread_idx) start=func+1;
    else end=func;
  }
  return func;
}

static inline void iteration_fractal_flame(float& px, float& py, float& color,
    int func_idx, int thread_idx,
    const float* random_numbers,
    const ConstMemParams* params)
{
  if(params->enable_sierpinski==1) sierpinski(px,py,thread_idx);
  else if(params->enable_sierpinski==2) sierpinski(px,py,func_idx);

  affine_transform(px,py,&params->pre_transform_params[func_idx][0]);

  float rad2=px*px+py*py, rad=sqrtf(rad2);
  float theta=atan2f(px,py), phi=atan2f(py,px);
  (void)phi;

  float ox=0,oy=0;
  const VariationParameter* vp=&params->variation_parameters[func_idx][0];
  for(int i=0;i<VARIATIONS_PER_FUNCTION;i++){
    if(fabsf(vp[i].factor)<0.01f) continue;
    float ex=0,ey=0;
    switch(vp[i].idx){
      case 0: ex=px;ey=py;break;
      case 1: ex=sinf(px);ey=sinf(py);break;
      case 2: ex=px/(rad2+1e-10f);ey=py/(rad2+1e-10f);break;
      case 5: ex=theta/M_PI;ey=rad-1.f;break;
      case 6: ex=rad*sinf(theta+rad);ey=rad*cosf(theta-rad);break;
      case 7: ex=rad*sinf(theta*rad);ey=-rad*cosf(theta*rad);break;
      case 9: ex=rad*(cosf(theta)+sinf(rad));ey=rad*(sinf(theta)-cosf(rad));break;
      case 13:{float sr=sqrtf(rad),om=(random_numbers[thread_idx&(NUM_RANDOMS-1)]>0.5f)?0.f:M_PI;
               ex=sr*cosf(theta/2+om);ey=sr*sinf(theta/2+om);break;}
      default: ex=px;ey=py;break;
    }
    ox+=vp[i].factor*ex; oy+=vp[i].factor*ey;
  }
  px=ox; py=oy;
  affine_transform(px,py,&params->post_transform_params[func_idx][0]);
  color=.5f*(color+params->function_colors[func_idx]);
}

#pragma omp end declare target

int main(int argc, char** argv)
{
  if(argc!=2){ printf("Usage: %s <repeat>\n",argv[0]); return 1; }
  const int repeat=atoi(argv[1]);

  ConstMemParams cmp; float function_weights[NUM_FUNCTIONS];
  memset(&cmp,0,sizeof cmp);
  for(int i=0;i<NUM_FUNCTIONS;i++){
    function_weights[i]=0.f;
    cmp.variation_parameters[i][0].factor=1.f;
    cmp.pre_transform_params[i][0]=1.f; cmp.pre_transform_params[i][4]=1.f;
    cmp.post_transform_params[i][0]=1.f; cmp.post_transform_params[i][4]=1.f;
  }
  function_weights[0]=1.f;
  cmp.pre_transform_params[0][0]=1.4f; cmp.pre_transform_params[0][1]=0.6f;
  cmp.pre_transform_params[1][2]=0.4f;
  cmp.pre_transform_params[2][0]=0.3f; cmp.pre_transform_params[2][2]=1.f;
  cmp.function_colors[0]=1.f; cmp.function_colors[2]=0.4f;
  function_weights[1]=0.5f; function_weights[2]=0.6f;
  cmp.enable_sierpinski=1;
  cmp.variation_parameters[2][0].idx=9; cmp.variation_parameters[0][0].idx=13;
  cmp.variation_parameters[1][0].idx=5; cmp.variation_parameters[1][1].idx=7;
  cmp.variation_parameters[2][0].factor=1.f; cmp.variation_parameters[0][0].factor=1.f;
  cmp.variation_parameters[1][0].factor=1.f; cmp.variation_parameters[1][1].factor=-.3f;
  cmp.pre_transform_params[2][2]=-1.f; cmp.pre_transform_params[2][5]=-1.f;

  srand(2);
  unsigned ms[624]; for(int i=0;i<624;i++) ms[i]=rand();
  for(int i=0;i<10000;i++) mersenne_twister(ms);

  float* rn_tmp=new float[NUM_RANDOMS];
  for(int i=0;i<NUM_RANDOMS;i++) rn_tmp[i]=mersenne_twister(ms)*(1.f/4294967296.f);

  PermSortElement* to_sort=new PermSortElement[NUM_THREADS];
  unsigned short* perm_data=new unsigned short[NUM_THREADS*NUM_PERMUTATIONS];
  for(int i=0;i<NUM_PERMUTATIONS;i++){
    for(int j=0;j<NUM_THREADS;j++){ to_sort[j].value=mersenne_twister(ms); to_sort[j].idx=j; }
    std::sort(to_sort,to_sort+NUM_THREADS);
    for(int j=0;j<NUM_THREADS;j++) perm_data[i*NUM_THREADS+j]=to_sort[j].idx;
  }
  delete[] to_sort;

  float* pts_x=new float[NUM_THREADS];
  float* pts_y=new float[NUM_THREADS];
  for(int i=0;i<NUM_THREADS;i++){
    pts_x[i]=(float(i)/NUM_THREADS-.5f)*2.f;
    pts_y[i]=(radical_inverse(i,2)-.5f)*2.f;
  }

  // Device arrays
  float*          d_rn    = (float*)malloc(NUM_RANDOMS * sizeof(float));
  unsigned short* d_perms = (unsigned short*)malloc((size_t)NUM_THREADS*NUM_PERMUTATIONS*sizeof(unsigned short));
  float*          d_spx   = (float*)malloc(NUM_THREADS * sizeof(float));
  float*          d_spy   = (float*)malloc(NUM_THREADS * sizeof(float));
  float*          d_col   = (float*)malloc(NUM_THREADS * sizeof(float));
  float*          d_stpx  = (float*)malloc(NUM_THREADS * sizeof(float));
  float*          d_stpy  = (float*)malloc(NUM_THREADS * sizeof(float));
  float*          d_vx    = (float*)malloc((size_t)NUM_THREADS*NUM_POINTS_PER_THREAD*sizeof(float));
  float*          d_vy    = (float*)malloc((size_t)NUM_THREADS*NUM_POINTS_PER_THREAD*sizeof(float));
  float*          d_vc    = (float*)malloc((size_t)NUM_THREADS*NUM_POINTS_PER_THREAD*sizeof(float));
  ConstMemParams* d_par   = (ConstMemParams*)malloc(sizeof(ConstMemParams));

  for(int i=0;i<NUM_RANDOMS;i++) d_rn[i]=rn_tmp[i];
  for(int i=0;i<NUM_THREADS*NUM_PERMUTATIONS;i++) d_perms[i]=perm_data[i];
  for(int i=0;i<NUM_THREADS;i++){ d_stpx[i]=pts_x[i]; d_stpy[i]=pts_y[i]; }
  *d_par = cmp;

  const int perm_total = NUM_THREADS*NUM_PERMUTATIONS;
  const int vert_total = NUM_THREADS*NUM_POINTS_PER_THREAD;

  #pragma omp target enter data \
    map(to: d_rn[0:NUM_RANDOMS], d_perms[0:perm_total], d_stpx[0:NUM_THREADS], d_stpy[0:NUM_THREADS], d_par[0:1]) \
    map(alloc: d_spx[0:NUM_THREADS], d_spy[0:NUM_THREADS], d_col[0:NUM_THREADS], \
               d_vx[0:vert_total], d_vy[0:vert_total], d_vc[0:vert_total])

  delete[] rn_tmp; delete[] perm_data; delete[] pts_x; delete[] pts_y;

  struct timeval tv,tv2;
  gettimeofday(&tv,NULL);

  int perm_pos=0;
  for(int n=0;n<repeat;n++){
    // Update thread_function_mapping
    float sum=0; for(int i=0;i<NUM_FUNCTIONS;i++) sum+=function_weights[i];
    int ntsum=0;
    for(int i=0;i<NUM_FUNCTIONS;i++){
      int nt=(int)((function_weights[i]/sum)*NUM_THREADS);
      cmp.thread_function_mapping[i]=nt; ntsum+=nt;
    }
    cmp.thread_function_mapping[0]+=NUM_THREADS-ntsum;
    for(int i=1;i<NUM_FUNCTIONS;i++) cmp.thread_function_mapping[i]+=cmp.thread_function_mapping[i-1];
    *d_par = cmp;
    #pragma omp target update to(d_par[0:1])

    // KernelInitialize
    const int pp0 = perm_pos;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for(int idx=0;idx<NUM_THREADS;idx++){
      int func=get_function_idx(idx,d_par);
      int perm_idx=d_perms[NUM_THREADS*pp0+idx];
      float px=d_stpx[perm_idx], py=d_stpy[perm_idx];
      float color=0.5f;
      iteration_fractal_flame(px,py,color,func,idx,d_rn,d_par);
      d_spx[idx]=px; d_spy[idx]=py; d_col[idx]=color;
    }
    perm_pos++;

    // KernelIterate
    for(int i=0;i<NUM_ITERATIONS;i++){
      const int pp = perm_pos;
      #pragma omp target teams distribute parallel for thread_limit(256)
      for(int idx=0;idx<NUM_THREADS;idx++){
        int func=get_function_idx(idx,d_par);
        int perm_idx=d_perms[NUM_THREADS*pp+idx];
        float px=d_spx[perm_idx], py=d_spy[perm_idx];
        float color=d_col[perm_idx];
        iteration_fractal_flame(px,py,color,func,idx,d_rn,d_par);
        d_spx[idx]=px; d_spy[idx]=py; d_col[idx]=color;
      }
      perm_pos=(perm_pos+1)%NUM_PERMUTATIONS;
    }

    // KernelGenerate
    const int pp1 = perm_pos;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for(int idx=0;idx<NUM_THREADS;idx++){
      int func=get_function_idx(idx,d_par);
      float px=d_spx[d_perms[NUM_THREADS*pp1+idx]];
      float py=d_spy[d_perms[NUM_THREADS*pp1+idx]];
      float color=d_col[d_perms[NUM_THREADS*pp1+idx]];
      for(int i=0;i<NUM_POINTS_PER_THREAD;i++){
        int ppi=(pp1+i)%NUM_PERMUTATIONS;
        (void)d_perms[NUM_THREADS*ppi+idx];
        iteration_fractal_flame(px,py,color,func,idx,d_rn,d_par);
        d_vx[idx*NUM_POINTS_PER_THREAD+i]=px;
        d_vy[idx*NUM_POINTS_PER_THREAD+i]=py;
        d_vc[idx*NUM_POINTS_PER_THREAD+i]=color;
      }
    }
    perm_pos=(perm_pos+NUM_POINTS_PER_THREAD)%NUM_PERMUTATIONS;
  }

  gettimeofday(&tv2,NULL);
  float frametime=(tv2.tv_sec-tv.tv_sec)*1000000.f+tv2.tv_usec-tv.tv_usec;
  printf("Total frame time is %.1f us\n",frametime);

  #pragma omp target update from(d_vx[0:vert_total])

  float vsum=0; for(int i=0;i<NUM_THREADS*NUM_POINTS_PER_THREAD;i++) vsum+=d_vx[i];
  printf("Vertex x checksum: %f\n",vsum/(NUM_THREADS*NUM_POINTS_PER_THREAD));

  #pragma omp target exit data \
    map(delete: d_rn[0:NUM_RANDOMS], d_perms[0:perm_total], d_stpx[0:NUM_THREADS], d_stpy[0:NUM_THREADS], \
                d_par[0:1], d_spx[0:NUM_THREADS], d_spy[0:NUM_THREADS], d_col[0:NUM_THREADS], \
                d_vx[0:vert_total], d_vy[0:vert_total], d_vc[0:vert_total])

  free(d_rn); free(d_perms); free(d_spx); free(d_spy); free(d_col);
  free(d_stpx); free(d_stpy); free(d_vx); free(d_vy); free(d_vc); free(d_par);
  return 0;
}
