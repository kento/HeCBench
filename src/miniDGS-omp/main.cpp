// OpenMP target offload port of miniDGS-kokkos: Discontinuous Galerkin Maxwell solver.
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

static constexpr int p_N        = 6;
static constexpr int p_Nfp      = (p_N + 1) * (p_N + 2) / 2;             // 28
static constexpr int p_Np       = (p_N + 1) * (p_N + 2) * (p_N + 3) / 6; // 84
static constexpr int p_Nfields  = 6;
static constexpr int p_Nfaces   = 4;
static constexpr int BSIZE      = 16 * ((p_Np + 15) / 16);               // 96

static constexpr float rk4a[5] = {
     0.0f,
    -567301805773.0f  / 1357537059087.0f,
    -2404267990393.0f / 2016746695238.0f,
    -3550918686646.0f / 2091501179385.0f,
    -1275806237668.0f /  842570457699.0f
};
static constexpr float rk4b[5] = {
     1432997174477.0f /  9575080441755.0f,
     5161836677717.0f / 13612068292357.0f,
     1720146321549.0f /  2090206949498.0f,
     3134564353537.0f /  4481467310338.0f,
     2277821191437.0f / 14882151754819.0f
};

static void buildSyntheticMesh(int K,
                                std::vector<float>& vgeo,
                                std::vector<float>& DrDsDt,
                                std::vector<float>& LIFT,
                                std::vector<float>& surfinfo,
                                std::vector<float>& Q) {
  const int NfpNf = p_Nfp * p_Nfaces;
  vgeo.assign(K * 12, 0.0f);
  for (int k = 0; k < K; k++) {
    vgeo[k*12+ 0] = 1.f;
    vgeo[k*12+ 5] = 1.f;
    vgeo[k*12+10] = 1.f;
  }
  DrDsDt.assign(BSIZE * BSIZE * 4, 0.0f);
  for (int n = 0; n < p_Np; n++)
    for (int m = 0; m < p_Np; m++) {
      const int didx = 4 * (n + m * BSIZE);
      float v = (n == m) ? 1.0f / (float)p_Np : -1.0f / (float)(p_Np * p_Np);
      DrDsDt[didx+0] = v;
      DrDsDt[didx+1] = v;
      DrDsDt[didx+2] = v;
    }
  LIFT.assign(p_Np * NfpNf, 0.0f);
  for (int n = 0; n < p_Np; n++)
    for (int j = 0; j < NfpNf; j++)
      LIFT[n + j * p_Np] = (n == (j % p_Np)) ? 0.5f : 0.0f;
  surfinfo.assign(K * NfpNf * 7, 0.0f);
  for (int k = 0; k < K; k++)
    for (int n = 0; n < NfpNf; n++) {
      const int idx = 7 * (k * NfpNf) + n;
      const int nM  = n % p_Np;
      const int idM = nM + k * p_Nfields * BSIZE;
      surfinfo[idx + 0*NfpNf] = (float)idM;
      surfinfo[idx + 1*NfpNf] = -1.f;
      surfinfo[idx + 2*NfpNf] =  0.5f;
      surfinfo[idx + 3*NfpNf] =  1.f;
      surfinfo[idx + 4*NfpNf] =  1.f;
      surfinfo[idx + 5*NfpNf] =  0.f;
      surfinfo[idx + 6*NfpNf] =  0.f;
    }
  Q.assign(K * p_Nfields * BSIZE, 0.0f);
  for (int k = 0; k < K; k++)
    for (int n = 0; n < p_Np; n++) {
      float x = (float)(k * p_Np + n) / (float)(K * p_Np) * 2.f * 3.14159265f;
      Q[n + 5*BSIZE + k*p_Nfields*BSIZE] = cosf(x);
    }
}

int main(int argc, char* argv[]) {
  int K = 100;
  if (argc > 1) K = atoi(argv[1]);

  const int NfpNf      = p_Nfp * p_Nfaces;
  const int Ntotal     = K * BSIZE * p_Nfields;
  const int fluxQ_sz   = K * p_Nfields * NfpNf;
  const int DrDsDt_sz  = BSIZE * BSIZE * 4;
  const int LIFT_sz    = p_Np * NfpNf;
  const int surfinfo_sz = K * NfpNf * 7;
  const int vgeo_sz    = K * 12;

  printf("miniDGS Maxwell solver (OMP): K=%d elements, p_N=%d\n", K, p_N);

  std::vector<float> h_vgeo, h_DrDsDt, h_LIFT, h_surfinfo, h_Q;
  buildSyntheticMesh(K, h_vgeo, h_DrDsDt, h_LIFT, h_surfinfo, h_Q);

  // Allocate host/device arrays
  float* d_Q        = (float*)malloc(Ntotal      * sizeof(float));
  float* d_rhsQ     = (float*)malloc(Ntotal      * sizeof(float));
  float* d_resQ     = (float*)malloc(Ntotal      * sizeof(float));
  float* d_fluxQ    = (float*)malloc(fluxQ_sz    * sizeof(float));
  float* d_vgeo     = (float*)malloc(vgeo_sz     * sizeof(float));
  float* d_DrDsDt   = (float*)malloc(DrDsDt_sz   * sizeof(float));
  float* d_LIFT     = (float*)malloc(LIFT_sz     * sizeof(float));
  float* d_surfinfo = (float*)malloc(surfinfo_sz * sizeof(float));

  memcpy(d_Q,        h_Q.data(),        Ntotal      * sizeof(float));
  memcpy(d_vgeo,     h_vgeo.data(),     vgeo_sz     * sizeof(float));
  memcpy(d_DrDsDt,   h_DrDsDt.data(),   DrDsDt_sz   * sizeof(float));
  memcpy(d_LIFT,     h_LIFT.data(),     LIFT_sz     * sizeof(float));
  memcpy(d_surfinfo, h_surfinfo.data(), surfinfo_sz * sizeof(float));
  memset(d_rhsQ,  0, Ntotal   * sizeof(float));
  memset(d_resQ,  0, Ntotal   * sizeof(float));
  memset(d_fluxQ, 0, fluxQ_sz * sizeof(float));

  // Map all arrays to device
  #pragma omp target enter data \
    map(to: d_Q[0:Ntotal], d_vgeo[0:vgeo_sz], \
            d_DrDsDt[0:DrDsDt_sz], d_LIFT[0:LIFT_sz], \
            d_surfinfo[0:surfinfo_sz]) \
    map(to: d_rhsQ[0:Ntotal], d_resQ[0:Ntotal], d_fluxQ[0:fluxQ_sz])

  const float dt        = 0.001f;
  const float FinalTime = 0.005f;
  const int   nSteps    = (int)(FinalTime / dt);
  printf("Running %d time steps (5 RK stages each)...\n", nSteps);

  double t0 = omp_get_wtime();

  for (int step = 0; step < nSteps; step++) {
    for (int rk = 0; rk < 5; rk++) {
      const float fa  = rk4a[rk];
      const float fb  = rk4b[rk];
      const float fdt = dt;

      // Zero rhsQ
      #pragma omp target teams distribute parallel for thread_limit(256)
      for (int i = 0; i < Ntotal; i++) d_rhsQ[i] = 0.0f;

      // Volume kernel (VolKernel)
      #pragma omp target teams distribute parallel for thread_limit(256)
      for (int gid = 0; gid < K * p_Np; gid++) {
        const int k     = gid / p_Np;
        const int n     = gid % p_Np;
        const int gbase = k * 12;
        const float drdx = d_vgeo[gbase+ 0], drdy = d_vgeo[gbase+ 1], drdz = d_vgeo[gbase+ 2];
        const float dsdx = d_vgeo[gbase+ 4], dsdy = d_vgeo[gbase+ 5], dsdz = d_vgeo[gbase+ 6];
        const float dtdx = d_vgeo[gbase+ 8], dtdy = d_vgeo[gbase+ 9], dtdz = d_vgeo[gbase+10];
        const int qbase = k * p_Nfields * BSIZE;
        float dHxdr=0,dHxds=0,dHxdt=0;
        float dHydr=0,dHyds=0,dHydt=0;
        float dHzdr=0,dHzds=0,dHzdt=0;
        float dExdr=0,dExds=0,dExdt=0;
        float dEydr=0,dEyds=0,dEydt=0;
        float dEzdr=0,dEzds=0,dEzdt=0;
        for (int m = 0; m < p_Np; m++) {
          const int didx = 4 * (n + m * BSIZE);
          const float Dr = d_DrDsDt[didx + 0];
          const float Ds = d_DrDsDt[didx + 1];
          const float Dt = d_DrDsDt[didx + 2];
          const float Hx = d_Q[qbase + m            ];
          const float Hy = d_Q[qbase + m +   BSIZE  ];
          const float Hz = d_Q[qbase + m + 2*BSIZE  ];
          const float Ex = d_Q[qbase + m + 3*BSIZE  ];
          const float Ey = d_Q[qbase + m + 4*BSIZE  ];
          const float Ez = d_Q[qbase + m + 5*BSIZE  ];
          dHxdr += Dr*Hx; dHxds += Ds*Hx; dHxdt += Dt*Hx;
          dHydr += Dr*Hy; dHyds += Ds*Hy; dHydt += Dt*Hy;
          dHzdr += Dr*Hz; dHzds += Ds*Hz; dHzdt += Dt*Hz;
          dExdr += Dr*Ex; dExds += Ds*Ex; dExdt += Dt*Ex;
          dEydr += Dr*Ey; dEyds += Ds*Ey; dEydt += Dt*Ey;
          dEzdr += Dr*Ez; dEzds += Ds*Ez; dEzdt += Dt*Ez;
        }
        const int rbase = n + p_Nfields * BSIZE * k;
        d_rhsQ[rbase          ] = -(drdy*dEzdr+dsdy*dEzds+dtdy*dEzdt - drdz*dEydr-dsdz*dEyds-dtdz*dEydt);
        d_rhsQ[rbase +   BSIZE] = -(drdz*dExdr+dsdz*dExds+dtdz*dExdt - drdx*dEzdr-dsdx*dEzds-dtdx*dEzdt);
        d_rhsQ[rbase + 2*BSIZE] = -(drdx*dEydr+dsdx*dEyds+dtdx*dEydt - drdy*dExdr-dsdy*dExds-dtdy*dExdt);
        d_rhsQ[rbase + 3*BSIZE] =  (drdy*dHzdr+dsdy*dHzds+dtdy*dHzdt - drdz*dHydr-dsdz*dHyds-dtdz*dHydt);
        d_rhsQ[rbase + 4*BSIZE] =  (drdz*dHxdr+dsdz*dHxds+dtdz*dHxdt - drdx*dHzdr-dsdx*dHzds-dtdx*dHzdt);
        d_rhsQ[rbase + 5*BSIZE] =  (drdx*dHydr+dsdx*dHyds+dtdx*dHydt - drdy*dHxdr-dsdy*dHxds-dtdy*dHxdt);
      }

      // Surface flux kernel (SurfFluxKernel)
      #pragma omp target teams distribute parallel for thread_limit(256)
      for (int gid = 0; gid < K * NfpNf; gid++) {
        const int k = gid / NfpNf;
        const int n = gid % NfpNf;
        const int m    = 7 * (k * NfpNf) + n;
        const int idM  = (int)d_surfinfo[m + 0*NfpNf];
        int       idP  = (int)d_surfinfo[m + 1*NfpNf];
        const float Fsc = d_surfinfo[m + 2*NfpNf];
        const float Bsc = d_surfinfo[m + 3*NfpNf];
        const float nx  = d_surfinfo[m + 4*NfpNf];
        const float ny  = d_surfinfo[m + 5*NfpNf];
        const float nz  = d_surfinfo[m + 6*NfpNf];
        float dHx, dHy, dHz, dEx, dEy, dEz;
        if (idP < 0) {
          dHx = Fsc*(0.0f - d_Q[idM + 0*BSIZE]); dHy = Fsc*(0.0f - d_Q[idM + 1*BSIZE]);
          dHz = Fsc*(0.0f - d_Q[idM + 2*BSIZE]); dEx = Fsc*(0.0f - d_Q[idM + 3*BSIZE]);
          dEy = Fsc*(0.0f - d_Q[idM + 4*BSIZE]); dEz = Fsc*(0.0f - d_Q[idM + 5*BSIZE]);
        } else {
          dHx = Fsc*(d_Q[idP + 0*BSIZE] - d_Q[idM + 0*BSIZE]);
          dHy = Fsc*(d_Q[idP + 1*BSIZE] - d_Q[idM + 1*BSIZE]);
          dHz = Fsc*(d_Q[idP + 2*BSIZE] - d_Q[idM + 2*BSIZE]);
          dEx = Fsc*(Bsc*d_Q[idP + 3*BSIZE] - d_Q[idM + 3*BSIZE]);
          dEy = Fsc*(Bsc*d_Q[idP + 4*BSIZE] - d_Q[idM + 4*BSIZE]);
          dEz = Fsc*(Bsc*d_Q[idP + 5*BSIZE] - d_Q[idM + 5*BSIZE]);
        }
        const float ndotdH = nx*dHx + ny*dHy + nz*dHz;
        const float ndotdE = nx*dEx + ny*dEy + nz*dEz;
        const int fbase = k * p_Nfields * NfpNf + n;
        d_fluxQ[fbase + 0*NfpNf] = -ny*dEz + nz*dEy + dHx - ndotdH*nx;
        d_fluxQ[fbase + 1*NfpNf] = -nz*dEx + nx*dEz + dHy - ndotdH*ny;
        d_fluxQ[fbase + 2*NfpNf] = -nx*dEy + ny*dEx + dHz - ndotdH*nz;
        d_fluxQ[fbase + 3*NfpNf] =  ny*dHz - nz*dHy + dEx - ndotdE*nx;
        d_fluxQ[fbase + 4*NfpNf] =  nz*dHx - nx*dHz + dEy - ndotdE*ny;
        d_fluxQ[fbase + 5*NfpNf] =  nx*dHy - ny*dHx + dEz - ndotdE*nz;
      }

      // Surface lift kernel (SurfLiftKernel)
      #pragma omp target teams distribute parallel for thread_limit(256)
      for (int gid = 0; gid < K * p_Np; gid++) {
        const int k = gid / p_Np;
        const int n = gid % p_Np;
        float rhsHx=0, rhsHy=0, rhsHz=0;
        float rhsEx=0, rhsEy=0, rhsEz=0;
        const int fbase = k * p_Nfields * NfpNf;
        for (int j = 0; j < NfpNf; j++) {
          const float L = d_LIFT[n + j * p_Np];
          rhsHx += L * d_fluxQ[fbase + j + 0*NfpNf];
          rhsHy += L * d_fluxQ[fbase + j + 1*NfpNf];
          rhsHz += L * d_fluxQ[fbase + j + 2*NfpNf];
          rhsEx += L * d_fluxQ[fbase + j + 3*NfpNf];
          rhsEy += L * d_fluxQ[fbase + j + 4*NfpNf];
          rhsEz += L * d_fluxQ[fbase + j + 5*NfpNf];
        }
        const int rbase = n + p_Nfields * BSIZE * k;
        d_rhsQ[rbase          ] += rhsHx;
        d_rhsQ[rbase +   BSIZE] += rhsHy;
        d_rhsQ[rbase + 2*BSIZE] += rhsHz;
        d_rhsQ[rbase + 3*BSIZE] += rhsEx;
        d_rhsQ[rbase + 4*BSIZE] += rhsEy;
        d_rhsQ[rbase + 5*BSIZE] += rhsEz;
      }

      // RK update kernel (RKUpdateKernel)
      #pragma omp target teams distribute parallel for thread_limit(256)
      for (int i = 0; i < Ntotal; i++) {
        float res   = fa * d_resQ[i] + fdt * d_rhsQ[i];
        d_resQ[i]   = res;
        d_Q[i]     += fb * res;
      }
    }
  }

  double t1      = omp_get_wtime();
  double elapsed = t1 - t0;

  double flopsV = (double)p_Np * p_Np * 36 + p_Np * 66;
  double flopsS = (double)p_Nfp * p_Nfaces * (15+10+36) + p_Np * (p_Nfaces * p_Nfp * 12 + 6);
  double flopsR = (double)p_Np * p_Nfields * 4;
  double gflops = 5.0 * nSteps * (flopsV + flopsS + flopsR) * ((double)K / (1.0e9 * elapsed));
  printf("Elapsed time: %.4f s,  estimated GFLOPS: %.3f\n", elapsed, gflops);

  // Copy Q back to host and report max|Ez|
  #pragma omp target update from(d_Q[0:Ntotal])
  float maxEz = 0.0f;
  for (int k = 0; k < K; k++)
    for (int n = 0; n < p_Np; n++) {
      float ez = d_Q[n + 5*BSIZE + k*p_Nfields*BSIZE];
      if (fabsf(ez) > maxEz) maxEz = fabsf(ez);
    }
  printf("max|Ez| = %e\n", maxEz);

  // Cleanup
  #pragma omp target exit data \
    map(delete: d_Q[0:Ntotal], d_vgeo[0:vgeo_sz], \
                d_DrDsDt[0:DrDsDt_sz], d_LIFT[0:LIFT_sz], \
                d_surfinfo[0:surfinfo_sz], \
                d_rhsQ[0:Ntotal], d_resQ[0:Ntotal], d_fluxQ[0:fluxQ_sz])

  free(d_Q);
  free(d_rhsQ);
  free(d_resQ);
  free(d_fluxQ);
  free(d_vgeo);
  free(d_DrDsDt);
  free(d_LIFT);
  free(d_surfinfo);

  return 0;
}
