// Minimod - Seismic wave simulation - OpenMP target offloading port
// Acoustic wave propagation using 8th-order finite differences (stencil radius R=4).
// Implements Perfectly Matched Layer (PML) absorbing boundary conditions.
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>

static constexpr float _fmax = 25.0f;
static constexpr float vmin  = 1500.0f;
static constexpr float vmax  = 4500.0f;
static constexpr float cfl   = 0.8f;
#define POW2(x) ((x) * (x))

using llint = long long int;
using uint  = unsigned int;

struct grid_t {
    llint ntaperx, ntapery, ntaperz;
    llint ndampx,  ndampy,  ndampz;
    llint nx, ny, nz;
    llint ldimx, ldimy, ldimz;
    llint dx, dy, dz;
    llint x1, x2, x3, x4, x5, x6;
    llint y1, y2, y3, y4, y5, y6;
    llint z1, z2, z3, z4, z5, z6;
    llint lx, ly, lz;
    llint ntx, nty, tsx, tsy;
};

static grid_t init_grid(llint nx, llint ny, llint nz, llint tsx = 10, llint tsy = 10) {
    grid_t g;
    g.nx=nx; g.ny=ny; g.nz=nz;
    g.dx=20; g.dy=20; g.dz=20;
    g.lx=4; g.ly=4; g.lz=4;
    g.ntaperx=3; g.ntapery=3; g.ntaperz=3;
    g.ldimx = nx + 4*g.lx;
    g.ldimy = ny + 2*g.ly;
    g.ldimz = ((nz + 2*g.lz + 31) / 32) * 32;
    printf("ldimx: %lld, ldimy: %lld, ldimz: %lld\n", g.ldimx, g.ldimy, g.ldimz);
    const float lambdamax = vmax / _fmax;
    g.ndampx = (llint)(g.ntaperx * lambdamax / g.dx);
    g.ndampy = (llint)(g.ntapery * lambdamax / g.dy);
    g.ndampz = (llint)(g.ntaperz * lambdamax / g.dz);
    g.x1=0; g.x2=g.ndampx; g.x3=g.ndampx; g.x4=g.nx-g.ndampx; g.x5=g.nx-g.ndampx; g.x6=g.nx;
    g.y1=0; g.y2=g.ndampy; g.y3=g.ndampy; g.y4=g.ny-g.ndampy; g.y5=g.ny-g.ndampy; g.y6=g.ny;
    g.z1=0; g.z2=g.ndampz; g.z3=g.ndampz; g.z4=g.nz-g.ndampz; g.z5=g.nz-g.ndampz; g.z6=g.nz;
    g.tsx=tsx; g.tsy=tsy;
    g.ntx=nx/tsx; g.nty=ny/tsy;
    printf("ndamp = %lld %lld %lld\n", g.ndampx, g.ndampy, g.ndampz);
    return g;
}

// IDX and stencil_lap are marked declare target so they are callable from device regions.
#pragma omp declare target

static int64_t IDX(int i, int j, int k, int lx, int ly, int lz, int ldimy, int ldimz) {
    return ((int64_t)(i + lx) * ldimy + (j + ly)) * ldimz + (k + lz);
}

static float stencil_lap(
    const float* u,
    int i, int j, int k,
    int lx, int ly, int lz, int ldimy, int ldimz,
    float coef0,
    float cx1, float cx2, float cx3, float cx4,
    float cy1, float cy2, float cy3, float cy4,
    float cz1, float cz2, float cz3, float cz4)
{
#define U(di,dj,dk) u[IDX(i+(di),j+(dj),k+(dk),lx,ly,lz,ldimy,ldimz)]
    return coef0 * U(0,0,0)
         + cx1*(U(1,0,0)+U(-1,0,0)) + cy1*(U(0,1,0)+U(0,-1,0)) + cz1*(U(0,0,1)+U(0,0,-1))
         + cx2*(U(2,0,0)+U(-2,0,0)) + cy2*(U(0,2,0)+U(0,-2,0)) + cz2*(U(0,0,2)+U(0,0,-2))
         + cx3*(U(3,0,0)+U(-3,0,0)) + cy3*(U(0,3,0)+U(0,-3,0)) + cz3*(U(0,0,3)+U(0,0,-3))
         + cx4*(U(4,0,0)+U(-4,0,0)) + cy4*(U(0,4,0)+U(0,-4,0)) + cz4*(U(0,0,4)+U(0,0,-4));
#undef U
}

#pragma omp end declare target

// -----------------------------------------------------------------------
// launch_inner: 8th-order FD stencil update in the non-PML interior region
// -----------------------------------------------------------------------
static void launch_inner(
    float* u, float* v, float* vp,
    int x0, int x1, int y0, int y1, int z0, int z1,
    int lx, int ly, int lz, int ldimy, int ldimz,
    float coef0,
    float cx1, float cx2, float cx3, float cx4,
    float cy1, float cy2, float cy3, float cy4,
    float cz1, float cz2, float cz3, float cz4)
{
    if (x0 >= x1 || y0 >= y1 || z0 >= z1) return;
    #pragma omp target teams distribute parallel for collapse(3) thread_limit(256)
    for (int i = x0; i < x1; i++)
    for (int j = y0; j < y1; j++)
    for (int k = z0; k < z1; k++) {
        const float lap = stencil_lap(u, i, j, k, lx, ly, lz, ldimy, ldimz, coef0,
                                      cx1, cx2, cx3, cx4,
                                      cy1, cy2, cy3, cy4,
                                      cz1, cz2, cz3, cz4);
        const int64_t idx = IDX(i, j, k, lx, ly, lz, ldimy, ldimz);
        v[idx] = 2.0f*u[idx] + vp[idx]*lap - v[idx];
    }
}

// -----------------------------------------------------------------------
// launch_pml: PML absorbing boundary update
// -----------------------------------------------------------------------
static void launch_pml(
    float* u, float* v, float* vp, float* phi, float* eta,
    int x0, int x1, int y0, int y1, int z0, int z1,
    int lx, int ly, int lz, int ldimy, int ldimz,
    float coef0,
    float cx1, float cx2, float cx3, float cx4,
    float cy1, float cy2, float cy3, float cy4,
    float cz1, float cz2, float cz3, float cz4,
    float hdx_2, float hdy_2, float hdz_2)
{
    if (x0 >= x1 || y0 >= y1 || z0 >= z1) return;
    #pragma omp target teams distribute parallel for collapse(3) thread_limit(256)
    for (int i = x0; i < x1; i++)
    for (int j = y0; j < y1; j++)
    for (int k = z0; k < z1; k++) {
        const float lap = stencil_lap(u, i, j, k, lx, ly, lz, ldimy, ldimz, coef0,
                                      cx1, cx2, cx3, cx4,
                                      cy1, cy2, cy3, cy4,
                                      cz1, cz2, cz3, cz4);
        const int64_t idx = IDX(i, j, k, lx, ly, lz, ldimy, ldimz);
#define E(di,dj,dk)  eta[IDX(i+(di),j+(dj),k+(dk),lx,ly,lz,ldimy,ldimz)]
#define UU(di,dj,dk)   u[IDX(i+(di),j+(dj),k+(dk),lx,ly,lz,ldimy,ldimz)]
        const float s_eta_c = E(0,0,0);
        v[idx] = ((2.0f*s_eta_c + 2.0f - s_eta_c*s_eta_c)*u[idx]
                  + vp[idx]*(lap + phi[idx]) - v[idx]) / (2.0f*s_eta_c + 1.0f);
        phi[idx] = (phi[idx]
                    - ((E(1,0,0)-E(-1,0,0))*(UU(1,0,0)-UU(-1,0,0))*hdx_2
                      +(E(0,1,0)-E(0,-1,0))*(UU(0,1,0)-UU(0,-1,0))*hdy_2
                      +(E(0,0,1)-E(0,0,-1))*(UU(0,0,1)-UU(0,0,-1))*hdz_2))
                   / (1.0f + s_eta_c);
#undef E
#undef UU
    }
}

// -----------------------------------------------------------------------
// Host-side PML initialisation helpers (unchanged from Kokkos version)
// -----------------------------------------------------------------------
static void pml_profile_init(float* profile, llint i_min, llint i_max,
                              llint n_first, llint n_last, float scale)
{
    const llint shift = i_min - 1;
    const llint first_end = n_first + shift;
    const llint n = i_max - i_min + 1;
    const llint last_beg = n - n_last + 1 + shift;
    for (llint i = i_min; i <= i_max; ++i) profile[i] = 0.f;
    float tmp = scale / POW2(first_end - shift);
    for (llint i = 1; i <= first_end - shift; ++i) profile[first_end - i + 1] = POW2(i) * tmp;
    for (llint i = 1; i <= n - (last_beg - i_min); ++i) profile[last_beg + i - 1] = POW2(i) * tmp;
}

static void pml_extend_region(float* eta,
    const float* etax, const float* etay, const float* etaz,
    llint ldimy, llint ldimz, llint lx, llint ly, llint lz,
    llint xbeg, llint xend, llint ybeg, llint yend, llint zbeg, llint zend)
{
    const llint ng = 1;
    for (llint ix = xbeg-ng; ix <= xend+ng; ++ix)
    for (llint iy = ybeg-ng; iy <= yend+ng; ++iy)
    for (llint iz = zbeg-ng; iz <= zend+ng; ++iz)
        eta[IDX((int)ix,(int)iy,(int)iz,(int)lx,(int)ly,(int)lz,(int)ldimy,(int)ldimz)]
            = etax[ix] + etay[iy] + etaz[iz];
}

static void init_eta(const grid_t& g, float dt_sch, float* eta)
{
    const llint total = g.ldimx * g.ldimy * g.ldimz;
    memset(eta, 0, total * sizeof(float));
    float param;
    float* etax = new float[g.nx+2]();
    float* etay = new float[g.ny+2]();
    float* etaz = new float[g.nz+2]();
    param = dt_sch * 3.0f * vmax * logf(1000.0f) / (2.0f * g.ndampx * g.dx);
    pml_profile_init(etax, 0, g.nx+1, g.ndampx, g.ndampx, param);
    param = dt_sch * 3.0f * vmax * logf(1000.0f) / (2.0f * g.ndampy * g.dy);
    pml_profile_init(etay, 0, g.ny+1, g.ndampy, g.ndampy, param);
    param = dt_sch * 3.0f * vmax * logf(1000.0f) / (2.0f * g.ndampz * g.dz);
    pml_profile_init(etaz, 0, g.nz+1, g.ndampz, g.ndampz, param);
    const llint lx=g.lx, ly=g.ly, lz=g.lz;
    const llint ldimy=g.ldimy, ldimz=g.ldimz;
    pml_extend_region(eta,etax,etay,etaz,ldimy,ldimz,lx,ly,lz, 1,g.nx,   1,g.ny,   g.z1+1,g.z2);
    pml_extend_region(eta,etax,etay,etaz,ldimy,ldimz,lx,ly,lz, 1,g.nx,   1,g.ny,   g.z5+1,g.z6);
    pml_extend_region(eta,etax,etay,etaz,ldimy,ldimz,lx,ly,lz, 1,g.nx,   g.y1+1,g.y2, g.z3+1,g.z4);
    pml_extend_region(eta,etax,etay,etaz,ldimy,ldimz,lx,ly,lz, 1,g.nx,   g.y5+1,g.y6, g.z3+1,g.z4);
    pml_extend_region(eta,etax,etay,etaz,ldimy,ldimz,lx,ly,lz, g.x1+1,g.x2, g.y3+1,g.y4, g.z3+1,g.z4);
    pml_extend_region(eta,etax,etay,etaz,ldimy,ldimz,lx,ly,lz, g.x5+1,g.x6, g.y3+1,g.y4, g.z3+1,g.z4);
    delete[] etax; delete[] etay; delete[] etaz;
}

static void init_coef(float dx, float coef[5]) {
    const float dx2 = dx * dx;
    coef[0] = -205.0f/72.0f/dx2; coef[1] =   8.0f/ 5.0f/dx2;
    coef[2] =   -1.0f/ 5.0f/dx2; coef[3] =   8.0f/315.0f/dx2;
    coef[4] =   -1.0f/560.0f/dx2;
}

static float compute_dt_sch(const float cx[5], const float cy[5], const float cz[5]) {
    float ftmp = fabsf(cx[0]) + fabsf(cy[0]) + fabsf(cz[0]);
    for (uint i = 1; i < 5; i++)
        ftmp += 2.0f * (fabsf(cx[i]) + fabsf(cy[i]) + fabsf(cz[i]));
    return 2.0f * cfl / (sqrtf(ftmp) * vmax);
}

static void gaussian_source(uint nt, float dt, float* src) {
    const float sigma = 0.6f * _fmax, tau = 1.0f, scale = 8.0f;
    for (uint it = 1; it <= nt; ++it) {
        const float t = dt * (it - 1);
        src[it-1] = -2.0f*scale*sigma*(sigma - 2.0f*sigma*scale*POW2(sigma*t - tau))
                    * expf(-scale * POW2(sigma*t - tau));
    }
}

int main(int argc, char* argv[])
{
    llint nx=100, ny=100, nz=100;
    uint nsteps=100, niters=1;
    bool warm_up_enabled=false, finalio=false;

    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i],"--grid")==0   && i+1<argc) nx=ny=nz=strtoll(argv[++i],nullptr,10);
        else if (strcmp(argv[i],"--nsteps")==0 && i+1<argc) { nsteps=(uint)strtol(argv[++i],nullptr,10); printf("nsteps = %u\n",nsteps); }
        else if (strcmp(argv[i],"--niters")==0 && i+1<argc) { niters=(uint)strtol(argv[++i],nullptr,10); printf("niters = %u\n",niters); }
        else if (strcmp(argv[i],"--warm-up")==0) warm_up_enabled=true;
        else if (strcmp(argv[i],"--finalio")==0) finalio=true;
    }

    double total_kernel_time = 0.0;
    bool warm_up_iter = warm_up_enabled;
    const uint total_iters = niters + (warm_up_enabled ? 1u : 0u);

    for (uint iiter = 0; iiter < total_iters; ++iiter) {
        const grid_t g = init_grid(nx, ny, nz);
        printf("grid = %lld %lld %lld\n", g.nx, g.ny, g.nz);

        const llint sx = nx/2, sy = ny/2, sz = nz/2;
        const llint total_cells = g.ldimx * g.ldimy * g.ldimz;

        float* h_vp  = new float[total_cells]();
        float* h_phi = new float[total_cells]();
        float* h_eta = new float[total_cells]();
        float* source = new float[nsteps];

        float coefx[5], coefy[5], coefz[5];
        init_coef((float)g.dx, coefx);
        init_coef((float)g.dy, coefy);
        init_coef((float)g.dz, coefz);
        const float dt_sch = compute_dt_sch(coefx, coefy, coefz);

        const float vp_all = POW2(2000.0f * dt_sch);
        for (llint i=0; i<nx; ++i)
        for (llint j=0; j<ny; ++j)
        for (llint k=0; k<nz; ++k) {
            const int64_t idx = IDX((int)i,(int)j,(int)k,(int)g.lx,(int)g.ly,(int)g.lz,(int)g.ldimy,(int)g.ldimz);
            h_vp[idx]  = vp_all;
            h_phi[idx] = 0.0f;
        }
        gaussian_source(nsteps, dt_sch, source);
        init_eta(g, dt_sch, h_eta);

        const float hdx_2 = 1.0f / (4.0f * POW2(g.dx));
        const float hdy_2 = 1.0f / (4.0f * POW2(g.dy));
        const float hdz_2 = 1.0f / (4.0f * POW2(g.dz));
        const float coef0 = coefx[0] + coefy[0] + coefz[0];

        // Allocate device buffers
        float* d_u   = (float*)malloc(total_cells * sizeof(float));
        float* d_v   = (float*)malloc(total_cells * sizeof(float));
        float* d_phi = (float*)malloc(total_cells * sizeof(float));
        float* d_eta = (float*)malloc(total_cells * sizeof(float));
        float* d_vp  = (float*)malloc(total_cells * sizeof(float));

        #pragma omp target enter data map(alloc: d_u[0:total_cells], d_v[0:total_cells])
        #pragma omp target enter data map(alloc: d_phi[0:total_cells], d_eta[0:total_cells], d_vp[0:total_cells])

        // Copy initial data to host buffers, then push to device
        memcpy(d_vp,  h_vp,  total_cells * sizeof(float));
        memcpy(d_phi, h_phi, total_cells * sizeof(float));
        memcpy(d_eta, h_eta, total_cells * sizeof(float));
        #pragma omp target update to(d_vp[0:total_cells], d_phi[0:total_cells], d_eta[0:total_cells])

        // Zero u and v on device
        {
            llint tc = total_cells;
            #pragma omp target teams distribute parallel for thread_limit(256)
            for (llint i = 0; i < tc; i++) {
                d_u[i] = 0.0f;
                d_v[i] = 0.0f;
            }
        }

        const int lx=(int)g.lx, ly=(int)g.ly, lz=(int)g.lz;
        const int ldimy=(int)g.ldimy, ldimz=(int)g.ldimz;

        const float cx1=coefx[1], cx2=coefx[2], cx3=coefx[3], cx4=coefx[4];
        const float cy1=coefy[1], cy2=coefy[2], cy3=coefy[3], cy4=coefy[4];
        const float cz1=coefz[1], cz2=coefz[2], cz3=coefz[3], cz4=coefz[4];

        const double t_start = omp_get_wtime();

        for (uint istep = 1; istep <= nsteps; ++istep) {
            launch_pml(d_u,d_v,d_vp,d_phi,d_eta, 0,(int)nx,    0,(int)ny,    (int)g.z1,(int)g.z2,
                       lx,ly,lz,ldimy,ldimz, coef0, cx1,cx2,cx3,cx4, cy1,cy2,cy3,cy4, cz1,cz2,cz3,cz4, hdx_2,hdy_2,hdz_2);
            launch_pml(d_u,d_v,d_vp,d_phi,d_eta, 0,(int)nx,    (int)g.y1,(int)g.y2, (int)g.z3,(int)g.z4,
                       lx,ly,lz,ldimy,ldimz, coef0, cx1,cx2,cx3,cx4, cy1,cy2,cy3,cy4, cz1,cz2,cz3,cz4, hdx_2,hdy_2,hdz_2);
            launch_pml(d_u,d_v,d_vp,d_phi,d_eta, (int)g.x1,(int)g.x2, (int)g.y3,(int)g.y4, (int)g.z3,(int)g.z4,
                       lx,ly,lz,ldimy,ldimz, coef0, cx1,cx2,cx3,cx4, cy1,cy2,cy3,cy4, cz1,cz2,cz3,cz4, hdx_2,hdy_2,hdz_2);
            launch_inner(d_u,d_v,d_vp,             (int)g.x3,(int)g.x4, (int)g.y3,(int)g.y4, (int)g.z3,(int)g.z4,
                         lx,ly,lz,ldimy,ldimz, coef0, cx1,cx2,cx3,cx4, cy1,cy2,cy3,cy4, cz1,cz2,cz3,cz4);
            launch_pml(d_u,d_v,d_vp,d_phi,d_eta, (int)g.x5,(int)g.x6, (int)g.y3,(int)g.y4, (int)g.z3,(int)g.z4,
                       lx,ly,lz,ldimy,ldimz, coef0, cx1,cx2,cx3,cx4, cy1,cy2,cy3,cy4, cz1,cz2,cz3,cz4, hdx_2,hdy_2,hdz_2);
            launch_pml(d_u,d_v,d_vp,d_phi,d_eta, 0,(int)nx,    (int)g.y5,(int)g.y6, (int)g.z3,(int)g.z4,
                       lx,ly,lz,ldimy,ldimz, coef0, cx1,cx2,cx3,cx4, cy1,cy2,cy3,cy4, cz1,cz2,cz3,cz4, hdx_2,hdy_2,hdz_2);
            launch_pml(d_u,d_v,d_vp,d_phi,d_eta, 0,(int)nx,    0,(int)ny,    (int)g.z5,(int)g.z6,
                       lx,ly,lz,ldimy,ldimz, coef0, cx1,cx2,cx3,cx4, cy1,cy2,cy3,cy4, cz1,cz2,cz3,cz4, hdx_2,hdy_2,hdz_2);

            // Inject source into d_v
            const float src_val = source[istep-1];
            const int64_t src_idx = IDX((int)sx,(int)sy,(int)sz,lx,ly,lz,ldimy,ldimz);
            #pragma omp target
            d_v[src_idx] += src_val;

            // Swap time levels: d_u becomes the new current field
            float* tmp = d_u; d_u = d_v; d_v = tmp;
        }

        const double kernel_time = warm_up_iter ? 0.0 : (omp_get_wtime() - t_start);

        // Checksum: min/max of d_u
        float min_u = FLT_MAX, max_u = -FLT_MAX;
        {
            llint tc = total_cells;
            #pragma omp target teams distribute parallel for thread_limit(256) \
                    reduction(min:min_u) reduction(max:max_u)
            for (llint i = 0; i < tc; i++) {
                if (d_u[i] < min_u) min_u = d_u[i];
                if (d_u[i] > max_u) max_u = d_u[i];
            }
        }
        printf("Checksum: min_u, max_u = %f, %f\n", min_u, max_u);

        if (finalio && !warm_up_iter) {
            #pragma omp target update from(d_u[0:total_cells])
            char fname[64];
            snprintf(fname, sizeof(fname), "snapshot.it%u.n%lld.raw", nsteps, nz);
            FILE* f = fopen(fname, "wb");
            for (llint i=0; i<nx; ++i)
            for (llint j=0; j<ny; ++j) {
                const int64_t base = IDX((int)i,(int)j,0,lx,ly,lz,ldimy,ldimz);
                fwrite(d_u + base, sizeof(float), nz, f);
            }
            fclose(f);
        }

        total_kernel_time += kernel_time;
        warm_up_iter = false;

        #pragma omp target exit data map(delete: d_u[0:total_cells], d_v[0:total_cells])
        #pragma omp target exit data map(delete: d_phi[0:total_cells], d_eta[0:total_cells], d_vp[0:total_cells])
        free(d_u); free(d_v); free(d_phi); free(d_eta); free(d_vp);
        delete[] h_vp; delete[] h_phi; delete[] h_eta; delete[] source;
    }

    printf("Average kernel time per iteration: %g s\n", total_kernel_time / niters);
    return 0;
}
