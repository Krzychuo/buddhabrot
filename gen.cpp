#include <omp.h>
#include <bits/stdc++.h>
using namespace std;
using f64 = double;
using u8 = unsigned char;
#define f first
#define s second

const int N = 4096;
const long long RAND_ITER = 6'000'000'000;
const int MX_ITER = 200;
const int DROP = 0;
const int NUM_THREADS = 16;
const f64 L = -1.25;
const f64 R = 1.25;
const f64 D = -1.60;
const f64 U = 0.90;
const f64 LEN = (R-L)/(N-1);

#pragma pack(push, 1)
struct RGB{
    u8 r, g, b;
};
#pragma pack(pop)

std::random_device rd;
std::seed_seq seq{
    rd(),rd(),rd(),rd(),
    rd(),rd(),rd(),rd(),
    rd(),rd(),rd(),rd(),
    rd(),rd(),rd(),rd()
};
std::vector<std::uint32_t> seeds(NUM_THREADS);

struct PCG32 {
    uint64_t state;
    uint64_t inc;

    uint32_t operator()() {
        uint64_t oldstate = state;
        state = oldstate * 6364136223846793005ULL + (inc | 1);
        uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
        uint32_t rot = oldstate >> 59u;
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }

    f64 uniform_neg2_to_2() {
        uint32_t r = (*this)() >> 8;
        f64 u = r * (1.0 / 16777216.0);
        return 4.0 * u - 2.0;
    }
};

RGB image[N][N];
f64 cnt[N*N];
pair<f64,f64> hist[NUM_THREADS][MX_ITER];
bool f = false;

inline bool in_cardioid_or_bulb(const f64 cr, const f64 ci){
    f64 p = (cr-0.25)*(cr-0.25)+ci*ci;
    if ((2*p+cr-0.25)*(2*p+cr-0.25) < p) return true;            // cardioid
    // f64 p = hypot(cr-0.25, ci);
    // if (cr < p - 2*p*p + 0.25) return true;            // cardioid
    if ((cr+1)*(cr+1) + ci*ci < 0.0625) return true;   // period-2 bulb
    return false;
}

inline void splat(f64 x, f64 y, f64 (&cnt)[N*N]) {
    int x0 = floor(x), y0 = floor(y);
    x -= x0; y -= y0;
    if (0<=x0 && x0+1<N && y0+1<N) { // 0<=y0 needn't be checked
        cnt[x0*N+y0] += (1-x)*(1-y); // atomic updates are not needed
        cnt[x0*N+y0+1] += (1-x)*y; // if 2 threads want to write
        cnt[(x0+1)*N+y0] += x*(1-y); // there will be negligable error
        cnt[(x0+1)*N+y0+1] += x*y;
    }
}

int main(){
    seq.generate(seeds.begin(), seeds.end());
    
    #pragma omp parallel
    {
        int thr_num = omp_get_thread_num();
        PCG32 rng{(uint64_t)seeds[thr_num], 54u};
        #pragma omp for schedule(static)
        for(long long _=0;_<RAND_ITER;_++){
            f64 x0 = rng.uniform_neg2_to_2();
            f64 y0 = rng.uniform_neg2_to_2();
            if(in_cardioid_or_bulb(x0, y0)) continue;
            f64 x=0, y=0, x2=0, y2=0;
            int iter = 0;

            while(x2 + y2 < 4 && iter < MX_ITER){
                y = 2 * y * x + y0;
                x = x2 - y2 + x0;
                y2 = y * y;
                x2 = x * x;
                hist[thr_num][iter] = {x, y};
                iter += 1;
            }

            if(iter < MX_ITER && iter > DROP){
                for(int i=0;i<iter-1;i++){
                    if( hist[thr_num][i].f < D - LEN || 
                        hist[thr_num][i].f > U + LEN || 
                        hist[thr_num][i].s < L - LEN || 
                        hist[thr_num][i].s > R + LEN) continue;
                    f64 px = (hist[thr_num][i].f - D) / LEN;
                    f64 py = ((hist[thr_num][i].s >= 0 ? hist[thr_num][i].s : -hist[thr_num][i].s) - L) / LEN;
                    splat(px, py, cnt);
                }
            }
        }
    }

    for(int i=0;i<N;i++){
        for(int j=0;j<N/2-1;j++){
            cnt[i*N+j] += cnt[i*N+N-j-1];
        }
        f64 v = cnt[i*N+N/2-1] + cnt[i*N+N/2];
        cnt[i*N+N/2-1] = v;
        cnt[i*N+N/2] = v;
    }

    f64 mx = 0;
    for(int i=0;i<N;i++){
        for(int j=0;j<N/2;j++){
            mx = max(mx, cnt[i*N+j]);
        }
    }

    std::vector<int> flat; flat.reserve(N*N);
    for (int i=0;i<N;i++) for (int j=0;j<N;j++) flat.push_back(cnt[i*N+j]);
    size_t k = (size_t)std::floor(0.995 * flat.size());
    std::nth_element(flat.begin(), flat.begin()+k, flat.end());
    mx = flat[k];

    // const double exposure = 0.02;
    // double denom = std::log(1.0 + exposure * mx);

    for (int i=0;i<N;i++) {
        for (int j=0;j<N;j++) {
            //f64 v = (f64)log(1.0 + exposure * min(cnt[i][j], mx)) / denom;
            f64 v = min((f64)1, f64(cnt[i*N+j]) / f64(mx));
            unsigned char g = (unsigned char)std::round(255.0 * v);
            image[i][j] = {g,g,g};
        }
    }

    FILE* f = fopen("buddhacpp.ppm", "wb");
    fprintf(f, "P6\n%d %d\n255\n", N, N);
    for (int i=0;i<N;++i) {
        for(int j=0;j<N;j++){
            fwrite(reinterpret_cast<const void*>(&image[i][j]), sizeof(RGB), 1, f);
        }  
    }
    fclose(f);

    return 0;
}