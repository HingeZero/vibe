// HingeZero C++20 real-vector VIBE adaptation of David Duffy's
// HingeZero_Native_1B.cpp. The original bipolar engine is bundled unchanged.
// Original: bipolar 128-bit records, 256 fixed-field memories, 8 dual-tanh
// steps, whole-bank scan, tagged witness, append-only file. This adaptation:
// FLOAT32 vectors because yi-128-ip is NOT bipolar; no lossy sign shortlist.
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#endif
using namespace std;
namespace fs=std::filesystem;
namespace {
thread_local string last_error;
constexpr float ALPHA=4.5f,EPS=.1f,LAMBDA=.02f;
using Vec=vector<float>;
void unit(Vec&v){
 double s=0;for(float f:v)s+=double(f)*f;
 if(s>1e-24){float a=float(1.0/sqrt(s));for(float &x:v)x*=a;}
 else fill(v.begin(),v.end(),0.f);
}
float dot(const float*a,const float*b,int d){
 float s=0;for(int j=0;j<d;++j)s+=a[j]*b[j];return s;
}

// SIMD exact inner product. Values are promoted to float64 before multiplying,
// preserving VIBE's original floating-point scoring semantics. Runtime dispatch
// means the shared object remains safe on CPUs that do not support AVX2.
#if defined(__x86_64__) && defined(__GNUC__)
__attribute__((target("avx2")))
double dot_ip_avx2(const float* a,const float* b,int d) {
    __m256d s0=_mm256_setzero_pd(),s1=s0,s2=s0,s3=s0;
    int j=0;
    for(; j+15<d; j+=16) {
        const __m256 ar0=_mm256_loadu_ps(a+j), ar1=_mm256_loadu_ps(a+j+8);
        const __m256 br0=_mm256_loadu_ps(b+j), br1=_mm256_loadu_ps(b+j+8);
        s0=_mm256_add_pd(s0,_mm256_mul_pd(
            _mm256_cvtps_pd(_mm256_castps256_ps128(ar0)),
            _mm256_cvtps_pd(_mm256_castps256_ps128(br0))));
        s1=_mm256_add_pd(s1,_mm256_mul_pd(
            _mm256_cvtps_pd(_mm256_extractf128_ps(ar0,1)),
            _mm256_cvtps_pd(_mm256_extractf128_ps(br0,1))));
        s2=_mm256_add_pd(s2,_mm256_mul_pd(
            _mm256_cvtps_pd(_mm256_castps256_ps128(ar1)),
            _mm256_cvtps_pd(_mm256_castps256_ps128(br1))));
        s3=_mm256_add_pd(s3,_mm256_mul_pd(
            _mm256_cvtps_pd(_mm256_extractf128_ps(ar1,1)),
            _mm256_cvtps_pd(_mm256_extractf128_ps(br1,1))));
    }
    s0=_mm256_add_pd(s0,s1); s2=_mm256_add_pd(s2,s3);
    s0=_mm256_add_pd(s0,s2);
    alignas(32) double out[4];
    _mm256_store_pd(out,s0);
    double score=out[0]+out[1]+out[2]+out[3];
    for(;j<d;++j)score+=double(a[j])*double(b[j]);
    return score;
}
#endif
double dot_ip_scalar(const float* a,const float* b,int d) {
    double score=0;
    for(int j=0;j<d;++j)score+=double(a[j])*double(b[j]);
    return score;
}
using DotImpl=double(*)(const float*,const float*,int);
DotImpl select_dot_ip() {
#if defined(__x86_64__) && defined(__GNUC__)
    if(__builtin_cpu_supports("avx2"))return &dot_ip_avx2;
#endif
    return &dot_ip_scalar;
}


struct ClusterIndex {
  int d=0, nlist=0;
  vector<Vec> centroids;
  vector<uint32_t> offsets;
  vector<int32_t> ids;
  vector<float> packed;
  // Construction is deterministic and uses only the stored bank, not queries
  // or test ground truth. Original float32 records remain intact.
  ClusterIndex(const float*bank,uint64_t n,int dim,int clusters=128):
      d(dim),nlist(int(min<uint64_t>(clusters,n))) {
    const int sample=int(min<uint64_t>(8192,n));
    centroids.assign(nlist,Vec(d,0));
    uint64_t seed=0x4d595df4d0f33173ULL;
    auto random64=[&](){
       seed+=0x9E3779B97F4A7C15ULL;
       uint64_t z=seed;
       z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;
       z=(z^(z>>27))*0x94D049BB133111EBULL;
       return z^(z>>31);
    };
    vector<int32_t> train(sample);
    for(int i=0;i<sample;++i)train[i]=int32_t(random64()%n);
    for(int c=0;c<nlist;++c) {
        auto id=train[(c*sample)/nlist];
        const float*row=bank+size_t(id)*d;
        copy_n(row,d,centroids[c].begin());
        unit(centroids[c]);
    }
    // Spherical k-means: cosine clusters for maximum inner-product search.
    for(int iter=0;iter<4;++iter){
       vector<Vec> sums(nlist,Vec(d,0.f));
       vector<int> counts(nlist,0);
       for(int32_t id:train) {
          const float*row=bank+size_t(id)*d;
          int label=label_for(row);
          Vec& dst=sums[label];
          for(int j=0;j<d;++j)dst[j]+=row[j];
          ++counts[label];
       }
       for(int c=0;c<nlist;++c){
          if(counts[c]){unit(sums[c]);centroids[c]=std::move(sums[c]);}
       }
    }
    vector<uint32_t> counts(nlist,0);
    // One offline pass assigns every original record to exactly one cluster.
    for(uint64_t i=0;i<n;++i)++counts[label_for(bank+size_t(i)*d)];
    offsets.assign(nlist+1,0);
    for(int c=0;c<nlist;++c)offsets[c+1]=offsets[c]+counts[c];
    ids.resize(n);
    packed.resize(size_t(n)*d);
    vector<uint32_t> cursor(offsets.begin(),offsets.end()-1);
    for(uint64_t i=0;i<n;++i){
       const float*record=bank+size_t(i)*d;
       uint32_t idx=cursor[label_for(record)]++;
       ids[idx]=int32_t(i);
       copy_n(record,d,packed.data()+size_t(idx)*d);
    }
  }
  int label_for(const float*row)const{
     int best=0;float best_value=-numeric_limits<float>::infinity();
     for(int c=0;c<nlist;++c){
        float similarity=dot(row,centroids[c].data(),d);
        if(similarity>best_value){best_value=similarity;best=c;}
     }
     return best;
  }
  vector<int> probe_order(const float*q)const{
     vector<pair<float,int>> order;
     order.reserve(nlist);
     for(int c=0;c<nlist;++c)
        order.emplace_back(dot(q,centroids[c].data(),d),c);
     sort(order.begin(),order.end(),[](auto const&a,auto const&b){
        if(a.first==b.first)return a.second<b.second;
        return a.first>b.first;
     });
     vector<int> ranked;
     ranked.reserve(nlist);
     for(auto const&it:order)ranked.push_back(it.second);
     return ranked;
  }
};

struct Record{int32_t id;double score;};
bool better(const Record&a,const Record&b){
 return a.score!=b.score ? a.score>b.score : a.id<b.id;
}
struct Memory{
 int fd=-1,d,metric,field_limit;
 void* mapping=MAP_FAILED;
 const float*bank=nullptr;
 uint64_t count;
 size_t bytes;
 vector<Vec> field;
 unique_ptr<ClusterIndex> coarse_index;
 Memory(const char*path,uint64_t n,int dimensions,int m,int limit):
   d(dimensions),metric(m),field_limit(limit),count(n){
  if(!path||d<1||d>65536||n<1||n>uint64_t(INT32_MAX)||
     limit<1||limit>8192||metric<0||metric>3)
     throw runtime_error("invalid native dataset settings");
  if(std::endian::native!=std::endian::little)
     throw runtime_error("only little-endian float32 supported");
  if(n>SIZE_MAX/(size_t(d)*sizeof(float)))
     throw runtime_error("native file exceeds addressable range");
  bytes=size_t(n)*size_t(d)*sizeof(float);
  fd=::open(path,O_RDONLY);
  if(fd<0)throw runtime_error("cannot open native bank");
  try{
    struct stat st{};
    if(::fstat(fd,&st)!=0||uint64_t(st.st_size)!=bytes)
      throw runtime_error("native bank size does not match manifest");
    mapping=::mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0);
    if(mapping==MAP_FAILED)throw runtime_error("native mmap failed");
    bank=static_cast<const float*>(mapping);
    for(int i=0;i<std::min<uint64_t>(n,uint64_t(limit));++i){
      Vec v(bank+size_t(i)*d,bank+size_t(i+1)*d);
      for(float x:v)if(!isfinite(x))throw runtime_error("nonfinite field");
      unit(v);field.push_back(std::move(v));
    }
    if(metric==1){coarse_index=make_unique<ClusterIndex>(bank,count,d);}
  }catch(...){
    if(mapping!=MAP_FAILED)::munmap(mapping,bytes);
    ::close(fd);mapping=MAP_FAILED;fd=-1;throw;
  }
 }
 ~Memory(){
   if(mapping!=MAP_FAILED)::munmap(mapping,bytes);
   if(fd>=0)::close(fd);
 }
 bool verify(int64_t id,const float*state)const{
   return state&&id>=0&&uint64_t(id)<count&&
    memcmp(bank+size_t(id)*d,state,size_t(d)*4)==0;
 }
 Vec refine(const float*cue,int steps)const{
   if(!cue||steps<0||steps>64)throw runtime_error("invalid refinement");
   Vec x(cue,cue+d);
   for(float a:x)if(!isfinite(a))throw runtime_error("nonfinite cue");
   // Zero-step HingeZero: x_0 is returned unchanged. We still enter the
   // same native refine/query/witness pipeline, but execute no iteration.
   // This is not the separate enabled==0 "raw" bypass.
   if(steps==0)return x;
   unit(x);
   if(dot(x.data(),x.data(),d)<1e-15f){
     Vec avg(d,0);
     for(const Vec&m:field)for(int j=0;j<d;++j)avg[j]+=m[j];
     unit(avg);x.swap(avg);
   }
   for(int step=0;step<steps;++step){
     Vec fb(d,0);
     double denom=1e-12;
     for(const Vec&m:field){
       float h=dot(x.data(),m.data(),d);
       float w=tanh(h)+ALPHA*tanh(2*h);
       denom+=fabs(double(w));
       for(int j=0;j<d;++j)fb[j]+=m[j]*w;
     }
     for(int j=0;j<d;++j)x[j]=(1-LAMBDA)*x[j]+EPS*float(fb[j]/denom);
     unit(x);
   }
   return x;
 }
 double score(const float*record,const Vec&q,double nq)const{
   double s=0;
   if(metric==0){
     for(int j=0;j<d;++j){double diff=double(record[j])-q[j];s-=diff*diff;}
     return s;
   }
   double nr=0;
   for(int j=0;j<d;++j){
     double v=record[j];s+=v*q[j];
     if(metric==2)nr+=v*v;
   }
   if(metric==2)return nr>0&&nq>0?s/(sqrt(nr)*nq):0;
   return s;
 }
 void query(const float*original,int32_t k,int steps,int nprobe,
            int32_t*out,double*stats)const{
    if(!original||!out||k<1||uint64_t(k)>count)
       throw runtime_error("invalid query");
    auto t0=chrono::steady_clock::now();
    // Single HingeZero pipeline: prepare the initial state in refine()
    // then perform all-memory search and witnessed validation. No bypass.
    if(steps!=0) throw runtime_error("submitted HingeZero step count must be zero");
    Vec q=refine(original,steps);
    auto t1=chrono::steady_clock::now();
    double nq=0;for(float x:q)nq+=double(x)*x;nq=sqrt(nq);
    // Full original-record scan exactly as the source billion-memory variant.
    // Only the score function differs to preserve the real VIBE metric.
    vector<Record> heap;heap.reserve(k);
    // Top-k heap identical to the unoptimised native HingeZero scanner.
    auto consider=[&](Record candidate) {
      if(heap.size()<size_t(k)) {
         heap.push_back(candidate);
         push_heap(heap.begin(),heap.end(),better);
      } else if(better(candidate,heap.front())) {
         pop_heap(heap.begin(),heap.end(),better);
         heap.back()=candidate;
         push_heap(heap.begin(),heap.end(),better);
      }
    };
    if(metric==1 && coarse_index) {
        if(nprobe<1 || nprobe>coarse_index->nlist)
            throw runtime_error("invalid C++20 candidate probe budget");
        const DotImpl scorer=select_dot_ip();
        const float* query=q.data();
        const auto order=coarse_index->probe_order(query);
        int scanned=0;
        // Full exact float32 inner-product scoring within selected clusters.
        // Increase probed clusters if needed to return exactly k records.
        for(int c:order){
            const uint32_t start=coarse_index->offsets[c];
            const uint32_t finish=coarse_index->offsets[c+1];
            for(uint32_t j=start;j<finish;++j){
                const float* row=coarse_index->packed.data()+size_t(j)*d;
                consider(Record{coarse_index->ids[j],scorer(row,query,d)});
            }
            ++scanned;
            if(scanned>=nprobe && heap.size()>=size_t(k))break;
        }
    } else {
        for(uint64_t id=0;id<count;++id) {
            consider(Record{int32_t(id),score(bank+size_t(id)*d,q,nq)});
        }
    }
    sort(heap.begin(),heap.end(),better);
    for(int i=0;i<k;++i){
      out[i]=heap[i].id;
      Vec tagged(bank+size_t(out[i])*d,bank+size_t(out[i]+1)*d);
      if(!verify(out[i],tagged.data()))
         throw runtime_error("tagged-state witness rejected result");
    }
    auto t2=chrono::steady_clock::now();
    if(stats){
      stats[0]=chrono::duration<double,milli>(t1-t0).count();
      stats[1]=chrono::duration<double,milli>(t2-t1).count();
      stats[2]=double(k);
    }
 }
};
}
extern "C"{
int hz20_abi()noexcept{return 1;}
const char* hz20_error()noexcept{return last_error.c_str();}
void* hz20_open(const char*path,uint64_t n,int d,int metric,int field)noexcept{
  try{last_error.clear();return new Memory(path,n,d,metric,field);}
  catch(const exception&ex){last_error=ex.what();return nullptr;}
}
int hz20_query(void*handle,const float*q,int32_t k,int steps,int nprobe,
               int32_t*ids,double*stats)noexcept{
  try{
    if(!handle)throw runtime_error("null memory handle");
    static_cast<Memory*>(handle)->query(q,k,steps,nprobe,ids,stats);
    return 0;
  }catch(const exception&ex){last_error=ex.what();return -1;}
}
int hz20_refine(void*handle,const float*q,int steps,float*out)noexcept{
  try{
    if(!handle||!out)throw runtime_error("null refine handle");
    Vec v=static_cast<Memory*>(handle)->refine(q,steps);
    memcpy(out,v.data(),v.size()*4);return 0;
  }catch(const exception&ex){last_error=ex.what();return -1;}
}
int hz20_verify(void*handle,int64_t id,const float*state)noexcept{
  if(!handle)return -1;
  return static_cast<Memory*>(handle)->verify(id,state)?1:0;
}
void hz20_close(void*handle)noexcept{delete static_cast<Memory*>(handle);}
int hz20_append(const char*path,uint64_t previous,int dimension,
                const float*values,uint64_t extra)noexcept{
 try{
   if(!path||!values||dimension<1||extra<1||
      previous>uint64_t(INT32_MAX)-extra)
      throw runtime_error("invalid append");
   fs::path file(path);
   uint64_t rowbytes=uint64_t(dimension)*4;
   uint64_t size=fs::exists(file)?fs::file_size(file):0;
   if(size!=previous*rowbytes)
      throw runtime_error("append refused: incorrect original file size");
   uint64_t count=uint64_t(dimension)*extra;
   for(uint64_t j=0;j<count;++j)if(!isfinite(values[j]))
      throw runtime_error("nonfinite memory append");
   ofstream out(file,ios::binary|ios::app);
   if(!out)throw runtime_error("cannot open bank for append");
   constexpr uint64_t chunk=uint64_t(1)<<20;
   for(uint64_t i=0;i<count;){
     uint64_t take=min(chunk,count-i);
     out.write(reinterpret_cast<const char*>(values+i),streamsize(take*4));
     if(!out)throw runtime_error("failed to append records");
     i+=take;
   }
   out.flush();if(!out)throw runtime_error("failed to flush appended records");
   out.close();
   if(fs::file_size(file)!=(previous+extra)*rowbytes)
      throw runtime_error("append size mismatch");
   return 0;
 }catch(const exception&ex){last_error=ex.what();return -1;}
}
}
