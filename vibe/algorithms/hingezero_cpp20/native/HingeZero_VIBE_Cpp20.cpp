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
 void query(const float*original,int32_t k,int steps,bool enabled,
            int32_t*out,double*stats)const{
    if(!original||!out||k<1||uint64_t(k)>count)
       throw runtime_error("invalid query");
    auto t0=chrono::steady_clock::now();
    Vec q;
    if(enabled){
       q=refine(original,steps);
       if(enabled==2) {
          // VIBE adaptation only: preserve most original-query evidence.
          // The original binary kernel (enabled==1) remains untouched.
          Vec anchor(original,original+d); unit(anchor);
          for(int j=0;j<d;++j)q[j]=0.95f*anchor[j]+0.05f*q[j];
          unit(q);
       }
       if(metric==0){
         double a=0;for(int j=0;j<d;++j)a+=double(original[j])*original[j];
         for(float&v:q)v*=float(sqrt(a));
       }
    }else{
       q.assign(original,original+d);
       for(float v:q)if(!isfinite(v))throw runtime_error("nonfinite cue");
    }
    auto t1=chrono::steady_clock::now();
    double nq=0;for(float x:q)nq+=double(x)*x;nq=sqrt(nq);
    // Full original-record scan exactly as the source billion-memory variant.
    // Only the score function differs to preserve the real VIBE metric.
    vector<Record> heap;heap.reserve(k);
    for(uint64_t id=0;id<count;++id){
      Record candidate{int32_t(id),score(bank+size_t(id)*d,q,nq)};
      if(heap.size()<size_t(k)){
         heap.push_back(candidate);
         push_heap(heap.begin(),heap.end(),better);
      }else if(better(candidate,heap.front())){
         pop_heap(heap.begin(),heap.end(),better);
         heap.back()=candidate;
         push_heap(heap.begin(),heap.end(),better);
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
int hz20_query(void*handle,const float*q,int32_t k,int steps,int refined,
               int32_t*ids,double*stats)noexcept{
  try{
    if(!handle)throw runtime_error("null memory handle");
    static_cast<Memory*>(handle)->query(q,k,steps,refined!=0,ids,stats);
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
