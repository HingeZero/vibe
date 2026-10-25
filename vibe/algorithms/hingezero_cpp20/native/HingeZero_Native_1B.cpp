// HingeZero Native C++20: David Duffy's witness-constrained memory architecture.
// C++20 packed, disk-streamed adaptation for BIPOLAR 128-dimensional patterns only.
// No external Hamming/candidate-reranking stage: all P rows are scored directly
// against the HingeZero-refined query (equivalent to normalized dot/cosine at d=128).
// The field uses the first min(P,256) rows, corresponding to uniform zero evidence.
// Precision/storage adaptation: this is NOT a byte-for-byte port of the NumPy core.
// Copyright (c) 2026 - research implementation; HingeZero architecture: David Duffy.
#include <array>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
namespace fs=std::filesystem;
constexpr int D=128, FIELD=256, STEPS=8;
constexpr float ALPHA=4.5f, EPS=.10f, LAMBDA=.02f;
constexpr float SCALE=0.08838834764831845f; // 1/sqrt(128)
constexpr uint64_t BASESEED=0x176e2eadcafe5315ULL;
struct Pattern { uint64_t lo,hi; };
static_assert(sizeof(Pattern)==16);
using Vec=array<float,D>;
using Clock=chrono::steady_clock;

auto elapsed_ms(Clock::time_point start) { return chrono::duration<double,milli>(Clock::now()-start).count(); }
uint64_t splitmix(uint64_t& x) { x+=0x9E3779B97F4A7C15ULL; auto z=x; z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31); }
Pattern pattern(uint64_t id) { uint64_t s=BASESEED ^ (id*0xD1342543DE82EF95ULL); return {splitmix(s),splitmix(s)}; }
Vec decode(Pattern p){ Vec v{}; for(int j=0;j<64;++j) v[j]=((p.lo>>j)&1ULL)?SCALE:-SCALE; for(int j=0;j<64;++j) v[64+j]=((p.hi>>j)&1ULL)?SCALE:-SCALE; return v; }
void unit(Vec& v){ double s=0; for(float f:v) s+=double(f)*f; if(s>1e-24) {auto scale=float(1.0/sqrt(s)); for(auto& x:v) x*=scale;} else v.fill(0);}
float dot(Vec const&a,Vec const&b){float s=0; for(int i=0;i<D;i++)s+=a[i]*b[i]; return s;}
struct Bank{
 fs::path path; uint64_t count; ifstream in;
 Bank(fs::path p,uint64_t n):path(std::move(p)),count(n),in(path,ios::binary){
   if(!in) throw runtime_error("cannot open bank file");
   if(fs::file_size(path)!=count*sizeof(Pattern))throw runtime_error("bank size mismatch");
 }
 Pattern read(uint64_t i){if(i>=count)throw runtime_error("out of range"); in.clear(); in.seekg(static_cast<streamoff>(i*16)); Pattern p{};in.read(reinterpret_cast<char*>(&p),16);if(!in)throw runtime_error("short read");return p;}
 void rewind(){in.clear();in.seekg(0);}
};
struct State {uint64_t id; Vec value;};
class Witness{
 Bank& bank; vector<Vec> field;
 public:
 explicit Witness(Bank&b):bank(b){ for(uint64_t i=0;i<min<uint64_t>(bank.count,FIELD);++i)field.push_back(decode(bank.read(i))); }
 Vec refine(Vec query) const {
   unit(query);
   // Uniform evidence, consistent with original core when all weights are zero.
   if(dot(query,query)<1e-15f){Vec avg{};for(auto const& m:field)for(int j=0;j<D;j++)avg[j]+=m[j];unit(avg);query=avg;}
   for(int t=0;t<STEPS;t++){
     Vec fb{}; double denom=1e-12;
     for(auto const&m:field){float r=dot(query,m);float w=tanh(r)+ALPHA*tanh(2*r);denom+=fabs(double(w)); for(int j=0;j<D;j++)fb[j]+=m[j]*w;}
     for(int j=0;j<D;j++)query[j]=(1-LAMBDA)*query[j]+EPS*float(fb[j]/denom);
     unit(query);
   }
   return query;
 }
 bool accepts(State const& s){if(s.id>=bank.count)return false;return s.value==decode(bank.read(s.id));}
 State observe_tag(State const&s){if(!accepts(s))throw runtime_error("invalid tagged state");return s;} // preservation branch before refinement
 bool verify(State const&s){if(!accepts(s))return false;State again=observe_tag(s);return again.id==s.id&&again.value==s.value;}
 // Full-bank native cosine decoder. Streaming preserves original argmax/tie ordering.
 State observe(Vec const&cue,double&scan_ms,double&refine_ms,bool bypass=false){
   auto t=Clock::now();Vec q=bypass?cue:refine(cue); if(bypass)unit(q); refine_ms=elapsed_ms(t);
   // For bipolar 128-bit memories, score each row with the exact signed sum of
   // the 128 refined coordinates. Byte lookup is an arithmetic acceleration of
   // this dot product. The entire bank is scored, not prefiltered by Hamming.
   float lut[16][256];
   for(int pos=0;pos<16;++pos){
     float base=0;for(int j=0;j<8;j++)base-=q[pos*8+j]*SCALE;
     for(int b=0;b<256;b++){
       float acc=base;for(int j=0;j<8;j++)if(b&(1<<j))acc+=2*SCALE*q[pos*8+j];
       lut[pos][b]=acc;
     }
   }
   constexpr size_t BLOCK=1<<16; vector<Pattern> buf(BLOCK);bank.rewind();
   double best=-numeric_limits<double>::infinity();uint64_t winner=0,idx=0;
   t=Clock::now();
   while(idx<bank.count){
     auto n=size_t(min<uint64_t>(BLOCK,bank.count-idx));
     bank.in.read(reinterpret_cast<char*>(buf.data()),streamsize(n*16));
     if(bank.in.gcount()!=streamsize(n*16))throw runtime_error("short bank stream");
     for(size_t k=0;k<n;k++){
       auto p=buf[k]; double score=0;
       for(int b=0;b<8;b++){score+=lut[b][(p.lo>>(8*b))&255]; score+=lut[b+8][(p.hi>>(8*b))&255];}
       if(score>best){best=score;winner=idx+k;}
     }
     idx+=n;
   }
   scan_ms=elapsed_ms(t);
   return State{winner,decode(bank.read(winner))};
 }
};

void create(fs::path path,uint64_t count){
 if(fs::exists(path)) throw runtime_error("refusing to overwrite existing bank");
 constexpr size_t BLOCK=1<<16;vector<Pattern> buf(BLOCK);ofstream out(path,ios::binary|ios::trunc);if(!out)throw runtime_error("cannot create bank");
 auto t=Clock::now(); for(uint64_t i=0;i<count;){size_t n=size_t(min<uint64_t>(BLOCK,count-i));for(size_t k=0;k<n;k++)buf[k]=pattern(i+k);out.write(reinterpret_cast<char*>(buf.data()),streamsize(n*16));if(!out)throw runtime_error("write failed");i+=n; if(count>=100000000&&i%100000000<(1<<16))cerr<<"written "<<i<<" / "<<count<<"\n";}
 out.flush();if(!out)throw runtime_error("flush failed");out.close();cout<<"{\"generated\":"<<count<<",\"bytes\":"<<fs::file_size(path)<<",\"seconds\":"<<elapsed_ms(t)/1000<<"}\n";
}
// Deterministic damaged test cues; all vectors are complete 128 coordinates.
Vec cue(Pattern p,string regime,uint64_t target){
 Vec v=decode(p);uint64_t seed=BASESEED^target^0xba5edULL;
 auto seq=[&](){return splitmix(seed);};
 array<int,D> indices; iota(indices.begin(),indices.end(),0);
 for(int j=D-1;j>0;--j) swap(indices[j],indices[seq()%(j+1)]);
 if(regime=="exact") return v;
 if(regime=="flip20" ||regime=="flip35"){
   int n=regime=="flip20"?26:45;
   for(int j=0;j<n;j++)v[indices[j]]=-v[indices[j]];
 } else if(regime=="missing50"){
   for(int j=0;j<64;j++)v[indices[j]]=0;
 } else if(regime=="severe"){
   for(int j=32;j<D;j++)v[indices[j]]=0;
   for(int j=0;j<6;j++)v[indices[j]]=-v[indices[j]];
 } else if(regime=="multi4"){
   Vec sum{};
   for(int k=0;k<4;k++){
     auto tmp=decode(p);auto ids=indices;
     for(int j=D-1;j>0;--j)swap(ids[j],ids[seq()%(j+1)]);
     for(int j=32;j<D;j++)tmp[ids[j]]=0;
     for(int j=0;j<6;j++)tmp[ids[j]]=-tmp[ids[j]];
     for(int j=0;j<D;j++)sum[j]+=tmp[j];
   }
   v=sum;
 } else throw runtime_error("unknown cue regime");
 unit(v);return v;
}

void append(fs::path path,uint64_t old_count,uint64_t additional){
 if(!fs::exists(path)||fs::file_size(path)!=old_count*sizeof(Pattern))throw runtime_error("append precondition failed: wrong existing size");
 constexpr size_t BLOCK=1<<16;vector<Pattern> buf(BLOCK);
 fstream out(path,ios::binary|ios::in|ios::out);if(!out)throw runtime_error("append open failed");
 out.seekp(0,ios::end); auto t=Clock::now();
 for(uint64_t off=0;off<additional;){
   size_t n=size_t(min<uint64_t>(BLOCK,additional-off));
   for(size_t k=0;k<n;k++)buf[k]=pattern(old_count+off+k);
   out.write(reinterpret_cast<const char*>(buf.data()),streamsize(n*16));
   if(!out)throw runtime_error("append write failed");
   off+=n;
 }
 out.flush();if(!out)throw runtime_error("append flush failed");out.close();
 if(fs::file_size(path)!=(old_count+additional)*sizeof(Pattern))throw runtime_error("append size invalid");
 cout<<"{\"old_count\":"<<old_count<<",\"appended\":"<<additional<<",\"total\":"<<(old_count+additional)<<",\"seconds\":"<<elapsed_ms(t)/1000<<"}\n";
}

void test(fs::path path,uint64_t count,uint64_t target,string regime,bool ablate=false){
 Bank bank(path,count);Witness hz(bank); if(target>=count)throw runtime_error("target missing");
 auto raw=bank.read(target);auto input=cue(raw,regime,target);
 double scan=0,ref=0;auto selected=hz.observe(input,scan,ref,ablate);
 auto tagged=hz.observe_tag(selected);
 bool intact=hz.verify(selected)&&tagged.id==selected.id && tagged.value==selected.value;
 cout<<setprecision(8)<<"{\"load\":"<<count<<",\"target\":"<<target<<",\"regime\":\""<<regime<<"\",\"selected\":"<<selected.id<<",\"correct\":"<<(selected.id==target?"true":"false")<<",\"witness\":"<<(intact?"true":"false")<<",\"refine_ms\":"<<ref<<",\"scan_ms\":"<<scan<<",\"ablation\":"<<(ablate?"true":"false")<<",\"refined_head\":[";
 auto rv=hz.refine(input);for(int j=0;j<6;j++){if(j)cout<<",";cout<<rv[j];}cout<<"]}\n";
}
void parity(fs::path path,uint64_t count,uint64_t target,string regime){Bank bank(path,count);Witness hz(bank);Vec x=cue(bank.read(target),regime,target);auto q=hz.refine(x);cout<<"{\"refined\":[";for(int j=0;j<D;j++){if(j)cout<<",";cout<<setprecision(9)<<q[j];}cout<<"]}\n";}
int main(int argc,char**argv){
 try {
   if(argc<2)throw runtime_error("Usage: hz create FILE COUNT | hz test FILE COUNT TARGET REGIME [--ablate] | hz parity FILE COUNT TARGET REGIME | hz check FILE COUNT TAG");
   string op=argv[1];if(op=="create"&&argc==4){create(argv[2],stoull(argv[3]));return 0;}
   if(op=="append"&&argc==5){append(argv[2],stoull(argv[3]),stoull(argv[4]));return 0;}
   if((op=="test"||op=="parity")&&argc>=6){fs::path path=argv[2];auto count=stoull(argv[3]),target=stoull(argv[4]);string regime=argv[5];if(op=="test")test(path,count,target,regime,argc>=7&&string(argv[6])=="--ablate");else parity(path,count,target,regime);return 0;}
   if(op=="check"&&argc==5){Bank b(argv[2],stoull(argv[3]));Witness hz(b);auto id=stoull(argv[4]);State s{id,decode(b.read(id))};cout<<"{\"tag\":"<<id<<",\"accepted\":"<<(hz.verify(s)?"true":"false")<<"}\n";return 0;}
   throw runtime_error("invalid command args");
 }catch(exception const&e){cerr<<"ERROR: "<<e.what()<<"\n";return 1;}
}
