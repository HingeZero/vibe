// HingeZero C++20 research ANN memory navigator.
// Deterministic, corpus-only, original float32 records and witnesses retained.
// Single VIBE algorithm, graph ef_search controls the quality/speed tradeoff.
// Inspired by published navigable small-world indexing principles; independent
// native implementation, not FAISS/HNSWlib or a claimed novel graph algorithm.
struct GraphANN {
  struct Node {
    std::vector<std::vector<int32_t>> edges;
  };
  struct Match {
    float score;
    int32_t id;
  };
  const float *bank;
  int d;
  uint64_t n;
  int m0=32, m=16, ef_build=96;
  int32_t entry=0, max_layer=0;
  std::vector<Node> nodes;
  std::vector<float> norms;
  std::vector<float> lift;
  double radius_sq=0;
  static uint64_t mix(uint64_t &seed){
     uint64_t z=(seed+=0x9e3779b97f4a7c15ULL);
     z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;
     z=(z^(z>>27))*0x94d049bb133111ebULL;
     return z^(z>>31);
  }
  static float scalar_score(const float*a,const float*b,int d){
    float s=0.0f;
    for(int i=0;i<d;++i)s+=a[i]*b[i];
    return s;
  }
#if defined(__x86_64__) && defined(__GNUC__)
  __attribute__((target("avx2")))
  static float vector_score(const float*a,const float*b,int d){
    __m256 s0=_mm256_setzero_ps(),s1=s0,s2=s0,s3=s0;
    int i=0;
    for(;i+31<d;i+=32){
       s0=_mm256_add_ps(s0,_mm256_mul_ps(
           _mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i)));
       s1=_mm256_add_ps(s1,_mm256_mul_ps(
           _mm256_loadu_ps(a+i+8),_mm256_loadu_ps(b+i+8)));
       s2=_mm256_add_ps(s2,_mm256_mul_ps(
           _mm256_loadu_ps(a+i+16),_mm256_loadu_ps(b+i+16)));
       s3=_mm256_add_ps(s3,_mm256_mul_ps(
           _mm256_loadu_ps(a+i+24),_mm256_loadu_ps(b+i+24)));
    }
    s0=_mm256_add_ps(_mm256_add_ps(s0,s1),_mm256_add_ps(s2,s3));
    alignas(32) float result[8];
    _mm256_store_ps(result,s0);
    float v=0;
    for(float f:result)v+=f;
    for(;i<d;++i)v+=a[i]*b[i];
    return v;
  }
#endif
  using FastDot=float(*)(const float*,const float*,int);
  static FastDot select_fast_dot(){
#if defined(__x86_64__) && defined(__GNUC__)
    if(__builtin_cpu_supports("avx2"))return &vector_score;
#endif
    return &scalar_score;
  }
  static float fast_score(const float*a,const float*b,int d){
     static const FastDot scorer=select_fast_dot();
     return scorer(a,b,d);
  }
  float score(const float*q,int32_t id,int32_t build_src=-1) const{
    // MIPS-to-sphere lift for insertion only. Each y=(x,sqrt(R^2-||x||^2))
    // has ||y||=R. Build graph by neighbour similarity in lifted space.
    // Query is (q,0), so q.y = q.x; true VIBE MIPS ranking is preserved.
    float s=fast_score(q,bank+size_t(id)*d,d);
    if(build_src>=0)s+=lift[build_src]*lift[id];
    return s;
  }
  float diversity(const int32_t a,const int32_t b)const {
    const float xy=fast_score(bank+size_t(a)*d,bank+size_t(b)*d,d);
    return float((double(xy)+double(lift[a])*lift[b])/radius_sq);
  }
  int sampled_level(uint64_t &seed) const {
    // Exponentially decreasing number of upper-layer vertices.
    uint64_t bits=mix(seed)>>11;
    double u=(double(bits)+1.0)/(9007199254740992.0+1.0);
    int lvl=int(-std::log(u)*(1.0/std::log(double(m))));
    return std::min(lvl,8);
  }
  int32_t walk(const float*q,int32_t start,int layer,int32_t build_src=-1)const {
     int32_t current=start;
     float best=score(q,current,build_src);
     bool changed=true;
     while(changed){
       changed=false;
       const auto &neigh=nodes[current].edges[layer];
       for(int32_t cand:neigh){
         float s=score(q,cand,build_src);
         if(s>best || (s==best && cand<current)){
           best=s;current=cand;changed=true;
         }
       }
     }
     return current;
  }
  std::vector<Match> explore(const float*q,int32_t start,int ef,int layer,int32_t build_src=-1)const {
     // VIBE uses one benchmark core. Stamp visited per-thread to keep
     // the query path safe and avoid O(n) reset per query.
     static thread_local std::vector<uint32_t> seen;
     static thread_local uint32_t epoch=0;
     if(seen.size()<n)seen.resize(size_t(n),0u);
     if(++epoch==0){std::fill(seen.begin(),seen.end(),0u);epoch=1;}
     using Item=std::pair<float,int32_t>;
     std::priority_queue<Item> frontier;
     std::priority_queue<Item,std::vector<Item>,std::greater<Item>> best;
     const float init=score(q,start,build_src);
     frontier.emplace(init,start);
     best.emplace(init,start);
     seen[start]=epoch;
     while(!frontier.empty()){
       Item cur=frontier.top();frontier.pop();
       if(best.size()>=size_t(ef) && cur.first<best.top().first)break;
       const auto &adj=nodes[cur.second].edges[layer];
       for(int32_t id:adj){
         if(seen[id]==epoch)continue;
         seen[id]=epoch;
         const float sim=score(q,id,build_src);
         if(best.size()<size_t(ef)||sim>best.top().first){
           frontier.emplace(sim,id);
           best.emplace(sim,id);
           if(best.size()>size_t(ef))best.pop();
         }
       }
     }
     std::vector<Match> out;
     out.reserve(best.size());
     while(!best.empty()){
       const auto [sim,id]=best.top();best.pop();
       out.push_back(Match{sim,id});
     }
     std::sort(out.begin(),out.end(),[](const Match&a,const Match&b){
       return a.score==b.score?a.id<b.id:a.score>b.score;
     });
     return out;
  }
  std::vector<int32_t> select_neighbors(int32_t owner,
                       const std::vector<Match>& pool,int max_degree) const {
     std::vector<int32_t> selected;
     selected.reserve(max_degree);
     // Diversity heuristic: prefer different vector directions; avoid a
     // dense local clique of near-duplicates while retaining strong links.
     for(const Match& cand:pool){
       if(cand.id==owner)continue;
       bool redundant=false;
       for(int32_t prev:selected){
         if(diversity(cand.id,prev)>0.99f) {
           redundant=true;break;
         }
       }
       if(!redundant){
         selected.push_back(cand.id);
         if(int(selected.size())>=max_degree)return selected;
       }
     }
     for(const Match& cand:pool){
       if(cand.id==owner)continue;
       if(std::find(selected.begin(),selected.end(),cand.id)==selected.end()){
         selected.push_back(cand.id);
         if(int(selected.size())>=max_degree)break;
       }
     }
     return selected;
  }
  void connect(int32_t from,int32_t to,int layer){
     auto &neigh=nodes[from].edges[layer];
     if(std::find(neigh.begin(),neigh.end(),to)!=neigh.end())return;
     neigh.push_back(to);
     const int cap=layer==0?m0:m;
     if(int(neigh.size())<=cap)return;
     std::vector<Match> pool;
     pool.reserve(neigh.size());
     const float*q=bank+size_t(from)*d;
     for(int32_t id:neigh)pool.push_back(Match{score(q,id,from),id});
     std::sort(pool.begin(),pool.end(),[](const Match&a,const Match&b){
       return a.score==b.score?a.id<b.id:a.score>b.score;
     });
     // Keep strongest edges under the memory budget. Reverse edges
     // can be pruned; search still has an entrypoint and multi-layer links.
     auto selected=select_neighbors(from,pool,cap);
     neigh.swap(selected);
  }
  GraphANN(const float*original,uint64_t count,int dimension):
       bank(original),d(dimension),n(count),nodes(count),norms(count),lift(count) {
     if(!bank||n<1||d<1||n>uint64_t(INT32_MAX))
       throw std::runtime_error("Invalid HingeZero ANN corpus");
     for(uint64_t i=0;i<n;++i){
       double mag=0;
       const float*p=bank+size_t(i)*d;
       for(int j=0;j<d;++j)mag+=double(p[j])*p[j];
       norms[i]=float(std::sqrt(mag));
       radius_sq=std::max(radius_sq,mag);
     }
     radius_sq*=1.00001; // avoid negative lift due to float rounding
     for(uint64_t i=0;i<n;++i) {
        lift[i]=float(std::sqrt(std::max(0.0,radius_sq-double(norms[i])*norms[i])));
     }
     uint64_t seed=0x8c1ee0d41239b1a7ULL;
     max_layer=sampled_level(seed);
     nodes[0].edges.resize(max_layer+1);
     entry=0;
     for(int32_t id=1;uint64_t(id)<n;++id){
       const int layer=sampled_level(seed);
       nodes[id].edges.resize(layer+1);
       const float*q=bank+size_t(id)*d;
       int32_t cur=entry;
       // Greedy skip-layers cannot access the unconnected new node.
       for(int l=max_layer;l>layer;--l)cur=walk(q,cur,l,id);
       for(int l=std::min(layer,max_layer);l>=0;--l){
         const auto pool=explore(q,cur,ef_build,l,id);
         if(pool.empty())throw std::runtime_error("HingeZero empty ANN build layer");
         const auto chosen=select_neighbors(id,pool,l==0?m0:m);
         for(int32_t other:chosen){
            connect(id,other,l);
            connect(other,id,l);
         }
         cur=pool[0].id;
       }
       if(layer>max_layer){
         entry=id;
         max_layer=layer;
       }
     }
  }
  std::vector<Match> candidates(const float*q,int ef)const{
    if(ef<1 || ef>32768)
      throw std::runtime_error("Invalid HingeZero ANN ef_search budget");
    int32_t cur=entry;
    for(int l=max_layer;l>0;--l)cur=walk(q,cur,l);
    return explore(q,cur,ef,0);
  }
};
