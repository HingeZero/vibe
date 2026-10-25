// Research-only exact certified PCA cascade for zero-step HingeZero.
// Copyright © 2026 HingeZero/David Duffy research prototype.
// Included into the HingeZero native kernel's anonymous namespace.
// No query ground truth, evaluation labels, FAISS or approximate shortlist.
struct CascadeIndex {
    static constexpr int STAGES=7;
    static constexpr int WIDTH[STAGES]={8,8,16,32,32,16,16};
    int d;
    uint64_t n;
    vector<double> mean;               // original-coordinate sample mean
    vector<double> basis;              // ordered orthogonal PCA vectors
    vector<double> scale;              // float32->int8 per-record factor
    vector<double> original_norm;      // exact centered-record norm
    vector<vector<int8_t>> blocks;     // stage-major, CPU cache-friendly
    vector<vector<double>> quant_error, residual; // conservative norms

    static void update_top(vector<Record>&heap,int k,Record r) {
        if(heap.size()<size_t(k)){
            heap.push_back(r);
            push_heap(heap.begin(),heap.end(),better);
        }else if(better(r,heap.front())){
            pop_heap(heap.begin(),heap.end(),better);
            heap.back()=r;
            push_heap(heap.begin(),heap.end(),better);
        }
    }
    CascadeIndex(const float*bank,uint64_t rows,int dim)
       :d(dim),n(rows),mean(dim,0),basis(size_t(dim)*dim,0),
        scale(rows),original_norm(rows),blocks(STAGES),
        quant_error(STAGES),residual(STAGES) {
      if(d!=128 || n<1)throw runtime_error("cascade requires 128-dimensional corpus");
      const int sample=int(min<uint64_t>(n,2048));
      vector<double> cov(size_t(d)*d,0.0);
      // Corpus-only training; a deterministic, representative strided sample.
      for(int s=0;s<sample;++s){
          uint64_t row=(uint64_t(s)*n)/sample;
          const float* v=bank+size_t(row)*d;
          for(int j=0;j<d;++j)mean[j]+=double(v[j])/sample;
      }
      vector<double> centered(d);
      for(int s=0;s<sample;++s){
          uint64_t row=(uint64_t(s)*n)/sample;
          const float*v=bank+size_t(row)*d;
          for(int j=0;j<d;++j)centered[j]=double(v[j])-mean[j];
          for(int i=0;i<d;++i) {
              const double a=centered[i];
              for(int j=i;j<d;++j)cov[size_t(i)*d+j]+=a*centered[j]/sample;
          }
      }
      for(int i=0;i<d;++i)for(int j=0;j<i;++j)
          cov[size_t(i)*d+j]=cov[size_t(j)*d+i];
      vector<double> eigvec(size_t(d)*d,0);
      for(int i=0;i<d;++i)eigvec[size_t(i)*d+i]=1.0;
      // Jacobi eigenbasis; orthogonality is maintained at every iteration.
      for(int sweep=0;sweep<9;++sweep){
         for(int p=0;p<d-1;++p)for(int q=p+1;q<d;++q){
            const double apq=cov[size_t(p)*d+q];
            if(abs(apq)<1e-14)continue;
            const double app=cov[size_t(p)*d+p],aqq=cov[size_t(q)*d+q];
            const double tau=(aqq-app)/(2.0*apq);
            const double t=copysign(1.0,tau)/(abs(tau)+sqrt(1.0+tau*tau));
            const double cosine=1.0/sqrt(1.0+t*t),sine=t*cosine;
            cov[size_t(p)*d+p]=app-t*apq;
            cov[size_t(q)*d+q]=aqq+t*apq;
            cov[size_t(p)*d+q]=cov[size_t(q)*d+p]=0.0;
            for(int k=0;k<d;++k) {
               if(k==p || k==q)continue;
               const double apk=cov[size_t(k)*d+p], aqk=cov[size_t(k)*d+q];
               cov[size_t(k)*d+p]=cov[size_t(p)*d+k]=cosine*apk-sine*aqk;
               cov[size_t(k)*d+q]=cov[size_t(q)*d+k]=sine*apk+cosine*aqk;
            }
            for(int k=0;k<d;++k){
               const double vkp=eigvec[size_t(k)*d+p],vkq=eigvec[size_t(k)*d+q];
               eigvec[size_t(k)*d+p]=cosine*vkp-sine*vkq;
               eigvec[size_t(k)*d+q]=sine*vkp+cosine*vkq;
            }
         }
      }
      vector<int> order(d);
      iota(order.begin(),order.end(),0);
      sort(order.begin(),order.end(),[&](int a,int b){
        return cov[size_t(a)*d+a]>cov[size_t(b)*d+b];
      });
      for(int i=0;i<d;++i)for(int j=0;j<d;++j)
         basis[size_t(i)*d+j]=eigvec[size_t(j)*d+order[i]];
      for(int s=0;s<STAGES;++s) {
          blocks[s].resize(size_t(n)*WIDTH[s]);
          quant_error[s].resize(n);
          residual[s].resize(n);
      }
      vector<double> y(d);
      for(uint64_t id=0;id<n;++id) {
         const float*v=bank+size_t(id)*d;
         double total=0.0;
         for(int j=0;j<d;++j) {
             centered[j]=double(v[j])-mean[j];
             total+=centered[j]*centered[j];
         }
         original_norm[id]=sqrt(total);
         double maxabs=0;
         for(int j=0;j<d;++j){
             double score=0;
             for(int z=0;z<d;++z)
                 score+=basis[size_t(j)*d+z]*centered[z];
             y[j]=score;
             maxabs=max(maxabs,abs(score));
         }
         double factor=(maxabs>0? maxabs/127.0:1.0);
         scale[id]=factor;
         double projNormSq=0,quantErrorSq=0;
         int pos=0;
         for(int s=0;s<STAGES;++s) {
            for(int j=0;j<WIDTH[s];++j) {
                const double value=y[pos];
                long long z=llround(value/factor);
                z=max(-127LL,min(127LL,z));
                blocks[s][size_t(id)*WIDTH[s]+j]=int8_t(z);
                const double approx=double(z)*factor;
                const double err=value-approx;
                quantErrorSq+=err*err;
                projNormSq+=value*value;
                ++pos;
            }
            // Include guard for Jacobi/basis and finite rounding error.
            quant_error[s][id]=sqrt(quantErrorSq) + 1e-11*(1+original_norm[id]);
            residual[s][id]=sqrt(max(0.0,total-projNormSq))
                                   +1e-10*(1+original_norm[id]);
         }
      }
    }

    void search(const float*bank,const float*q,int k,
                vector<Record>&exact_heap,DotImpl scorer,double*stats)const {
        if(d!=128)throw runtime_error("cascade query dimensions differ");
        vector<double> rotated(d);
        double meanDot=0,queryNormSq=0;
        for(int i=0;i<d;++i) {
            meanDot+=mean[i]*q[i];
            queryNormSq+=double(q[i])*q[i];
            double v=0;
            for(int j=0;j<d;++j)v+=basis[size_t(i)*d+j]*double(q[j]);
            rotated[i]=v;
        }
        // Norms for each subspace; true-query transform is orthonormal.
        double suffix[STAGES],prefix[STAGES];
        double total=0;
        for(double v:rotated)total+=v*v;
        double sofar=0;
        int offset=0;
        for(int s=0;s<STAGES;++s){
            for(int j=0;j<WIDTH[s];++j){
               sofar+=rotated[offset+j]*rotated[offset+j];
            }
            offset+=WIDTH[s];
            prefix[s]=sqrt(sofar);
            suffix[s]=sqrt(max(0.0,total-sofar));
        }
        vector<double> approx(n,0.0);
        vector<Record> seed;
        const int seeds=int(min<uint64_t>(1024,n));
        seed.reserve(seeds);
        // First eight principal axes are contiguous and require ~1.5MB.
        const auto&first=blocks[0];
        for(uint64_t id=0;id<n;++id){
            const int8_t*code=first.data()+size_t(id)*8;
            double partial=0;
            for(int j=0;j<8;++j)partial+=double(code[j])*rotated[j];
            approx[id]=scale[id]*partial;
            update_top(seed,seeds,Record{int32_t(id),approx[id]});
        }
        vector<uint8_t> verified(n,0);
        int exact_count=0;
        for(const Record&r:seed){
           update_top(exact_heap,k,
              Record{r.id,scorer(bank+size_t(r.id)*d,q,d)});
           verified[r.id]=1;
           ++exact_count;
        }
        vector<int32_t> survivors;
        survivors.reserve(n);
        // Certified progressive elimination; no heuristic shortlist.
        // The bound is approximate rotated dot + quantization residual
        // plus orthogonal unexamined-tail norm, with an outward guard.
        for(uint64_t id=0;id<n;++id) {
            const double up=meanDot+approx[id]
                 +quant_error[0][id]*prefix[0]
                 +residual[0][id]*suffix[0]
                 +1e-6*(1+original_norm[id]*sqrt(queryNormSq));
            if(exact_heap.size()<size_t(k) || up>=exact_heap.front().score)
                survivors.push_back(int32_t(id));
        }
        offset=8;
        for(int s=1;s<STAGES && !survivors.empty();++s) {
           const int width=WIDTH[s];
           const auto&codes=blocks[s];
           vector<int32_t> remaining;
           remaining.reserve(survivors.size());
           for(int32_t id:survivors){
               const int8_t*code=codes.data()+size_t(id)*width;
               double sum=0;
               for(int j=0;j<width;++j)
                   sum+=double(code[j])*rotated[offset+j];
               approx[id]+=scale[id]*sum;
               const double upper=meanDot+approx[id]
                    +quant_error[s][id]*prefix[s]
                    +residual[s][id]*suffix[s]
                    +1e-6*(1+original_norm[id]*sqrt(queryNormSq));
               if(exact_heap.size()<size_t(k) || upper>=exact_heap.front().score)
                   remaining.push_back(id);
           }
           survivors.swap(remaining);
           offset+=width;
        }
        for(int32_t id:survivors) {
           if(verified[id])continue;
           update_top(exact_heap,k,
               Record{id,scorer(bank+size_t(id)*d,q,d)});
           ++exact_count;
        }
        if(stats)stats[2]=exact_count;
    }
};
