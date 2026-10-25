// HingeZero exact-identity associative memory index, 2026.
// Distinct from the VIBE nearest-neighbour ranking route.
// Keys reference immutable mmap bytes; a collision compares ALL 4*d bytes.
struct ExactMemoryIndex {
    struct Hash {
        size_t operator()(std::string_view key) const noexcept {
            uint64_t h=14695981039346656037ULL;
            for(unsigned char x:key) {
                h^=x;
                h*=1099511628211ULL;
            }
            return static_cast<size_t>(h);
        }
    };
    // string_view equality is length + original bytes, never hash-only.
    std::unordered_map<std::string_view,int32_t,Hash> first_id;
    size_t stride=0;
    uint64_t n=0;
    ExactMemoryIndex(const float* bank,uint64_t count,int dims)
       :stride(size_t(dims)*sizeof(float)),n(count) {
        if(!bank||!count||dims<1)throw std::runtime_error("Invalid exact-memory index");
        first_id.reserve(size_t(count));
        const char* bytes=reinterpret_cast<const char*>(bank);
        for(uint64_t i=0;i<count;++i) {
           // Duplicate vectors deterministically resolve to their oldest
           // memory ID. A separate ID lookup still sees every record.
           first_id.emplace(std::string_view(bytes+size_t(i)*stride,stride),
                            int32_t(i));
        }
    }
    int32_t lookup(const float*cue) const {
       if(!cue)throw std::runtime_error("Null exact-memory key");
       auto key=std::string_view(reinterpret_cast<const char*>(cue),stride);
       auto it=first_id.find(key);
       return it==first_id.end()?-1:it->second;
    }
};
