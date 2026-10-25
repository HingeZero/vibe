# Experimental zero-step single-core HingeZero CPU ANN

The full original float32 memory bank and witness checks remain.
The native HingeZero refinement function is invoked with steps=0.
One CPU benchmark core is used. Eight-bit compact signed signatures
are scanned for a candidate shortlist. The original float32 records
of shortlisted candidate IDs are scored exactly in float64 and returned.

Unlike the submitted certified method, the shortlist does **not**
prove that every true nearest neighbour has been considered.
Only a VIBE score of recall@100=1.0 demonstrates perfect *measured*
overlap on the benchmark. New/unseen queries have no exact guarantee.

The shortlist budget is an explicit VIBE query hyperparameter (128-2048).
Do not train on or cache test queries. The VIBE harness compares the
same dataset at k=100 and reports all budget values, including failures.

Never promote based on unmatched machine QPS or on a few queries.
The upstream PR's submitted branch remains unchanged.
