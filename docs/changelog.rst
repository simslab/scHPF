.. _changelog:

*********
Changelog
*********

0.3.0
=====

- Refactor so loss can be an arbitrary function
- Fix bugs in and expand options for projetion
- prep-like CLI to prepare data for projection onto a trained model
- cellscore fraction file for score CLI
- Verbose option for load_txt
- Update options for validation cells & selection
- Version as an object attribute
- Handle change in scipy API
- new GENCODE files
- use pathlib for paths
- (feature request) options to specify a and c from the train CLI
- Documentation with ReadTheDocs


0.2.4
=====
- Emergency patch preprocessing error for loom files. Also fixed an errant test.
  Not really enough to justify a new release but fixed a pretty
  irritating/embarrassing error.  

0.2.3
=====
- fix no split on dot bug
- Max pairwise table + default max pairwise in score
- Note about ld.so error
- Fix max pairwise second greatest bug
- Some integration tests


0.2.2
=====
- partial test suite
- max pairwise test for gene overlap
- faster preprocessing of larage text files
- refactor preprocessing and training control flow out of CLI
- move load and save methods outside of scHPF object


0.2.1
=====
- Slight speedup during inference for Xphi
- Fix bug (occurred first in 0.2.0-alpha) that occurs when genes in
  whitespace-delim input to prep that have no counts


0.2.0
=====
Numba implmentation with scikit-learn-like API


0.1.0
=====
- Tensorflow implementation

